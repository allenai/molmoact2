from __future__ import annotations

import importlib.util
import json
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import Tensor
from transformers import AutoModelForImageTextToText, AutoProcessor, PreTrainedTokenizerFast

from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

log = logging.getLogger(__name__)


def _resolve_discrete_action_processor_path(
    processor_name_or_path: str,
    *,
    local_files_only: bool,
    hf_token: Optional[str] = None,
) -> Path:
    processor_path = Path(processor_name_or_path).expanduser()
    if processor_path.exists():
        return processor_path
    return Path(
        snapshot_download(
            processor_name_or_path,
            local_files_only=local_files_only,
            token=hf_token,
        )
    )


def _load_discrete_action_processor_from_path(
    processor_path: Path,
    *,
    hf_token: Optional[str] = None,
) -> Any:
    """Load a custom FAST processor without requiring a model config.json."""
    processor_config_path = processor_path / "processor_config.json"
    if not processor_config_path.is_file():
        raise FileNotFoundError(f"Missing processor_config.json at {processor_config_path}")

    processor_config = json.loads(processor_config_path.read_text(encoding="utf-8"))
    auto_processor_ref = (processor_config.get("auto_map") or {}).get("AutoProcessor")
    if not isinstance(auto_processor_ref, str) or "." not in auto_processor_ref:
        raise ValueError(
            f"Processor config at {processor_config_path} does not define a loadable AutoProcessor entry."
        )

    module_name, class_name = auto_processor_ref.rsplit(".", 1)
    module_path = processor_path / f"{module_name}.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Missing processor module {module_path}")

    spec = importlib.util.spec_from_file_location(
        f"cached_hf_action_processor_{hash(str(module_path.resolve()))}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import action processor module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processor_cls = getattr(module, class_name)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        str(processor_path),
        local_files_only=True,
        token=hf_token,
    )
    processor_kwargs = {
        key: value
        for key, value in processor_config.items()
        if key not in {"auto_map", "processor_class"}
    }
    return processor_cls(tokenizer, **processor_kwargs)


def _load_discrete_action_processor(processor_name_or_path: str) -> Any:
    offline_mode = (
        os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
        or os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    hf_token = os.environ.get("HF_ACCESS_TOKEN")
    processor_path = _resolve_discrete_action_processor_path(
        processor_name_or_path,
        local_files_only=offline_mode,
        hf_token=hf_token,
    )
    return _load_discrete_action_processor_from_path(processor_path, hf_token=hf_token)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return _to_text(value[0]) if value else ""
    if isinstance(value, dict):
        for key in ("question", "task", "prompt", "instruction", "text", "content"):
            text = _to_text(value.get(key))
            if text:
                return text
        if "messages" in value:
            return _to_text(value["messages"])
        return ""
    if torch.is_tensor(value):
        if value.numel() == 0:
            return ""
        return str(value.reshape(-1)[0].item())
    array = np.asarray(value)
    if array.ndim == 0:
        return str(array.item())
    if array.size == 0:
        return ""
    return str(array.reshape(-1)[0])


def _find_nested_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for sub_value in value.values():
            found = _find_nested_value(sub_value, key)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_nested_value(item, key)
            if found is not None:
                return found
    return None


def _looks_like_single_observation(value: Any) -> bool:
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return getattr(value, "ndim", 0) > 0
    if isinstance(value, str):
        return True
    if isinstance(value, dict):
        return any(_looks_like_single_observation(subvalue) for subvalue in value.values())
    return False


def _batchify_single_observation(batch: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.unsqueeze(0) if value.ndim > 0 else value.reshape(1)
        elif isinstance(value, np.ndarray):
            out[key] = np.expand_dims(value, axis=0) if value.ndim > 0 else value.reshape(1)
        elif isinstance(value, str):
            out[key] = [value]
        elif isinstance(value, dict):
            out[key] = _batchify_single_observation(value)
        else:
            out[key] = value
    return out


def _maybe_batchify_single_observation_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(batch, dict):
        return batch
    if any(
        str(key).startswith("observation.images.")
        and (torch.is_tensor(value) or isinstance(value, np.ndarray))
        and getattr(value, "ndim", 0) == 3
        for key, value in batch.items()
    ):
        return _batchify_single_observation(batch)
    if "observation.state" in batch:
        value = batch["observation.state"]
        if (torch.is_tensor(value) or isinstance(value, np.ndarray)) and getattr(value, "ndim", 0) == 1:
            return _batchify_single_observation(batch)
    if _looks_like_single_observation(batch) and not any(
        (torch.is_tensor(value) or isinstance(value, np.ndarray)) and getattr(value, "ndim", 0) >= 4
        for value in batch.values()
    ):
        return _batchify_single_observation(batch)
    return batch


def _slice_batch_value(value: Any, idx: int, batch_size: int) -> Any:
    if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size:
        return value[idx]
    if isinstance(value, np.ndarray) and value.ndim > 0 and int(value.shape[0]) == batch_size:
        return value[idx]
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return value[idx]
    return value


def _infer_batch_size(batch: Dict[str, Any]) -> int:
    for value in batch.values():
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            if getattr(value, "ndim", 0) > 0:
                return int(value.shape[0])
        if isinstance(value, (list, tuple)) and value and not isinstance(value[0], (int, float, np.number)):
            return len(value)
    return 1


def _normalize_image(value: Any) -> np.ndarray:
    arr = _to_numpy(value)
    while arr.ndim > 3 and int(arr.shape[0]) == 1:
        arr = arr[0]
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in {3, 4}:
        raise ValueError(f"Unsupported image shape for MolmoAct2 HF policy: {arr.shape}.")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype in (np.float16, np.float32, np.float64):
        if arr.size > 0 and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _append_images(images: List[np.ndarray], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple)) and value and not (
        torch.is_tensor(value[0]) or isinstance(value[0], np.ndarray)
    ):
        for item in value:
            _append_images(images, item)
        return
    images.append(_normalize_image(value))


@dataclass
class MolmoAct2HFInferenceResult:
    style: str
    actions: Optional[torch.Tensor] = None
    generated_token_ids: Optional[torch.Tensor] = None
    depth_bins: Optional[torch.Tensor] = None


class MolmoAct2HFBackend:
    def __init__(self, config: MolmoAct2Config):
        self.config = config
        self.inference_action_mode = str(
            getattr(self.config, "inference_action_mode", "continuous") or "continuous"
        )
        if self.inference_action_mode not in {"continuous", "discrete"}:
            raise ValueError(
                f"Unsupported inference_action_mode={self.inference_action_mode!r}. "
                "Expected one of {'continuous', 'discrete'}."
            )
        self._action_queues: Dict[int, deque] = defaultdict(lambda: deque())
        self._depth_caches: Dict[int, Any] = {}
        self._last_depth_video_codes_by_batch: Dict[int, np.ndarray] = {}
        self._last_model_inference_s = 0.0
        self._last_model_inference_calls = 0
        self._load_hf_model()

    @staticmethod
    def _config_norm_tag(config: Any) -> str:
        return str(getattr(config, "norm_tag", "") or "").strip()

    def _load_hf_model(self) -> None:
        checkpoint_ref = str(self.config.checkpoint_path)
        checkpoint_dir = Path(checkpoint_ref).expanduser()
        checkpoint_source = str(checkpoint_dir) if checkpoint_dir.exists() else checkpoint_ref
        device = torch.device(self.config.device or "cpu")
        self.processor = AutoProcessor.from_pretrained(
            checkpoint_source,
            trust_remote_code=True,
            use_fast=False,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            checkpoint_source,
            trust_remote_code=True,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model = model.to(device)
        self.model.eval()
        self.action_tokenizer = None
        if self.inference_action_mode == "discrete":
            tokenizer_name = str(self.config.discrete_action_tokenizer or "").strip()
            if not tokenizer_name:
                raise ValueError(
                    "MolmoAct2HFPolicy with inference_action_mode='discrete' requires "
                    "`discrete_action_tokenizer` to be provided."
                )
            self.action_tokenizer = _load_discrete_action_processor(tokenizer_name)

    def reset(self) -> None:
        self._action_queues = defaultdict(lambda: deque())
        self._depth_caches = {}
        self._last_depth_video_codes_by_batch = {}
        self._last_model_inference_s = 0.0
        self._last_model_inference_calls = 0

    def get_last_depth_video_codes(self) -> Dict[int, np.ndarray]:
        return {
            int(batch_idx): np.asarray(codes, dtype=np.int64).copy()
            for batch_idx, codes in self._last_depth_video_codes_by_batch.items()
        }

    def get_optim_params(self) -> dict:
        raise NotImplementedError("MolmoAct2 HF policy is inference-only.")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("MolmoAct2 HF policy is inference-only.")

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        batch = _maybe_batchify_single_observation_batch(batch)
        batch_size = _infer_batch_size(batch)
        requested_norm_tag = (
            _to_text(kwargs.get("norm_tag"))
            or _to_text(kwargs.get("tag"))
            or self._config_norm_tag(self.config)
        ).strip()
        if not requested_norm_tag:
            raise ValueError("MolmoAct2 HF policy requires `norm_tag` for action inference.")
        requested_num_steps = kwargs.get("num_steps")
        if requested_num_steps is None:
            requested_num_steps = self.config.num_steps
        requested_n_action_steps = kwargs.get("n_action_steps")
        requested_generator = kwargs.get("generator")

        chunks: List[Tensor] = []
        self._last_depth_video_codes_by_batch = {}
        inference_s = 0.0
        for idx in range(batch_size):
            obs = self._slice_observation_batch(batch, idx, batch_size)
            start = time.perf_counter()
            with torch.inference_mode():
                output = self._predict_action_for_observation(
                    obs,
                    batch_idx=idx,
                    norm_tag=requested_norm_tag,
                    num_steps=requested_num_steps,
                    n_action_steps=requested_n_action_steps,
                    generator=requested_generator,
                )
            inference_s += time.perf_counter() - start
            actions = output.actions
            if actions.ndim == 2:
                actions = actions.unsqueeze(0)
            chunks.append(actions)

        self._last_model_inference_s = float(inference_s)
        self._last_model_inference_calls = int(batch_size)
        device = torch.device(self.config.device or "cpu")
        return torch.cat(chunks, dim=0).to(device=device, dtype=torch.float32)

    def _slice_observation_batch(self, batch: Dict[str, Any], idx: int, batch_size: int) -> Dict[str, Any]:
        return {
            key: _slice_batch_value(value, idx, batch_size)
            for key, value in batch.items()
        }

    @staticmethod
    def _extract_images(obs: Dict[str, Any]) -> List[np.ndarray]:
        images: List[np.ndarray] = []
        for key, value in obs.items():
            if str(key).startswith("observation.images."):
                _append_images(images, value)
        if not images:
            for key in ("images", "image"):
                if key in obs:
                    _append_images(images, obs[key])
        if not images:
            raise ValueError("No image data found in observation for MolmoAct2 HF policy.")
        return images

    @staticmethod
    def _extract_state(obs: Dict[str, Any]) -> np.ndarray:
        state = obs.get("observation.state")
        if state is None:
            state = obs.get("state")
        if state is None:
            raise ValueError("MolmoAct2 HF policy requires observation.state.")
        arr = np.asarray(_to_numpy(state), dtype=np.float32)
        while arr.ndim > 1 and int(arr.shape[0]) == 1:
            arr = arr[0]
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return arr

    @staticmethod
    def _extract_task(obs: Dict[str, Any]) -> str:
        for key in ("task", "language_instruction", "instruction", "prompt", "question"):
            text = _to_text(obs.get(key))
            if text:
                return text
        for key in ("task", "language_instruction", "instruction", "prompt", "question"):
            text = _to_text(_find_nested_value(obs, key))
            if text:
                return text
        return ""

    def _resolve_depth_reasoning(self) -> bool:
        return bool(getattr(self.config, "enable_depth_reasoning", False))

    def _predict_action_for_observation(
        self,
        obs: Dict[str, Any],
        *,
        batch_idx: int,
        norm_tag: str,
        num_steps: Optional[int],
        n_action_steps: Optional[int],
        generator: Optional[torch.Generator],
    ) -> Any:
        enable_depth_reasoning = self._resolve_depth_reasoning()
        depth_cache = self._depth_caches.get(batch_idx)
        output = self.model.predict_action(
            processor=self.processor,
            images=self._extract_images(obs),
            task=self._extract_task(obs),
            state=self._extract_state(obs),
            norm_tag=norm_tag,
            inference_action_mode=self.inference_action_mode,
            enable_depth_reasoning=enable_depth_reasoning,
            depth_cache=depth_cache,
            action_tokenizer=self.action_tokenizer,
            num_steps=num_steps,
            n_action_steps=n_action_steps,
            generator=generator,
            enable_cuda_graph=bool(getattr(self.config, "enable_inference_cuda_graph", True)),
        )
        if output.depth_cache is not None:
            self._depth_caches[batch_idx] = output.depth_cache
        if output.depth_bins is not None:
            self._last_depth_video_codes_by_batch[batch_idx] = (
                output.depth_bins.detach().cpu().reshape(-1).numpy().astype(np.int64).copy()
            )
        return output

    def generate_inference_result_from_observations(
        self,
        observations: List[Dict[str, Any]],
        *,
        norm_tag: Optional[str] = None,
        num_steps: Optional[int] = None,
        n_action_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> MolmoAct2HFInferenceResult:
        if not observations:
            raise ValueError("MolmoAct2 HF inference requires at least one observation.")
        requested_norm_tag = (
            _to_text(norm_tag)
            or self._config_norm_tag(self.config)
        ).strip()
        if not requested_norm_tag:
            raise ValueError("MolmoAct2 HF policy requires `norm_tag` for action inference.")
        self._last_depth_video_codes_by_batch = {}
        with torch.inference_mode():
            output = self._predict_action_for_observation(
                observations[-1],
                batch_idx=0,
                norm_tag=requested_norm_tag,
                num_steps=self.config.num_steps if num_steps is None else num_steps,
                n_action_steps=n_action_steps,
                generator=generator,
            )
        enable_depth_reasoning = self._resolve_depth_reasoning()
        style = "robot_depth_action" if enable_depth_reasoning else "robot_action"
        return MolmoAct2HFInferenceResult(
            style=style,
            actions=output.actions,
            generated_token_ids=getattr(output, "generated_token_ids", None),
            depth_bins=output.depth_bins,
        )

    @staticmethod
    def _enqueue_action_chunk(action_queue: deque, action_chunk: torch.Tensor) -> None:
        actions = action_chunk.detach()
        if actions.ndim == 3:
            if int(actions.shape[0]) != 1:
                raise ValueError(f"Expected one action batch from MolmoAct2 HF model, got {tuple(actions.shape)}.")
            actions = actions[0]
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        if actions.ndim != 2:
            raise ValueError(f"Expected action chunk shape [T, D], got {tuple(actions.shape)}.")
        for step in torch.unbind(actions, dim=0):
            action_queue.append(step)

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        batch = _maybe_batchify_single_observation_batch(batch)
        batch_size = _infer_batch_size(batch)
        requested_norm_tag = (
            _to_text(kwargs.get("norm_tag"))
            or _to_text(kwargs.get("tag"))
            or self._config_norm_tag(self.config)
        ).strip()
        if not requested_norm_tag:
            raise ValueError("MolmoAct2 HF policy requires `norm_tag` for action inference.")
        requested_num_steps = kwargs.get("num_steps")
        if requested_num_steps is None:
            requested_num_steps = self.config.num_steps
        requested_n_action_steps = kwargs.get("n_action_steps")
        requested_generator = kwargs.get("generator")

        actions: List[Tensor] = []
        self._last_depth_video_codes_by_batch = {}
        inference_s = 0.0
        inference_calls = 0

        for idx in range(batch_size):
            action_queue = self._action_queues[idx]
            if not action_queue:
                obs = self._slice_observation_batch(batch, idx, batch_size)

                start = time.perf_counter()
                with torch.inference_mode():
                    output = self._predict_action_for_observation(
                        obs,
                        batch_idx=idx,
                        norm_tag=requested_norm_tag,
                        num_steps=requested_num_steps,
                        n_action_steps=requested_n_action_steps,
                        generator=requested_generator,
                    )
                inference_s += time.perf_counter() - start
                inference_calls += 1
                self._enqueue_action_chunk(action_queue, output.actions)

            if not action_queue:
                raise RuntimeError("MolmoAct2 HF model returned an empty action chunk.")
            action = action_queue.popleft()
            if action.ndim == 1:
                action = action.unsqueeze(0)
            actions.append(action)

        self._last_model_inference_s = float(inference_s)
        self._last_model_inference_calls = int(inference_calls)
        device = torch.device(self.config.device or "cpu")
        return torch.cat(actions, dim=0).to(device=device, dtype=torch.float32)
