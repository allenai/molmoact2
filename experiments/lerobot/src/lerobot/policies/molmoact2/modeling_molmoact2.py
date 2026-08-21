from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.hf_backend import MolmoAct2HFBackend
from lerobot.policies.pretrained import PreTrainedPolicy
from olmo.extra_tokens import (
    ACTION_TOKENS,
    DEPTH_END_TOKEN,
    DEPTH_OUTPUT_TOKEN,
    DEPTH_TOKENS,
    STATE_TOKENS,
    build_indexed_token_id_to_bin_map,
    resolve_family_boundary_token_ids,
    style_uses_action_output,
    style_uses_depth_output,
)

from .prompt_utils import build_prompt_fields

log = logging.getLogger(__name__)

_DISCRETE_GENERATION_MAX_STEPS = 128


def _ensure_in_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _resolve_molmoact2_root() -> Path:
    source_path = Path(__file__).resolve()
    for parent in Path(__file__).resolve().parents:
        if parent.name == "lerobot":
            repo_root = parent.parent
            if (repo_root / "olmo").is_dir() and (repo_root / "launch_scripts").is_dir():
                return repo_root
        if (parent / "olmo").is_dir() and (parent / "launch_scripts").is_dir():
            return parent
    raise FileNotFoundError(f"Could not locate MolmoAct2 repo root from {source_path}.")


def _is_local_training_checkpoint(checkpoint_path: str) -> bool:
    checkpoint_path = str(checkpoint_path or "").strip()
    if not checkpoint_path:
        return False
    checkpoint_dir = Path(checkpoint_path).expanduser()
    return checkpoint_dir.exists() and (checkpoint_dir / "config.yaml").exists()


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
            if key in value:
                text = _to_text(value[key])
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


def _extract_prompt_from_messages(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("question", "task", "prompt", "instruction", "text", "content"):
            if key in value:
                text = _to_text(value[key])
                if text:
                    return text
        if "messages" in value:
            return _extract_prompt_from_messages(value["messages"])
        return ""
    if isinstance(value, (list, tuple)):
        return _extract_prompt_from_messages(value[0]) if value else ""
    return _to_text(value)


def _normalize_image(array: Any) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype in (np.float32, np.float64):
        if arr.size > 0 and float(arr.max()) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _append_images(images: List[np.ndarray], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _append_images(images, item)
        return
    array = _to_numpy(value)
    if array.ndim == 3:
        images.append(_normalize_image(array))
    elif array.ndim == 4:
        for frame in array:
            images.append(_normalize_image(frame))


def _infer_batch_size(batch: Dict[str, Any]) -> int:
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
        if isinstance(value, np.ndarray) and value.ndim > 0:
            return int(value.shape[0])
        if isinstance(value, (list, tuple)) and value:
            return int(len(value))
    return 1


def _batchify_single_observation(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.unsqueeze(0)
    if isinstance(value, np.ndarray):
        return np.expand_dims(value, axis=0)
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [value]
    if isinstance(value, dict):
        return {key: _batchify_single_observation(subvalue) for key, subvalue in value.items()}
    return [value]


def _looks_like_single_observation(value: Any) -> bool:
    if torch.is_tensor(value):
        return value.ndim in {0, 1, 3}
    if isinstance(value, np.ndarray):
        return value.ndim in {0, 1, 3}
    if isinstance(value, str):
        return True
    if isinstance(value, dict):
        return any(_looks_like_single_observation(subvalue) for subvalue in value.values())
    return False


def _maybe_batchify_single_observation_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(batch, dict):
        return batch
    if not _looks_like_single_observation(batch):
        return batch
    return _batchify_single_observation(batch)


def _slice_batch_value(value: Any, idx: int, batch_size: int) -> Any:
    if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size:
        return value[idx]
    if isinstance(value, np.ndarray) and value.ndim > 0 and int(value.shape[0]) == batch_size:
        return value[idx]
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return value[idx]
    return value


def _flatten_generated_token_ids(token_ids: torch.Tensor) -> List[int]:
    if token_ids.ndim == 3:
        return [int(x) for x in token_ids[0, 0].detach().cpu().tolist()]
    if token_ids.ndim == 2:
        return [int(x) for x in token_ids[0].detach().cpu().tolist()]
    if token_ids.ndim == 1:
        return [int(x) for x in token_ids.detach().cpu().tolist()]
    raise ValueError(f"Unexpected token tensor shape {tuple(token_ids.shape)}")


def _extract_discrete_token_ids(
    generated_ids: List[int],
    start_token_id: int,
    end_token_id: int,
    token_id_to_bin: Dict[int, int],
) -> List[int]:
    start_idx = None
    end_idx = None
    for idx, token_id in enumerate(generated_ids):
        if token_id == start_token_id:
            start_idx = idx
            break
    if start_idx is not None:
        for idx in range(start_idx + 1, len(generated_ids)):
            if generated_ids[idx] == end_token_id:
                end_idx = idx
                break
    span_start = 0 if start_idx is None else start_idx + 1
    span_end = len(generated_ids) if end_idx is None else end_idx
    return [
        token_id_to_bin[token_id]
        for token_id in generated_ids[span_start:span_end]
        if token_id in token_id_to_bin
    ]


def _slice_action_chunk(actions: torch.Tensor, n_obs_steps: int, n_action_steps: Optional[int]) -> torch.Tensor:
    if n_action_steps is None:
        return actions
    start = int(n_obs_steps) - 1
    end = start + int(n_action_steps)
    if end > actions.shape[1]:
        raise ValueError(f"Requested actions up to {end} but model produced horizon {actions.shape[1]}")
    return actions[:, start:end]


def _slice_action_dim(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
    if actions.shape[-1] < int(action_dim):
        raise ValueError(
            f"Requested action_dim {int(action_dim)} but chunk only has width {actions.shape[-1]}"
        )
    return actions[..., : int(action_dim)]


def _decode_token_ids(tokenizer: Any, token_ids: List[int]) -> str:
    if not token_ids:
        return ""
    try:
        return str(tokenizer.decode(token_ids, truncate_at_eos=False))
    except TypeError:
        try:
            return str(
                tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
        except TypeError:
            return str(tokenizer.decode(token_ids))


def _format_log_text(text: str) -> str:
    return str(text).replace("\n", "\\n")


def _build_example(
    images: List[np.ndarray],
    task: str,
    normalized_state: Optional[np.ndarray],
    *,
    inference_action_mode: str,
    state_format: str,
    num_state_tokens: int,
    style: str,
    setup_type: str = "",
    control_mode: str = "",
    add_setup_tokens: bool = False,
    add_control_tokens: bool = False,
) -> Dict[str, Any]:
    # Do not attach a top-level "prompt": DataFormatter treats that as a fully
    # formatted prompt and skips the robot-style templating path used in training.
    example: Dict[str, Any] = {
        "image": images if len(images) > 1 else images[0],
    }
    if state_format in {"continuous", "both"} and normalized_state is not None:
        example["state"] = normalized_state

    prompt_fields = build_prompt_fields(
        task=task,
        style=style,
        normalized_states=normalized_state if state_format in {"discrete", "both"} else None,
        setup_type=setup_type,
        control_mode=control_mode,
        num_state_tokens=num_state_tokens,
        add_setup_tokens=add_setup_tokens,
        add_control_tokens=add_control_tokens,
    )
    if inference_action_mode == "continuous":
        example["messages"] = prompt_fields
    else:
        example.update(prompt_fields)
        example["answers"] = ""
    return example


def _build_action_model_inputs(batch: Dict[str, Any]) -> Dict[str, Any]:
    model_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch.get("attention_mask"),
        "position_ids": batch.get("position_ids"),
        "response_mask": batch.get("response_mask"),
        "images": batch.get("images"),
        "image_masks": batch.get("image_masks"),
        "token_pooling": batch.get("token_pooling"),
        "low_res_token_pooling": batch.get("low_res_token_pooling"),
        "states": batch.get("states"),
    }
    return {key: value for key, value in model_inputs.items() if value is not None}


def _build_generation_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    generation_batch = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch.get("attention_mask"),
        "images": batch.get("images"),
        "image_masks": batch.get("image_masks"),
        "token_pooling": batch.get("token_pooling"),
        "low_res_token_pooling": batch.get("low_res_token_pooling"),
        "num_images": batch.get("num_images"),
        "multimodal_type": batch.get("multimodal_type"),
        "num_image_starts": batch.get("num_image_starts"),
    }
    return {key: value for key, value in generation_batch.items() if value is not None}


def _force_resize_crop_mode(model_cfg: Any) -> Any:
    mm_preprocessor = getattr(model_cfg, "mm_preprocessor", None)
    image_cfg = getattr(mm_preprocessor, "image", None) if mm_preprocessor is not None else None
    if image_cfg is None or getattr(image_cfg, "crop_mode", None) == "resize":
        return model_cfg
    image_cfg = replace(image_cfg, crop_mode="resize")
    mm_preprocessor = replace(mm_preprocessor, image=image_cfg)
    return replace(model_cfg, mm_preprocessor=mm_preprocessor)


def _disable_inference_token_bias(model: Any) -> bool:
    transformer = getattr(model, "transformer", None)
    if getattr(transformer, "token_bias", None) is None:
        return False
    transformer.token_bias = None
    return True


def _normalizer_feature_dim(normalizer: Any) -> Optional[int]:
    if normalizer is None:
        return None
    for attr_name in ("mean", "std", "min_val", "max_val", "q_low", "q_high", "mask"):
        value = getattr(normalizer, attr_name, None)
        if value is None:
            continue
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) == 0:
            continue
        return int(shape[-1])
    return None


def _build_action_dim_is_pad(
    *,
    action_dim: int,
    max_action_dim: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if int(action_dim) > int(max_action_dim):
        raise ValueError(
            f"Requested action_dim {int(action_dim)} exceeds checkpoint max_action_dim {int(max_action_dim)}."
        )
    mask = torch.ones((int(batch_size), int(max_action_dim)), device=device, dtype=torch.bool)
    mask[:, : int(action_dim)] = False
    return mask


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
        f"cached_action_processor_{hash(str(module_path.resolve()))}",
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
        key: value for key, value in processor_config.items() if key not in {"auto_map", "processor_class"}
    }
    return processor_cls(tokenizer, **processor_kwargs)


@dataclass
class _MolmoHandles:
    model: Any
    tokenizer: Any
    preprocessor: Any
    collator: Any
    device: torch.device
    inference_action_mode: str
    num_steps: Optional[int]
    action_horizon: int
    max_action_dim: int
    n_action_steps: Optional[int]
    n_obs_steps: int
    enable_depth_reasoning: bool
    num_depth_codes: int
    state_format: str
    num_state_tokens: int
    default_style: str
    norm_tag: str
    robot_processor: Any
    add_setup_tokens: bool
    add_control_tokens: bool
    action_start_token_id: Optional[int]
    action_end_token_id: Optional[int]
    eos_token_id: Optional[int]
    depth_start_token_id: Optional[int]
    depth_end_token_id: Optional[int]
    action_token_id_to_bin: Dict[int, int]
    depth_token_id_to_bin: Dict[int, int]
    depth_bin_to_token_id: Dict[int, int]
    discrete_action_processor: Any


@dataclass
class MolmoAct2InferenceResult:
    style: str
    actions: Optional[torch.Tensor] = None
    depth_bins: Optional[torch.Tensor] = None
    generated_token_ids: Optional[torch.Tensor] = None


class MolmoAct2Policy(PreTrainedPolicy):
    config_class = MolmoAct2Config
    name = "molmoact2"

    def __init__(self, config: MolmoAct2Config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self._hf_backend: MolmoAct2HFBackend | None = None
        if not _is_local_training_checkpoint(str(config.checkpoint_path)):
            self._hf_backend = MolmoAct2HFBackend(config)
            return

        if self.config.inference_action_mode not in {"continuous", "discrete"}:
            raise ValueError(
                f"Unsupported inference_action_mode='{self.config.inference_action_mode}'. "
                "Expected one of {'continuous', 'discrete'}."
            )
        self._handles: Optional[_MolmoHandles] = None
        self._move_to_device = None
        self._obs_history: Dict[int, deque] = defaultdict(lambda: deque())
        self._action_queues: Dict[int, deque] = defaultdict(lambda: deque())
        self._last_depth_video_codes_by_batch: Dict[int, np.ndarray] = {}
        self._last_model_inference_s = 0.0
        self._last_model_inference_calls = 0
        self._prepare_model()

    @staticmethod
    def _config_norm_tag(config: Any) -> str:
        return str(getattr(config, "norm_tag", "") or "").strip()

    @staticmethod
    def _validate_norm_tag_for_processor(
        robot_processor: Any,
        norm_tag: Optional[str],
        *,
        context: str,
    ) -> str:
        tag = str(norm_tag or "").strip()
        if robot_processor is None:
            return tag
        if not tag:
            raise ValueError(
                f"MolmoAct2Policy requires `norm_tag` {context} when the checkpoint "
                "contains robot normalization metadata."
            )
        metadata_by_tag = getattr(robot_processor, "metadata_by_tag", None)
        if not isinstance(metadata_by_tag, dict) or not metadata_by_tag:
            raise ValueError(
                "MolmoAct2 checkpoint has a robot processor but no configured normalization tags."
            )
        resolved_tag = robot_processor.resolve_tag(tag)
        if resolved_tag not in metadata_by_tag:
            allowed_tags = ", ".join(sorted(str(key) for key in metadata_by_tag))
            raise ValueError(
                f"Unknown MolmoAct2 normalization tag {tag!r} {context}; "
                f"resolved to {resolved_tag!r}, which is not configured. "
                f"Allowed tags: {allowed_tags}."
            )
        return tag

    def _prepare_model(self) -> None:
        cfg = self.config
        _ensure_in_path(_resolve_molmoact2_root())

        from olmo.models.model_config import BaseModelConfig
        from olmo.torch_util import move_to_device
        from olmo.train.checkpointer import load_model_state
        from olmo.util import prepare_cli_environment, resource_path

        prepare_cli_environment()

        checkpoint_dir = cfg.checkpoint_dir
        device = torch.device(cfg.device or "cpu")
        checkpoint_config_path = Path(resource_path(checkpoint_dir, "config.yaml"))
        model_cfg = BaseModelConfig.load(checkpoint_config_path, key="model")
        model_cfg = _force_resize_crop_mode(model_cfg)
        raw_max_action_dim = getattr(model_cfg, "max_action_dim", None)
        if raw_max_action_dim is None:
            raw_max_action_dim = getattr(model_cfg, "action_dim")
        max_action_dim = int(raw_max_action_dim)

        action_format = str(getattr(model_cfg, "action_format", "continuous")).lower()
        state_format = str(getattr(model_cfg, "state_format", "continuous")).lower()
        enable_depth_reasoning = bool(getattr(cfg, "enable_depth_reasoning", False))
        num_depth_codes = int(
            getattr(cfg, "num_depth_tokens_per_image", None)
            or getattr(model_cfg, "num_depth_codes", 100)
            or 100
        )
        if num_depth_codes <= 0:
            raise ValueError(f"MolmoAct2 num_depth_codes must be > 0, got {num_depth_codes}.")
        if cfg.inference_action_mode == "discrete" and action_format not in {"discrete", "both"}:
            raise ValueError(
                f"policy.inference_action_mode='discrete' requires checkpoint action_format in "
                f"{{'discrete', 'both'}}, got '{action_format}'."
            )
        if cfg.inference_action_mode == "continuous" and action_format not in {"continuous", "both"}:
            raise ValueError(
                f"policy.inference_action_mode='continuous' requires checkpoint action_format in "
                f"{{'continuous', 'both'}}, got '{action_format}'."
            )

        seq_len = cfg.seq_len or model_cfg.llm.max_sequence_length
        preprocessor = model_cfg.build_preprocessor(
            for_inference=True,
            is_training=False,
            max_seq_len=seq_len,
        )
        collator = model_cfg.build_collator(
            preprocessor.get_output_shapes(),
            pad_mode=None,
            include_metadata=True,
        )

        proc_cfg = getattr(model_cfg, "robot_processor", None)
        if proc_cfg is None:
            proc_cfg = getattr(model_cfg, "robot_preprocessor", None) or getattr(
                model_cfg,
                "robot_postprocessor",
                None,
            )
        robot_processor = proc_cfg.build_processor() if proc_cfg is not None else None

        with torch.device("meta"):
            model = model_cfg.build_model()
        model.to_empty(device=device)
        load_model_state(checkpoint_dir, model)
        if _disable_inference_token_bias(model):
            log.info("Disabled non-persistent discrete state token value bias for inference.")
        model.to(device)
        model.eval()

        tokenizer = model_cfg.build_tokenizer()
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        inner_tokenizer = getattr(tokenizer, "tokenizer", None)
        if eos_token_id is None and inner_tokenizer is not None:
            eos_token_id = getattr(inner_tokenizer, "eos_token_id", None)
        eos_token_id = None if eos_token_id is None else int(eos_token_id)
        formatter_cfg = getattr(model_cfg, "data_formatter", None)
        add_setup_tokens = bool(getattr(formatter_cfg, "add_setup_tokens", False))
        add_control_tokens = bool(getattr(formatter_cfg, "add_control_tokens", False))

        added_tokens = list(model_cfg.llm.tokenizer.resolve_new_tokens_for_both_input_and_output())
        num_state_tokens = 0
        if state_format in {"discrete", "both"}:
            num_state_tokens = STATE_TOKENS.count_bins(added_tokens)
            if num_state_tokens <= 0 or not STATE_TOKENS.has_boundaries(added_tokens):
                raise ValueError(
                    "Discrete state inference requires <state_start>, <state_end>, and <state_i> tokens."
                )

        action_start_token_id: Optional[int] = None
        action_end_token_id: Optional[int] = None
        depth_start_token_id: Optional[int] = None
        depth_end_token_id: Optional[int] = None
        action_token_id_to_bin: Dict[int, int] = {}
        depth_token_id_to_bin: Dict[int, int] = {}
        depth_bin_to_token_id: Dict[int, int] = {}
        discrete_action_processor = None
        has_depth_output = DEPTH_OUTPUT_TOKEN in added_tokens

        if has_depth_output:
            depth_start_token_id, depth_end_token_id = resolve_family_boundary_token_ids(
                tokenizer,
                DEPTH_TOKENS,
                "Depth inference requires single-token <depth_start>/<depth_end>.",
            )
            depth_token_id_to_bin = build_indexed_token_id_to_bin_map(
                tokenizer,
                added_tokens,
                DEPTH_TOKENS,
            )
            if not depth_token_id_to_bin:
                raise ValueError(
                    "No depth tokens found in tokenizer. Expected tokens like <depth_0>."
                )
            depth_bin_to_token_id = {
                int(token_bin): int(token_id)
                for token_id, token_bin in depth_token_id_to_bin.items()
            }

        if cfg.inference_action_mode == "discrete":
            action_start_token_id, action_end_token_id = resolve_family_boundary_token_ids(
                tokenizer,
                ACTION_TOKENS,
                "Discrete action inference requires single-token <action_start>/<action_end>.",
            )
            action_token_id_to_bin = build_indexed_token_id_to_bin_map(
                tokenizer,
                added_tokens,
                ACTION_TOKENS,
            )
            if not action_token_id_to_bin:
                raise ValueError(
                    "No discrete action tokens found in tokenizer. Expected tokens like <action_0>."
                )
            if eos_token_id is None:
                raise ValueError("Discrete action inference requires the tokenizer to define eos_token_id.")
            resolved_discrete_action_tokenizer = str(cfg.discrete_action_tokenizer or "").strip()
            if not resolved_discrete_action_tokenizer:
                raise ValueError(
                    "MolmoAct2Policy with inference_action_mode='discrete' requires "
                    "`discrete_action_tokenizer` to be provided."
                )
            cfg.discrete_action_tokenizer = resolved_discrete_action_tokenizer
            discrete_action_processor = self._build_discrete_action_processor(resolved_discrete_action_tokenizer)
            self._initialize_discrete_action_processor(
                discrete_action_processor,
                action_horizon=int(model_cfg.action_horizon),
                action_dim=max_action_dim,
            )
        default_style = "robot_depth_action" if enable_depth_reasoning else "robot_action"
        if style_uses_depth_output(default_style):
            if not enable_depth_reasoning:
                raise ValueError(
                    f"Style '{default_style}' requires enable_depth_reasoning=True."
                )
            if not has_depth_output or action_format not in {"discrete", "both"}:
                raise ValueError(
                    f"Style '{default_style}' requires a checkpoint with depth tokens and discrete/both action_format."
                )

        norm_tag = self._validate_norm_tag_for_processor(
            robot_processor,
            self._config_norm_tag(cfg),
            context="during policy initialization",
        )
        default_n_action_steps = getattr(model_cfg, "n_action_steps", None)
        if robot_processor is not None and norm_tag:
            default_n_action_steps = robot_processor.get_n_action_steps(norm_tag)

        default_num_steps = cfg.num_steps
        if default_num_steps is None:
            default_num_steps = getattr(model_cfg, "flow_matching_num_steps", None)

        self._handles = _MolmoHandles(
            model=model,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            collator=collator,
            device=device,
            inference_action_mode=cfg.inference_action_mode,
            num_steps=default_num_steps,
            action_horizon=int(model_cfg.action_horizon),
            max_action_dim=max_action_dim,
            n_action_steps=None if default_n_action_steps is None else int(default_n_action_steps),
            n_obs_steps=int(getattr(model_cfg, "n_obs_steps", 1)),
            enable_depth_reasoning=enable_depth_reasoning,
            num_depth_codes=num_depth_codes,
            state_format=state_format,
            num_state_tokens=num_state_tokens,
            default_style=default_style,
            norm_tag=norm_tag,
            robot_processor=robot_processor,
            add_setup_tokens=add_setup_tokens,
            add_control_tokens=add_control_tokens,
            action_start_token_id=action_start_token_id,
            action_end_token_id=action_end_token_id,
            eos_token_id=eos_token_id,
            depth_start_token_id=depth_start_token_id,
            depth_end_token_id=depth_end_token_id,
            action_token_id_to_bin=action_token_id_to_bin,
            depth_token_id_to_bin=depth_token_id_to_bin,
            depth_bin_to_token_id=depth_bin_to_token_id,
            discrete_action_processor=discrete_action_processor,
        )
        self._move_to_device = move_to_device

    @staticmethod
    def _build_discrete_action_processor(processor_name: str) -> Any:
        offline_mode = (
            os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
            or os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
        )
        hf_token = os.environ.get("HF_ACCESS_TOKEN")
        try:
            processor_path = _resolve_discrete_action_processor_path(
                processor_name,
                local_files_only=offline_mode,
                hf_token=hf_token,
            )
            return _load_discrete_action_processor_from_path(processor_path, hf_token=hf_token)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load discrete action tokenizer '{processor_name}'."
            ) from exc

    @staticmethod
    def _initialize_discrete_action_processor(
        processor: Any,
        action_horizon: int,
        action_dim: int,
    ) -> None:
        dummy = np.zeros((1, int(action_horizon), int(action_dim)), dtype=np.float32)
        processor(dummy)

    def _maybe_log_generated_text(
        self,
        input_ids: Optional[torch.Tensor],
        generated_token_ids: Optional[torch.Tensor],
        *,
        style: str,
    ) -> None:
        if not bool(getattr(self.config, "verbose", False)):
            return
        handles = self._handles
        if handles is None:
            return
        prompt_text = ""
        if input_ids is not None:
            prompt_ids = [token_id for token_id in _flatten_generated_token_ids(input_ids) if token_id >= 0]
            prompt_text = _format_log_text(_decode_token_ids(handles.tokenizer, prompt_ids))
        generated_text = ""
        if generated_token_ids is not None:
            generated_ids = _flatten_generated_token_ids(generated_token_ids)
            generated_text = _format_log_text(_decode_token_ids(handles.tokenizer, generated_ids))
        log.info(
            "[MolmoAct2Policy] style=%s prompt=%s generated=%s",
            style,
            prompt_text,
            generated_text,
        )

    def reset(self) -> None:
        hf_backend = getattr(self, "_hf_backend", None)
        if hf_backend is not None:
            hf_backend.reset()
            return
        self._obs_history = defaultdict(lambda: deque())
        self._action_queues = defaultdict(lambda: deque())
        self._last_depth_video_codes_by_batch = {}
        self._last_model_inference_s = 0.0
        self._last_model_inference_calls = 0

    def get_last_depth_video_codes(self) -> Dict[int, np.ndarray]:
        hf_backend = getattr(self, "_hf_backend", None)
        if hf_backend is not None:
            return hf_backend.get_last_depth_video_codes()
        return {
            int(batch_idx): np.asarray(codes, dtype=np.int64).copy()
            for batch_idx, codes in self._last_depth_video_codes_by_batch.items()
        }

    def _call_generate_inference_result(self, *args, **kwargs) -> tuple[MolmoAct2InferenceResult, float]:
        inference_start = time.perf_counter()
        result = self._generate_inference_result(*args, **kwargs)
        return result, time.perf_counter() - inference_start

    def get_optim_params(self) -> dict:
        hf_backend = getattr(self, "_hf_backend", None)
        if hf_backend is not None:
            return hf_backend.get_optim_params()
        raise NotImplementedError("MolmoAct2 policy is inference-only.")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        hf_backend = getattr(self, "_hf_backend", None)
        if hf_backend is not None:
            return hf_backend.forward(batch)
        raise NotImplementedError("MolmoAct2 policy is inference-only.")

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        hf_backend = getattr(self, "_hf_backend", None)
        if hf_backend is not None:
            return hf_backend.predict_action_chunk(batch, **kwargs)

        handles = self._handles
        if handles is None:
            raise RuntimeError("MolmoAct2 handles not initialized.")

        batch = _maybe_batchify_single_observation_batch(batch)
        batch_size = _infer_batch_size(batch)
        requested_norm_tag = (
            _to_text(kwargs.get("norm_tag"))
            or _to_text(kwargs.get("tag"))
            or handles.norm_tag
        )
        requested_norm_tag = self._validate_norm_tag_for_processor(
            handles.robot_processor,
            requested_norm_tag,
            context="for action chunk inference",
        )
        requested_num_steps = kwargs.get("num_steps")
        requested_generator = kwargs.get("generator")
        requested_n_action_steps = kwargs.get("n_action_steps")

        chunks: List[Tensor] = []
        depth_codes_by_batch: Dict[int, np.ndarray] = {}
        model_inference_s = 0.0
        model_inference_calls = 0
        for idx in range(batch_size):
            obs_slice = self._slice_observation_batch(batch, idx, batch_size)
            history = self._update_obs_history(idx, obs_slice, handles.n_obs_steps)
            result = self.generate_inference_result_from_observations(
                list(history),
                norm_tag=requested_norm_tag,
                num_steps=requested_num_steps,
                n_action_steps=requested_n_action_steps,
                generator=requested_generator,
            )
            model_inference_s += float(self._last_model_inference_s)
            model_inference_calls += int(self._last_model_inference_calls)
            if result.actions is None:
                raise ValueError(f"Style '{result.style}' does not produce actions for predict_action_chunk().")
            if result.depth_bins is not None:
                depth_codes_by_batch[idx] = (
                    _to_numpy(result.depth_bins).reshape(-1).astype(np.int64).copy()
                )
            chunks.append(result.actions)

        self._last_depth_video_codes_by_batch = depth_codes_by_batch
        self._last_model_inference_s = float(model_inference_s)
        self._last_model_inference_calls = int(model_inference_calls)
        action_tensor = torch.cat(chunks, dim=0)
        resolved_action_dim = self._resolve_action_dim_for_tag(handles, requested_norm_tag)
        return _slice_action_dim(action_tensor, resolved_action_dim)

    def _obs_to_example(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        handles = self._handles
        if handles is None:
            raise RuntimeError("MolmoAct2 handles not initialized.")

        images: List[np.ndarray] = []
        for key, value in obs.items():
            if key.startswith("observation.images"):
                _append_images(images, value)
        if not images:
            if "images" in obs:
                _append_images(images, obs["images"])
            elif "image" in obs:
                _append_images(images, obs["image"])
        if not images:
            raise ValueError("No image data found in observation for MolmoAct2.")

        state = None
        if "observation.state" in obs:
            state = _to_numpy(obs["observation.state"])
        elif "state" in obs:
            state = _to_numpy(obs["state"])
        if state is not None:
            state = np.asarray(state, dtype=np.float32)
            if state.ndim == 0:
                state = state.reshape(1)
            elif state.ndim > 1 and state.shape[0] == 1:
                state = state.reshape(-1)

        messages_value = obs.get("messages")
        if messages_value is None:
            messages_value = _find_nested_value(obs, "messages")

        prompt = ""
        for key in ("language_instruction", "instruction", "prompt", "question", "task"):
            prompt = _to_text(obs.get(key))
            if prompt:
                break
        if not prompt and messages_value is not None:
            prompt = _extract_prompt_from_messages(messages_value)
        if not prompt:
            prompt = _to_text(_find_nested_value(obs, "language_instruction"))
        if not prompt:
            prompt = _to_text(_find_nested_value(obs, "instruction"))
        if not prompt:
            prompt = _to_text(_find_nested_value(obs, "prompt"))
        if not prompt:
            prompt = _to_text(_find_nested_value(obs, "question"))
        if not prompt:
            prompt = _to_text(_find_nested_value(obs, "task"))

        return {
            "image": images if len(images) > 1 else images[0],
            "state": state,
            "prompt": str(prompt or ""),
            "task": str(prompt or ""),
        }

    @staticmethod
    def _extract_prompt_from_example(example: Dict[str, Any]) -> str:
        for key in ("prompt", "task", "question"):
            text = _to_text(example.get(key))
            if text:
                return text
        if "messages" in example:
            return _extract_prompt_from_messages(example["messages"])
        return ""

    def _combine_history_examples(
        self,
        examples: List[Dict[str, Any]],
        handles: _MolmoHandles,
        *,
        norm_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        images: List[np.ndarray] = []
        state_seq: List[np.ndarray] = []
        prompt = ""

        for example in examples:
            example_image = example["image"]
            if isinstance(example_image, list):
                images.extend(example_image)
            else:
                images.append(example_image)

            if example.get("state") is not None:
                state_seq.append(np.asarray(example["state"], dtype=np.float32))

            example_prompt = self._extract_prompt_from_example(example)
            if example_prompt:
                prompt = example_prompt

        normalized_state = None
        if state_seq:
            normalized_state = np.stack(state_seq, axis=0).astype(np.float32)
        if handles.state_format in {"continuous", "both"} and normalized_state is None:
            raise ValueError(
                f"Observation state is required for checkpoint state_format='{handles.state_format}'."
            )
        resolved_norm_tag = self._validate_norm_tag_for_processor(
            handles.robot_processor,
            norm_tag or handles.norm_tag,
            context="for observation normalization",
        )
        if handles.robot_processor is not None and normalized_state is not None:
            normalized_state = np.asarray(
                handles.robot_processor.normalize_state(normalized_state, repo_id=resolved_norm_tag),
                dtype=np.float32,
            )

        robot_metadata = (
            handles.robot_processor.get_metadata(resolved_norm_tag)
            if handles.robot_processor is not None
            else {}
        )
        return _build_example(
            images,
            prompt,
            normalized_state,
            inference_action_mode=handles.inference_action_mode,
            state_format=handles.state_format,
            num_state_tokens=handles.num_state_tokens,
            style=handles.default_style,
            setup_type=str(robot_metadata.get("setup_type", "") or ""),
            control_mode=str(robot_metadata.get("control_mode", "") or ""),
            add_setup_tokens=handles.add_setup_tokens,
            add_control_tokens=handles.add_control_tokens,
        )

    def _collate_example(self, example: Dict[str, Any], handles: _MolmoHandles) -> Dict[str, Any]:
        proc = handles.preprocessor(example)
        collated = handles.collator([proc])
        collated = self._move_to_device(collated, handles.device)
        collated.pop("labels", None)
        collated.pop("loss_masks", None)
        return collated

    @staticmethod
    def _build_action_model_inputs(collated: Dict[str, Any]) -> Dict[str, Any]:
        return _build_action_model_inputs(collated)

    @staticmethod
    def _build_generation_batch(collated: Dict[str, Any]) -> Dict[str, Any]:
        return _build_generation_batch(collated)

    @staticmethod
    def _resolve_style_for_inference(handles: _MolmoHandles) -> str:
        resolved_style = str(handles.default_style or "robot_action")
        if style_uses_depth_output(resolved_style):
            if not handles.enable_depth_reasoning:
                raise ValueError(
                    f"Style '{resolved_style}' requires enable_depth_reasoning=True."
                )
            if (
                handles.depth_start_token_id is None
                or handles.depth_end_token_id is None
                or not handles.depth_token_id_to_bin
            ):
                raise ValueError(
                    f"Style '{resolved_style}' requires a checkpoint with depth tokens."
                )
        return resolved_style

    def _decode_discrete_action_chunk(
        self,
        handles: _MolmoHandles,
        generated_token_ids: torch.Tensor,
        *,
        action_dim: Optional[int] = None,
    ) -> torch.Tensor:
        if handles.discrete_action_processor is None:
            raise RuntimeError("Discrete action processor is not initialized.")
        if handles.action_start_token_id is None or handles.action_end_token_id is None:
            raise RuntimeError("Discrete action boundary token IDs are not initialized.")

        flat_generated_ids = _flatten_generated_token_ids(generated_token_ids)
        discrete_token_ids = _extract_discrete_token_ids(
            flat_generated_ids,
            handles.action_start_token_id,
            handles.action_end_token_id,
            handles.action_token_id_to_bin,
        )
        if not discrete_token_ids:
            raise RuntimeError(
                "Model generated no decodable action tokens between <action_start>/<action_end>."
            )

        try:
            decoded = handles.discrete_action_processor.decode(
                [discrete_token_ids],
                time_horizon=int(handles.action_horizon),
                action_dim=int(action_dim or handles.max_action_dim),
            )
        except TypeError:
            decoded = handles.discrete_action_processor.decode([discrete_token_ids])

        action_chunk = np.asarray(decoded, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, None, :]
        elif action_chunk.ndim == 2:
            action_chunk = action_chunk[None, :, :]
        elif action_chunk.ndim > 3:
            action_chunk = action_chunk.reshape(1, action_chunk.shape[-2], action_chunk.shape[-1])
        if action_chunk.ndim != 3:
            raise RuntimeError(f"Decoded action chunk has unexpected shape {action_chunk.shape}")
        return torch.as_tensor(action_chunk, device=handles.device, dtype=torch.float32)

    @staticmethod
    def _resolve_action_dim_for_tag(handles: _MolmoHandles, norm_tag: Optional[str]) -> int:
        if handles.robot_processor is None:
            return int(handles.max_action_dim)
        norm_tag = MolmoAct2Policy._validate_norm_tag_for_processor(
            handles.robot_processor,
            norm_tag,
            context="for action dimension resolution",
        )
        expected_action_dim = handles.robot_processor.get_action_dim(norm_tag)
        if expected_action_dim is None:
            return int(handles.max_action_dim)
        if int(expected_action_dim) > int(handles.max_action_dim):
            resolved_tag = handles.robot_processor.resolve_tag(norm_tag)
            raise ValueError(
                "MolmoAct2 checkpoint max_action_dim is smaller than the selected normalization tag: "
                f"checkpoint max_action_dim={int(handles.max_action_dim)}, "
                f"norm_tag={norm_tag!r}, resolved_tag={resolved_tag!r}, "
                f"tag action_dim={int(expected_action_dim)}."
            )
        return int(expected_action_dim)

    @staticmethod
    def _resolve_action_horizon_for_tag(handles: _MolmoHandles, norm_tag: Optional[str]) -> int:
        if handles.robot_processor is None:
            raise ValueError("MolmoAct2 checkpoint is missing robot tag metadata required for action_horizon.")
        norm_tag = MolmoAct2Policy._validate_norm_tag_for_processor(
            handles.robot_processor,
            norm_tag,
            context="for action horizon resolution",
        )
        action_horizon = handles.robot_processor.get_action_horizon(norm_tag)
        if action_horizon is None:
            resolved_tag = handles.robot_processor.resolve_tag(norm_tag)
            raise ValueError(
                f"MolmoAct2 checkpoint metadata is missing action_horizon for norm_tag={norm_tag!r}, "
                f"resolved_tag={resolved_tag!r}."
            )
        if int(action_horizon) > int(handles.action_horizon):
            resolved_tag = handles.robot_processor.resolve_tag(norm_tag)
            raise ValueError(
                "MolmoAct2 checkpoint max action_horizon is smaller than the selected normalization tag: "
                f"checkpoint action_horizon={int(handles.action_horizon)}, "
                f"norm_tag={norm_tag!r}, resolved_tag={resolved_tag!r}, "
                f"tag action_horizon={int(action_horizon)}."
            )
        return int(action_horizon)

    @staticmethod
    def _resolve_n_action_steps_for_tag(
        handles: _MolmoHandles,
        norm_tag: Optional[str],
        requested_n_action_steps: Optional[int],
    ) -> int:
        tag_action_horizon = MolmoAct2Policy._resolve_action_horizon_for_tag(handles, norm_tag)
        if requested_n_action_steps is None:
            if handles.robot_processor is None:
                raise ValueError("MolmoAct2 checkpoint is missing robot tag metadata required for n_action_steps.")
            default_n_action_steps = handles.robot_processor.get_n_action_steps(norm_tag)
            if default_n_action_steps is None:
                resolved_tag = handles.robot_processor.resolve_tag(norm_tag)
                raise ValueError(
                    f"MolmoAct2 checkpoint metadata is missing n_action_steps for norm_tag={norm_tag!r}, "
                    f"resolved_tag={resolved_tag!r}."
                )
            resolved_n_action_steps = int(default_n_action_steps)
        else:
            resolved_n_action_steps = int(requested_n_action_steps)
        if resolved_n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {resolved_n_action_steps}.")
        if resolved_n_action_steps > tag_action_horizon:
            resolved_tag = handles.robot_processor.resolve_tag(norm_tag) if handles.robot_processor is not None else norm_tag
            raise ValueError(
                f"Requested n_action_steps={resolved_n_action_steps} exceeds action_horizon={tag_action_horizon} "
                f"for norm_tag={norm_tag!r}, resolved_tag={resolved_tag!r}."
            )
        return resolved_n_action_steps

    def _decode_discrete_depth_bins(
        self,
        handles: _MolmoHandles,
        generated_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if handles.depth_start_token_id is None or handles.depth_end_token_id is None:
            raise RuntimeError("Depth boundary token IDs are not initialized.")

        flat_generated_ids = _flatten_generated_token_ids(generated_token_ids)
        depth_token_bins = _extract_discrete_token_ids(
            flat_generated_ids,
            handles.depth_start_token_id,
            handles.depth_end_token_id,
            handles.depth_token_id_to_bin,
        )
        if not depth_token_bins:
            raise RuntimeError(
                "Model generated no decodable depth tokens between <depth_start>/<depth_end>."
            )
        return torch.as_tensor([depth_token_bins], device=handles.device, dtype=torch.long)

    def _generate_depth_conditioned_actions(
        self,
        collated: Dict[str, Any],
        handles: _MolmoHandles,
        *,
        action_dim_is_pad: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if handles.depth_end_token_id is None:
            raise RuntimeError(f"Style requires a single-token {DEPTH_END_TOKEN}.")

        generated = handles.model.generate(
            self._build_generation_batch(collated),
            max_steps=_DISCRETE_GENERATION_MAX_STEPS,
            beam_size=1,
            end_index=handles.depth_end_token_id,
            return_conditioning=False,
            include_final_token_in_conditioning=True,
        )
        depth_bins = self._decode_discrete_depth_bins(handles, generated.token_ids)

        if generated.attn_key_values is None:
            raise RuntimeError("Depth generation did not return KV cache.")
        full_input_ids = torch.cat(
            [collated["input_ids"], generated.token_ids[:, 0, :]],
            dim=1,
        )
        full_attention_mask = None
        encoder_kv_states = generated.attn_key_values

        action_chunk = handles.model.generate_actions(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            states=collated.get("states"),
            action_dim_is_pad=action_dim_is_pad,
            num_steps=handles.num_steps,
            generator=generator,
            encoder_hidden_states=None,
            encoder_kv_states=encoder_kv_states,
            encoder_attention_mask=handles.model._get_encoder_attention_mask(
                full_input_ids,
                full_attention_mask,
            ),
        )
        return action_chunk, generated.token_ids, depth_bins

    def _generate_inference_result(
        self,
        collated: Dict[str, Any],
        handles: _MolmoHandles,
        *,
        norm_tag: Optional[str] = None,
        num_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> MolmoAct2InferenceResult:
        style = self._resolve_style_for_inference(handles)
        generated_token_ids: Optional[torch.Tensor] = None
        resolved_num_steps = handles.num_steps if num_steps is None else int(num_steps)
        resolved_action_dim = (
            self._resolve_action_dim_for_tag(handles, norm_tag)
            if style_uses_action_output(style)
            else None
        )
        batch_size = int(collated["input_ids"].shape[0])
        action_dim_is_pad = None
        if resolved_action_dim is not None:
            action_dim_is_pad = _build_action_dim_is_pad(
                action_dim=resolved_action_dim,
                max_action_dim=handles.max_action_dim,
                batch_size=batch_size,
                device=handles.device,
            )
        with torch.no_grad():
            if handles.inference_action_mode == "continuous":
                if style_uses_depth_output(style):
                    if style_uses_action_output(style):
                        conditioned_handles = replace(handles, num_steps=resolved_num_steps)
                        action_chunk, generated_token_ids, depth_bins = self._generate_depth_conditioned_actions(
                            collated,
                            conditioned_handles,
                            action_dim_is_pad=action_dim_is_pad,
                            generator=generator,
                        )
                        action_chunk = _slice_action_dim(action_chunk, int(resolved_action_dim))
                        return MolmoAct2InferenceResult(
                            style=style,
                            actions=action_chunk,
                            depth_bins=depth_bins,
                            generated_token_ids=generated_token_ids,
                        )

                    if handles.depth_end_token_id is None:
                        raise RuntimeError(f"Style requires a single-token {DEPTH_END_TOKEN}.")
                    generated = handles.model.generate(
                        self._build_generation_batch(collated),
                        max_steps=_DISCRETE_GENERATION_MAX_STEPS,
                        beam_size=1,
                        end_index=handles.depth_end_token_id,
                    )
                    generated_token_ids = generated.token_ids
                    depth_bins = self._decode_discrete_depth_bins(handles, generated.token_ids)
                    return MolmoAct2InferenceResult(
                        style=style,
                        depth_bins=depth_bins,
                        generated_token_ids=generated_token_ids,
                    )

                if not style_uses_action_output(style):
                    return MolmoAct2InferenceResult(style=style)
                action_chunk = handles.model.generate_actions(
                    **self._build_action_model_inputs(collated),
                    action_dim_is_pad=action_dim_is_pad,
                    num_steps=resolved_num_steps,
                    generator=generator,
                )
                action_chunk = _slice_action_dim(action_chunk, int(resolved_action_dim))
                return MolmoAct2InferenceResult(style=style, actions=action_chunk)

            if style_uses_depth_output(style):
                end_index = handles.eos_token_id if style_uses_action_output(style) else handles.depth_end_token_id
                generated = handles.model.generate(
                    self._build_generation_batch(collated),
                    max_steps=_DISCRETE_GENERATION_MAX_STEPS,
                    beam_size=1,
                    end_index=end_index,
                )
                generated_token_ids = generated.token_ids
                depth_bins = self._decode_discrete_depth_bins(handles, generated_token_ids)
                action_chunk = None
                if style_uses_action_output(style):
                    action_chunk = self._decode_discrete_action_chunk(
                        handles,
                        generated_token_ids,
                        action_dim=resolved_action_dim,
                    )
                    action_chunk = _slice_action_dim(action_chunk, int(resolved_action_dim))
                return MolmoAct2InferenceResult(
                    style=style,
                    actions=action_chunk,
                    depth_bins=depth_bins,
                    generated_token_ids=generated_token_ids,
                )
            end_index = None
            if style_uses_action_output(style):
                if handles.eos_token_id is None:
                    raise RuntimeError("Discrete action generation requires tokenizer eos_token_id.")
                end_index = handles.eos_token_id
            generated = handles.model.generate(
                self._build_generation_batch(collated),
                max_steps=_DISCRETE_GENERATION_MAX_STEPS,
                beam_size=1,
                end_index=end_index,
            )
            generated_token_ids = generated.token_ids
            action_chunk = None
            depth_bins = None
            if style_uses_action_output(style):
                action_chunk = self._decode_discrete_action_chunk(
                    handles,
                    generated.token_ids,
                    action_dim=resolved_action_dim,
                )
                action_chunk = _slice_action_dim(action_chunk, int(resolved_action_dim))
            return MolmoAct2InferenceResult(
                style=style,
                actions=action_chunk,
                depth_bins=depth_bins,
                generated_token_ids=generated_token_ids,
            )

    def generate_inference_result_from_observations(
        self,
        observations: Dict[str, Any] | List[Dict[str, Any]],
        *,
        norm_tag: Optional[str] = None,
        num_steps: Optional[int] = None,
        n_action_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> MolmoAct2InferenceResult:
        hf_backend = getattr(self, "_hf_backend", None)
        if hf_backend is not None:
            hf_result = hf_backend.generate_inference_result_from_observations(
                observations if isinstance(observations, list) else [observations],
                norm_tag=norm_tag,
                num_steps=num_steps,
                n_action_steps=n_action_steps,
                generator=generator,
            )
            return MolmoAct2InferenceResult(
                style=hf_result.style,
                actions=hf_result.actions,
                depth_bins=hf_result.depth_bins,
                generated_token_ids=hf_result.generated_token_ids,
            )

        handles = self._handles
        if handles is None:
            raise RuntimeError("MolmoAct2 handles not initialized.")

        if isinstance(observations, dict):
            observations = [observations]
        if not observations:
            raise ValueError("Expected at least one observation to generate an inference result.")

        step_examples = [self._obs_to_example(obs) for obs in observations]
        if len(step_examples) < handles.n_obs_steps:
            step_examples = [step_examples[0]] * (handles.n_obs_steps - len(step_examples)) + step_examples
        elif len(step_examples) > handles.n_obs_steps:
            step_examples = step_examples[-handles.n_obs_steps :]

        combined_example = self._combine_history_examples(
            step_examples,
            handles,
            norm_tag=norm_tag,
        )
        style = self._resolve_style_for_inference(handles)
        collated = self._collate_example(combined_example, handles)
        result, inference_elapsed = self._call_generate_inference_result(
            collated,
            handles,
            norm_tag=norm_tag,
            num_steps=num_steps,
            generator=generator,
        )
        self._last_model_inference_s = float(inference_elapsed)
        self._last_model_inference_calls = 1
        self._maybe_log_generated_text(
            collated.get("input_ids"),
            result.generated_token_ids,
            style=style,
        )
        if result.actions is not None:
            resolved_norm_tag = norm_tag or handles.norm_tag
            resolved_n_action_steps = self._resolve_n_action_steps_for_tag(
                handles,
                resolved_norm_tag,
                n_action_steps,
            )
            result.actions = _slice_action_chunk(
                result.actions,
                handles.n_obs_steps,
                resolved_n_action_steps,
            )
            resolved_action_dim = self._resolve_action_dim_for_tag(
                handles,
                resolved_norm_tag,
            )
            result.actions = _slice_action_dim(result.actions, resolved_action_dim)
            if handles.robot_processor is not None:
                result.actions = handles.robot_processor.unnormalize_action(
                    result.actions,
                    repo_id=resolved_norm_tag,
                )
        return result

    def generate_action_chunk_from_observations(
        self,
        observations: Dict[str, Any] | List[Dict[str, Any]],
        *,
        norm_tag: Optional[str] = None,
        num_steps: Optional[int] = None,
        n_action_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], str]:
        result = self.generate_inference_result_from_observations(
            observations,
            norm_tag=norm_tag,
            num_steps=num_steps,
            n_action_steps=n_action_steps,
            generator=generator,
        )
        if result.actions is None:
            raise ValueError(f"Style '{result.style}' does not produce actions.")
        return result.actions, result.generated_token_ids, result.style

    def _slice_observation_batch(self, batch: Dict[str, Any], idx: int, batch_size: int) -> Dict[str, Any]:
        return {
            key: _slice_batch_value(value, idx, batch_size)
            for key, value in batch.items()
        }

    def _update_obs_history(self, idx: int, obs_slice: Dict[str, Any], n_obs_steps: int) -> deque:
        history = self._obs_history[idx]
        if history.maxlen != n_obs_steps:
            history = deque(history, maxlen=n_obs_steps)
            self._obs_history[idx] = history
        history.append(obs_slice)
        while len(history) < n_obs_steps:
            history.append(obs_slice)
        return history

    def _enqueue_action_chunk(
        self,
        action_queue: deque,
        action_chunk: torch.Tensor,
        handles: _MolmoHandles,
        *,
        norm_tag: Optional[str],
        n_action_steps: Optional[int],
    ) -> None:
        resolved_n_action_steps = self._resolve_n_action_steps_for_tag(
            handles,
            norm_tag,
            n_action_steps,
        )
        sliced = _slice_action_chunk(
            action_chunk,
            handles.n_obs_steps,
            resolved_n_action_steps,
        ).detach()
        for step in torch.unbind(sliced, dim=1):
            action_queue.append(step)

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        hf_backend = getattr(self, "_hf_backend", None)
        if hf_backend is not None:
            return hf_backend.select_action(batch, **kwargs)

        handles = self._handles
        if handles is None:
            raise RuntimeError("MolmoAct2 handles not initialized.")

        batch = _maybe_batchify_single_observation_batch(batch)
        batch_size = _infer_batch_size(batch)
        actions: List[Tensor] = []
        requested_norm_tag = (
            _to_text(kwargs.get("norm_tag"))
            or _to_text(kwargs.get("tag"))
            or handles.norm_tag
        )
        requested_norm_tag = self._validate_norm_tag_for_processor(
            handles.robot_processor,
            requested_norm_tag,
            context="for action inference",
        )
        requested_num_steps = kwargs.get("num_steps")
        requested_generator = kwargs.get("generator")
        requested_n_action_steps = kwargs.get("n_action_steps")
        model_inference_s = 0.0
        model_inference_calls = 0
        self._last_depth_video_codes_by_batch = {}

        for idx in range(batch_size):
            obs_slice = self._slice_observation_batch(batch, idx, batch_size)
            history = self._update_obs_history(idx, obs_slice, handles.n_obs_steps)
            action_queue = self._action_queues[idx]

            if not action_queue:
                step_examples = [self._obs_to_example(step_obs) for step_obs in history]
                combined_example = self._combine_history_examples(
                    step_examples,
                    handles,
                    norm_tag=requested_norm_tag,
                )
                style = self._resolve_style_for_inference(handles)
                collated = self._collate_example(combined_example, handles)
                result, inference_elapsed = self._call_generate_inference_result(
                    collated,
                    handles,
                    norm_tag=requested_norm_tag,
                    num_steps=requested_num_steps,
                    generator=requested_generator,
                )
                model_inference_s += float(inference_elapsed)
                model_inference_calls += 1
                if result.actions is None:
                    raise ValueError(f"Style '{style}' does not produce actions for select_action().")
                if result.depth_bins is not None:
                    self._last_depth_video_codes_by_batch[idx] = (
                        _to_numpy(result.depth_bins).reshape(-1).astype(np.int64).copy()
                    )
                self._maybe_log_generated_text(
                    collated.get("input_ids"),
                    result.generated_token_ids,
                    style=style,
                )
                action_chunk = result.actions
                resolved_action_dim = self._resolve_action_dim_for_tag(
                    handles,
                    requested_norm_tag,
                )
                action_chunk = _slice_action_dim(action_chunk, resolved_action_dim)
                if handles.robot_processor is not None:
                    action_chunk = handles.robot_processor.unnormalize_action(
                        action_chunk,
                        repo_id=requested_norm_tag,
                    )
                self._enqueue_action_chunk(
                    action_queue,
                    action_chunk,
                    handles,
                    norm_tag=requested_norm_tag,
                    n_action_steps=requested_n_action_steps,
                )

            next_action = action_queue.popleft()
            if next_action.ndim == 1:
                next_action = next_action.unsqueeze(0)
            actions.append(next_action)

        self._last_model_inference_s = float(model_inference_s)
        self._last_model_inference_calls = int(model_inference_calls)
        action_tensor = torch.cat(actions, dim=0)
        resolved_action_dim = self._resolve_action_dim_for_tag(handles, requested_norm_tag)
        action_tensor = _slice_action_dim(action_tensor, resolved_action_dim)
        if handles.robot_processor is not None:
            resolved_tag = handles.robot_processor.resolve_tag(requested_norm_tag)
            expected_action_dim = handles.robot_processor.get_action_dim(requested_norm_tag)
            if expected_action_dim is not None and int(action_tensor.shape[-1]) != int(expected_action_dim):
                raise ValueError(
                    "MolmoAct2 checkpoint action dimension does not match the selected normalization tag: "
                    f"checkpoint max_action_dim={int(handles.max_action_dim)}, "
                    f"norm_tag={requested_norm_tag!r}, resolved_tag={resolved_tag!r}, "
                    f"tag action_dim={int(expected_action_dim)}."
                )
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        return action_tensor.to(handles.device)
