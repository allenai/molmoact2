"""MolmoAct2-DROID inference server.

Wire protocol (matches `inference_script.py` client):

    GET  /act        -> health check, returns {"status": "ok", ...}
    POST /act        -> action inference
        request body  (json_numpy):
            {
              "external_cam": ndarray(H, W, 3) uint8 RGB,
              "wrist_cam":    ndarray(H, W, 3) uint8 RGB,
              "instruction":  str,
              "state":        ndarray(8,)  float32  [q1..q7, gripper],
              "timestamp":    float (optional),
            }
        response body (json_numpy):
            {"actions": ndarray(N, 8) float32, "dt_ms": float}

Run:

    uv run python host_server.py --host 0.0.0.0 --port 8000

Then point clients at http://<lan-ip>:8000 (e.g. http://172.16.0.42:8000).
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from typing import Any

import json_numpy
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Patches the stdlib `json` module so np.ndarray round-trips through JSON.
# Must be called before any json.dumps/loads we rely on.
json_numpy.patch()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("molmoact2.server")


REPO_ID = "allenai/MolmoAct2-DROID"
NORM_TAG = "franka_droid"
DEFAULT_NUM_STEPS = 10


def _patch_modeling_for_bf16(local_dir: str) -> None:
    """Patch the upstream `modeling_molmoact2.py` so the flow-matching
    trajectory inherits the model's dtype instead of being hardcoded to
    float32. Without this, loading in bfloat16 fails with
    `mat1 and mat2 must have the same dtype` once the action expert runs.

    The patch is idempotent and edits both the snapshot copy and the
    transformers `~/.cache/huggingface/modules/...` copy that
    `trust_remote_code` actually imports from.
    """
    patches = [
        # 1. Make the flow-matching trajectory inherit the model dtype so it
        #    matches the action-expert weights (bf16 vs hardcoded fp32).
        (
            "device=device,\n            dtype=torch.float32,\n            generator=generator,",
            "device=device,\n"
            "            dtype=source_tensor.dtype,  # patched_bf16_dtype\n"
            "            generator=generator,",
            "patched_bf16_dtype",
        ),
        # 2. `_to_array` calls `.numpy()` on the action tensor for
        #    unnormalisation; bf16 has no numpy dtype, so cast to fp32 first.
        (
            "return value.detach().cpu().numpy().astype(np.float32, copy=False)",
            "return value.detach().cpu().float().numpy().astype(np.float32, copy=False)  # patched_bf16_to_array",
            "patched_bf16_to_array",
        ),
    ]
    candidates = [os.path.join(local_dir, "modeling_molmoact2.py")]
    modules_root = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules"
    )
    if os.path.isdir(modules_root):
        for sub in os.listdir(modules_root):
            p = os.path.join(modules_root, sub, "modeling_molmoact2.py")
            if os.path.isfile(p):
                candidates.append(p)
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
        except OSError:
            continue
        new_src = src
        applied: list[str] = []
        for needle, replacement, marker in patches:
            if marker in new_src:
                continue
            if needle not in new_src:
                log.warning("patch %s: needle not found in %s", marker, path)
                continue
            new_src = new_src.replace(needle, replacement, 1)
            applied.append(marker)
        if new_src != src:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_src)
            log.info("Applied patches %s in %s", applied, path)


class Policy:
    """Holds the loaded model + processor and serializes inference calls."""

    def __init__(
        self,
        repo_id: str,
        device: str,
        dtype: torch.dtype,
        enable_cuda_graph: bool = False,
    ) -> None:
        self.default_cuda_graph = enable_cuda_graph
        # The MolmoAct2 model code reads `norm_stats.json` from
        # `config._name_or_path`. Loading by repo id makes that a non-path
        # string, so `predict_action` fails at runtime. Resolve the local
        # snapshot dir up front and load from there — `snapshot_download` is a
        # no-op when files are cached.
        local_dir = snapshot_download(repo_id=repo_id)
        log.info("Resolved snapshot dir: %s", local_dir)

        _patch_modeling_for_bf16(local_dir)

        log.info("Loading processor")
        # The tokenizer_config.json ships `extra_special_tokens` as a list, but
        # transformers >=4.46 wants a dict and crashes with
        # `'list' object has no attribute 'keys'`. The model code only looks
        # these up via `convert_tokens_to_ids`, so overriding with an empty
        # dict is safe and avoids monkey-patching transformers.
        self.processor = AutoProcessor.from_pretrained(
            local_dir, trust_remote_code=True, extra_special_tokens={}
        )

        log.info("Loading model (dtype=%s, device=%s)", dtype, device)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                local_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
        )
        self.device = device

        # The model's `_move_inputs_to_device` only moves tensors; it does not
        # cast float inputs to the model dtype. With bf16 weights the
        # processor's float32 `pixel_values` then triggers
        # `RuntimeError: mat1 and mat2 must have the same dtype`. Patch in a
        # cast on the instance so we don't have to fork the upstream code.
        target_dtype = next(self.model.parameters()).dtype

        def _move_and_cast(
            inputs: Any, dev: Any, _target: torch.dtype = target_dtype
        ) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    value = value.to(dev)
                    if value.is_floating_point() and value.dtype != _target:
                        value = value.to(_target)
                out[key] = value
            return out

        self.model._move_inputs_to_device = _move_and_cast
        # Coarse lock: real-robot clients poll at ~5 Hz, and CUDA graphs in the
        # action expert are not safe under concurrent calls.
        self._lock = threading.Lock()

    @torch.inference_mode()
    def predict(
        self,
        external_cam: np.ndarray,
        wrist_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        num_steps: int = DEFAULT_NUM_STEPS,
        enable_cuda_graph: bool = False,
    ) -> np.ndarray:
        ext_pil = _to_pil(external_cam)
        wri_pil = _to_pil(wrist_cam)
        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_f32.shape != (8,):
            raise ValueError(f"state must be shape (8,), got {state_f32.shape}")

        with self._lock:
            out = self.model.predict_action(
                processor=self.processor,
                images=[ext_pil, wri_pil],
                task=instruction,
                state=state_f32,
                norm_tag=NORM_TAG,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=num_steps,
                normalize_language=True,
                enable_cuda_graph=enable_cuda_graph,
            )
        raw = out.actions
        if torch.is_tensor(raw):
            raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = np.asarray(raw, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return actions


def _to_pil(arr: Any) -> Image.Image:
    if isinstance(arr, Image.Image):
        return arr.convert("RGB")
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got shape {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a, mode="RGB")


def build_app(policy: Policy) -> FastAPI:
    app = FastAPI(title="MolmoAct2-DROID server", version="0.1.0")

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "repo_id": REPO_ID,
                "norm_tag": NORM_TAG,
                "device": policy.device,
                "dtype": str(policy.model.dtype),
            }
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/act")
    async def act(request: Request) -> Response:
        raw = await request.body()
        try:
            payload = json_numpy.loads(raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            return _error_response(400, f"failed to decode json_numpy body: {e}")

        try:
            external_cam = payload["external_cam"]
            wrist_cam = payload["wrist_cam"]
            instruction = str(payload["instruction"])
            state = payload["state"]
        except KeyError as e:
            return _error_response(400, f"missing required field: {e}")

        num_steps = int(payload.get("num_steps", DEFAULT_NUM_STEPS))
        enable_cuda_graph = bool(
            payload.get("enable_cuda_graph", policy.default_cuda_graph)
        )

        t0 = time.perf_counter()
        try:
            actions = policy.predict(
                external_cam=external_cam,
                wrist_cam=wrist_cam,
                instruction=instruction,
                state=state,
                num_steps=num_steps,
                enable_cuda_graph=enable_cuda_graph,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("inference failed")
            return _error_response(500, f"inference failed: {e}")
        dt_ms = (time.perf_counter() - t0) * 1000.0

        body = json_numpy.dumps({"actions": actions, "dt_ms": dt_ms})
        return Response(content=body, media_type="application/json")

    return app


def _error_response(status: int, message: str) -> Response:
    body = json_numpy.dumps({"error": message})
    return Response(content=body, status_code=status, media_type="application/json")


def warmup(policy: Policy) -> None:
    """Run one inference on a dummy frame so the first real request is fast."""
    log.info("Warming up model with a dummy frame (cuda_graph=%s) ...",
             policy.default_cuda_graph)
    dummy_img = np.zeros((180, 320, 3), dtype=np.uint8)
    dummy_state = np.zeros(8, dtype=np.float32)
    t0 = time.perf_counter()
    try:
        policy.predict(
            external_cam=dummy_img,
            wrist_cam=dummy_img,
            instruction="warmup",
            state=dummy_state,
            num_steps=DEFAULT_NUM_STEPS,
            enable_cuda_graph=policy.default_cuda_graph,
        )
    except Exception:  # noqa: BLE001
        log.exception("warmup inference failed (server will still start)")
        return
    log.info("Warmup OK (%.1f ms)", (time.perf_counter() - t0) * 1000.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MolmoAct2-DROID inference server")
    p.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    p.add_argument("--repo-id", default=REPO_ID, help=f"HF repo id (default: {REPO_ID})")
    p.add_argument("--device", default="cuda:0", help="torch device (default: cuda:0)")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="model dtype (default: bfloat16)",
    )
    p.add_argument("--no-warmup", action="store_true", help="skip warmup pass")
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="enable CUDA graph capture for action expert (faster but ~2 GB more VRAM)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    # hf-transfer accelerates downloads but is harmless when files are cached.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    policy = Policy(
        repo_id=args.repo_id,
        device=args.device,
        dtype=dtype,
        enable_cuda_graph=args.cuda_graph,
    )
    if not args.no_warmup:
        warmup(policy)

    app = build_app(policy)

    import uvicorn

    log.info("Listening on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
