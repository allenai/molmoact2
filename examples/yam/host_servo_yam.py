"""Serve MolmoAct2-BimanualYAM through Servo's authenticated ``/act`` API.

This is a thin adapter: the sibling ``host_server_yam.py`` remains the owner
of checkpoint loading and inference, while Servo owns the public request,
authentication, identity-fencing, readiness, and response contracts.

Run with Servo's source tree on ``PYTHONPATH``::

    PYTHONPATH=examples/yam:/home/npow/code/servo/src \
      python examples/yam/host_servo_yam.py \
      --verifier-config ~/.config/servo/molmoact2-yam-odin-verifier.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from fastapi import Request
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

from host_server_yam import (
    DEFAULT_NUM_STEPS,
    NORM_TAG,
    REPO_ID,
    STATE_DIM,
    Policy,
)
from servo_contracts.inference import Observation
from servo_contracts.serving import EndpointIdentity, ServingMetadata
from servo_platform.adapters.serving.server import (
    EndpointContext,
    VLAActionServingRuntime,
    create_vla_serving_app,
)
from servo_platform.serving.auth import EndpointTokenVerifier
from servo_runtimes.images import decode_image


log = logging.getLogger("molmoact2.yam.servo")

ACTION_HORIZON = 30
ACTION_DIM = 14
SERVO_VERIFIER_SCHEMA = "servo.molmoact2-yam.endpoint-verifier.v1"

_ACT_REQUESTS = Counter(
    "molmoact2_yam_act_requests_total",
    "Authenticated MolmoAct2 BimanualYAM action requests handled by the runtime",
    ("outcome",),
)
_INFERENCE_DURATION = Histogram(
    "molmoact2_yam_inference_duration_seconds",
    "MolmoAct2 BimanualYAM model inference duration",
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0),
)
_HTTP_REQUESTS = Counter(
    "molmoact2_yam_http_requests_total",
    "HTTP requests handled by the Servo YAM endpoint",
    ("method", "route", "status"),
)
_HTTP_DURATION = Histogram(
    "molmoact2_yam_http_request_duration_seconds",
    "HTTP request duration for the Servo YAM endpoint",
    ("method", "route"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
)
_MODEL_READY = Gauge(
    "molmoact2_yam_model_ready",
    "Whether the MolmoAct2 BimanualYAM runtime completed load and smoke inference",
)
_MODEL_LOAD_DURATION = Gauge(
    "molmoact2_yam_model_load_duration_seconds",
    "Most recent MolmoAct2 BimanualYAM model load and warm-up duration",
)
_CUDA_ALLOCATED = Gauge(
    "molmoact2_yam_cuda_memory_allocated_bytes",
    "CUDA tensor memory currently allocated by PyTorch",
)
_CUDA_RESERVED = Gauge(
    "molmoact2_yam_cuda_memory_reserved_bytes",
    "CUDA memory currently reserved by PyTorch",
)
_RUNTIME_INFO = Info(
    "molmoact2_yam_runtime",
    "Static identity of the running MolmoAct2 BimanualYAM endpoint",
)


def _metrics_route(path: str) -> str:
    if path in {"/act", "/healthz", "/livez", "/metadata"}:
        return path
    return "other"


def _update_cuda_metrics(device: str) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return
    try:
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        _CUDA_ALLOCATED.set(torch.cuda.memory_allocated(index))
        _CUDA_RESERVED.set(torch.cuda.memory_reserved(index))
    except Exception:  # Metrics must never fail inference.
        pass


def _smoke_image_data_uri() -> str:
    output = io.BytesIO()
    Image.new("RGB", (320, 180)).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


class ServoYAMPolicy:
    """Adapt the released Transformers checkpoint to Servo's policy protocol."""

    family = "molmoact2"
    adapter_key = "molmoact2"

    def __init__(
        self,
        *,
        repo_id: str,
        device: str,
        dtype: torch.dtype,
        enable_cuda_graph: bool,
    ) -> None:
        self.repo_id = repo_id
        self.device = device
        self.dtype = dtype
        self.enable_cuda_graph = enable_cuda_graph
        self._policy: Policy | None = None

    def load(self) -> None:
        if self._policy is None:
            self._policy = Policy(
                repo_id=self.repo_id,
                device=self.device,
                dtype=self.dtype,
                enable_cuda_graph=self.enable_cuda_graph,
            )

    def smoke_observation(self) -> Observation:
        image = _smoke_image_data_uri()
        return Observation(
            images={"top": image, "left": image, "right": image},
            state=[0.0] * STATE_DIM,
            instruction="MolmoAct2 BimanualYAM endpoint readiness smoke",
        )

    def act(
        self,
        observation: Observation,
        instruction: str | None = None,
    ) -> dict[str, Any]:
        if self._policy is None:
            raise RuntimeError("MolmoAct2 BimanualYAM policy is not loaded")
        missing = {"top", "left", "right"} - set(observation.images)
        if missing:
            raise ValueError(
                "observation is missing required camera(s): " + ", ".join(sorted(missing))
            )
        state = np.asarray(observation.state, dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"state must be shape ({STATE_DIM},), got {state.shape}")
        actions = self._policy.predict(
            top_cam=decode_image(observation.images["top"]),
            left_cam=decode_image(observation.images["left"]),
            right_cam=decode_image(observation.images["right"]),
            instruction=instruction or observation.instruction or "",
            state=state,
            num_steps=DEFAULT_NUM_STEPS,
            enable_cuda_graph=self.enable_cuda_graph,
        )
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, ACTION_DIM):
            raise ValueError(
                "MolmoAct2 BimanualYAM returned an invalid action shape "
                f"{actions.shape}; expected ({ACTION_HORIZON}, {ACTION_DIM})"
            )
        if not np.isfinite(actions).all():
            raise ValueError("MolmoAct2 BimanualYAM returned non-finite actions")
        return {
            "actions": actions.tolist(),
            "horizon": ACTION_HORIZON,
            "action_dim": ACTION_DIM,
            "real": True,
            "telemetry": {
                "runtime_family": "molmoact2",
                "model_dtype": str(self.dtype),
                "normalization": {
                    "normalization_type": "quantiles",
                    "stats_key": NORM_TAG,
                },
                "action_space": "joint_position",
                "action_space_dims": {
                    "native_dim": ACTION_DIM,
                },
                "served_model": self.repo_id,
            },
        }


class ObservableYAMRuntime(VLAActionServingRuntime):
    """Record policy-level metrics around the normalized Servo runtime."""

    def prepare(self):
        _MODEL_READY.set(0)
        started = time.perf_counter()
        try:
            result = super().prepare()
        except Exception:
            _MODEL_LOAD_DURATION.set(time.perf_counter() - started)
            raise
        _MODEL_LOAD_DURATION.set(time.perf_counter() - started)
        _MODEL_READY.set(1)
        _update_cuda_metrics(self.policy.device)
        return result

    def act(self, req):
        started = time.perf_counter()
        try:
            result = super().act(req)
        except Exception:
            _ACT_REQUESTS.labels(outcome="error").inc()
            raise
        finally:
            _INFERENCE_DURATION.observe(time.perf_counter() - started)
            _update_cuda_metrics(self.policy.device)
        _ACT_REQUESTS.labels(outcome="success").inc()
        return result


def _load_endpoint_context(path: str) -> tuple[EndpointContext, dict[str, Any]]:
    config_path = Path(path).expanduser()
    try:
        bundle = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Servo verifier config at {config_path}") from exc
    if bundle.get("schema") != SERVO_VERIFIER_SCHEMA:
        raise ValueError("unsupported Servo BimanualYAM verifier schema")
    try:
        identity = EndpointIdentity.model_validate(bundle["identity"])
        metadata = ServingMetadata.model_validate(bundle["metadata"])
        public_pem = str(bundle["key"]["public_pem"])
        kid = str(bundle["key"]["kid"])
        keyset_revision = int(bundle["key"]["keyset_revision"])
        issuer = str(bundle["issuer"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Servo BimanualYAM verifier config is incomplete or invalid") from exc
    context = EndpointContext(
        identity=identity,
        verifier=EndpointTokenVerifier({kid: public_pem}, issuer=issuer),
        metadata=metadata,
        keyset_revision=keyset_revision,
    )
    return context, bundle


def build_servo_app(
    policy: ServoYAMPolicy,
    *,
    verifier_config: str,
):
    context, bundle = _load_endpoint_context(verifier_config)
    metadata = context.metadata
    if (
        metadata.family != "molmoact2"
        or set(metadata.observation.camera_keys) != {"top", "left", "right"}
        or metadata.observation.state_dim != STATE_DIM
        or metadata.action.dim != ACTION_DIM
        or metadata.action.horizon != ACTION_HORIZON
        or metadata.action.space != "joint_position"
    ):
        raise ValueError("verifier metadata is not the BimanualYAM endpoint contract")
    manifest = {
        "manifest_hash": context.identity.manifest_hash,
        "family": "molmoact2",
        "adapter_key": "molmoact2",
        "checkpoint_uri": f"hf://{policy.repo_id}",
        "checkpoint_hash": context.identity.checkpoint_digest,
        "normalization": {
            "normalization_type": "quantiles",
            "stats_key": NORM_TAG,
        },
        "action_space": {
            "type": "joint_position",
            "dim": ACTION_DIM,
            "horizon": ACTION_HORIZON,
        },
    }
    runtime = ObservableYAMRuntime(policy, manifest=manifest)
    runtime.prepare()
    app = create_vla_serving_app(runtime, context)

    @app.middleware("http")
    async def record_http_metrics(request: Request, call_next):
        route = _metrics_route(request.url.path)
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            _HTTP_REQUESTS.labels(
                method=request.method,
                route=route,
                status=str(status),
            ).inc()
            _HTTP_DURATION.labels(method=request.method, route=route).observe(
                time.perf_counter() - started
            )

    app.state.servo_credential_endpoint = bundle["endpoint_url"]
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Authenticated Servo endpoint for MolmoAct2-BimanualYAM"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8202)
    parser.add_argument("--verifier-config", required=True)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--metrics-host",
        default="127.0.0.1",
        help="Prometheus scrape listener (loopback by default)",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=9464,
        help="Prometheus scrape port; set to 0 to disable",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    policy = ServoYAMPolicy(
        repo_id=args.repo_id,
        device=args.device,
        dtype=dtype,
        enable_cuda_graph=args.cuda_graph,
    )
    _RUNTIME_INFO.info(
        {
            "model": args.repo_id,
            "device": args.device,
            "dtype": args.dtype,
        }
    )
    app = build_servo_app(policy, verifier_config=args.verifier_config)

    if args.metrics_port:
        start_http_server(args.metrics_port, addr=args.metrics_host)
        log.info(
            "Prometheus metrics ready at http://%s:%d/metrics",
            args.metrics_host,
            args.metrics_port,
        )

    import uvicorn

    log.info(
        "Servo BimanualYAM endpoint ready at %s:%d (public URL %s)",
        args.host,
        args.port,
        app.state.servo_credential_endpoint,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
