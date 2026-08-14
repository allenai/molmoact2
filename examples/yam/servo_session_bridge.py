"""Official Servo action-session bridge for the BimanualYAM eval client.

Everything that leaves this machine is official Servo SDK transport: one
control-plane action session (signed offer + lease) opened once and reused for
every action chunk of the run.  There is no custom ``/act`` endpoint, no
endpoint JWT, and no bespoke wire format on the network path.

The bridge exists only because of a local interpreter split:

* the robot runtime (``/home/npow/molmoact2-venv``) is Python 3.11 — it owns
  i2rt/CAN, pyrealsense2 and torch and must not be disturbed;
* the official ``servo`` SDK requires Python >= 3.12.

So this module is both halves of that bridge:

* :class:`ServoSessionHost` — the actual SDK usage (``Servo`` ->
  ``deployments.get`` -> ``deployment.policy`` -> ``sv.session(policy)``).
  Imported directly when the running interpreter can already import ``servo``.
* :func:`main` — a stdio request/response server, launched by
  :mod:`molmoact_client` under a Python >= 3.12 interpreter when it cannot.

The parent/child framing below is a local implementation detail (a pipe on this
host), deliberately kept trivial: a JSON header plus raw JPEG buffers so camera
frames never take a base64 round trip between the two local processes.

This module must stay importable on Python 3.11 with no ``servo`` installed:
keep the module level to the standard library and import the SDK lazily.
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# BimanualYAM contract facts (single source of truth for both processes)
# ---------------------------------------------------------------------------

#: Servo's stable camera names for the YAM embodiment, in the order the
#: released checkpoint consumes them. Servo maps these to the exact LeRobot
#: runtime keys declared by the deployment's immutable manifest.
CAMERA_KEYS: Tuple[str, str, str] = ("top", "left", "right")
STATE_DIM = 14
ACTION_HORIZON = 30
ACTION_SPACE = "joint_position"

#: Credential file written for machine (SDK) use, e.g.
#: ``~/.config/servo/molmoact2-yam-sdk.json``.
SDK_CREDENTIAL_SCHEMA = "servo.sdk-credentials.v1"

#: Environment variable naming a Python >= 3.12 interpreter that can import the
#: official ``servo`` package, used when the robot runtime cannot.
SERVO_PYTHON_ENV = "SERVO_PYTHON"

_FRAME_MAGIC = b"SVYB"
_FRAME_PREFIX = struct.Struct("<4sII")
_MAX_BUFFERS = 16


class ServoBridgeError(RuntimeError):
    """A Servo session/bridge failure with an optional remote error type."""

    def __init__(self, message: str, *, error_type: Optional[str] = None):
        super().__init__(message)
        self.error_type = error_type


# ---------------------------------------------------------------------------
# Local parent/child framing
# ---------------------------------------------------------------------------


def encode_frame(header: Mapping[str, Any], buffers: Sequence[bytes] = ()) -> bytes:
    """Encode one ``(JSON header, raw buffers)`` frame."""
    if len(buffers) > _MAX_BUFFERS:
        raise ValueError(f"a bridge frame carries at most {_MAX_BUFFERS} buffers")
    for buffer in buffers:
        if not isinstance(buffer, (bytes, bytearray)) or not buffer:
            raise ValueError("bridge frame buffers must be non-empty bytes")
        if len(buffer) > 0xFFFFFFFF:
            raise ValueError("bridge frame buffer exceeds uint32 framing")
    encoded_header = json.dumps(
        dict(header),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded_header) > 0xFFFFFFFF:
        raise ValueError("bridge frame header exceeds uint32 framing")
    lengths = struct.pack(f"<{len(buffers)}I", *(len(buffer) for buffer in buffers))
    return b"".join(
        (
            _FRAME_PREFIX.pack(_FRAME_MAGIC, len(encoded_header), len(buffers)),
            lengths,
            encoded_header,
            *(bytes(buffer) for buffer in buffers),
        )
    )


def _read_exactly(stream: Any, size: int) -> bytes:
    """Read exactly ``size`` bytes or raise; pipes may return short reads."""
    chunks: List[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("Servo bridge stream closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: Any) -> Tuple[Dict[str, Any], List[bytes]]:
    """Read one frame written by :func:`encode_frame`. Raises ``EOFError`` at end."""
    prefix = stream.read(_FRAME_PREFIX.size)
    if not prefix:
        raise EOFError("Servo bridge stream closed")
    if len(prefix) < _FRAME_PREFIX.size:
        prefix += _read_exactly(stream, _FRAME_PREFIX.size - len(prefix))
    magic, header_len, buffer_count = _FRAME_PREFIX.unpack(prefix)
    if magic != _FRAME_MAGIC:
        raise ValueError("Servo bridge frame magic does not match")
    if buffer_count > _MAX_BUFFERS:
        raise ValueError("Servo bridge frame declares too many buffers")
    lengths = (
        struct.unpack(f"<{buffer_count}I", _read_exactly(stream, 4 * buffer_count))
        if buffer_count
        else ()
    )
    header = json.loads(_read_exactly(stream, header_len).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("Servo bridge frame header must be a JSON object")
    return header, [_read_exactly(stream, length) for length in lengths]


def write_frame(stream: Any, header: Mapping[str, Any], buffers: Sequence[bytes] = ()) -> None:
    stream.write(encode_frame(header, buffers))
    stream.flush()


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def load_sdk_credentials(path: str) -> Dict[str, Any]:
    """Load and validate an SDK machine-key bundle. The secret never leaves here."""
    credential_path = Path(path).expanduser()
    try:
        payload = json.loads(credential_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ServoBridgeError(f"Servo SDK credentials cannot be read: {credential_path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SDK_CREDENTIAL_SCHEMA:
        raise ServoBridgeError(
            "Servo SDK credentials must use schema "
            f"{SDK_CREDENTIAL_SCHEMA!r} (got {payload.get('schema') if isinstance(payload, dict) else type(payload).__name__!r}); "
            "the browser session token in credentials.json is not a machine API key"
        )
    api_key = payload.get("api_key")
    base_url = payload.get("base_url")
    if not isinstance(api_key, str) or not api_key:
        raise ServoBridgeError("Servo SDK credentials are missing api_key")
    if not isinstance(base_url, str) or not base_url.startswith("https://"):
        raise ServoBridgeError("Servo SDK credentials must carry an https base_url")
    try:
        mode = credential_path.stat().st_mode
    except OSError:  # pragma: no cover - stat cannot fail after a successful read
        mode = 0
    if mode & 0o077:
        raise ServoBridgeError(
            f"Servo SDK credentials {credential_path} are group/world readable; "
            "run `chmod 600` on the file"
        )
    return payload


# ---------------------------------------------------------------------------
# The official-SDK half
# ---------------------------------------------------------------------------


class ServoSessionHost:
    """One official Servo action session, reused for every action chunk."""

    def __init__(
        self,
        *,
        credentials: str,
        deployment_id: str,
        instruction: Optional[str] = None,
        timeout_sec: Optional[float] = 600.0,
    ):
        if not deployment_id:
            raise ServoBridgeError("a managed Servo deployment id is required")
        self._credentials_path = credentials
        self.deployment_id = str(deployment_id)
        self.instruction = instruction
        self._timeout_sec = timeout_sec
        self._client: Any = None
        self._session: Any = None
        self.identity: Dict[str, Any] = {}

    def open(self) -> Dict[str, Any]:
        """Resolve the managed deployment and open one action session."""
        if self._session is not None:
            return dict(self.identity)
        try:
            from servo import Servo
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ServoBridgeError(
                "the official servo SDK is not importable in this interpreter "
                f"({sys.executable}); install servo-client or point "
                f"{SERVO_PYTHON_ENV} at a Python >= 3.12 that has it"
            ) from exc

        credentials = load_sdk_credentials(self._credentials_path)
        client = Servo(
            base_url=credentials["base_url"],
            api_key=credentials["api_key"],
            timeout=self._timeout_sec,
        )
        deployment = client.deployments.get(self.deployment_id)
        policy = deployment.policy(instruction=self.instruction)
        binding = policy.active_binding
        session = client.session(policy)
        session.open()
        if getattr(session, "_action_transport", None) is None:
            # ``Session.open`` swallows CapabilityNotAvailable from the direct
            # data plane and falls back to the central OpenPI WebSocket, which
            # re-introduces base64-encoded JPEG frames and a control-plane
            # hairpin. That is silent unless we check for it here, and it must
            # fail closed: this client exists specifically to avoid base64.
            try:
                session.close(success=False, suppress_completion_error=True)
            finally:
                closer = getattr(getattr(client, "_http", None), "close", None)
                if closer is not None:
                    try:
                        closer()
                    except Exception:  # noqa: BLE001 - teardown is best effort
                        pass
            raise ServoBridgeError(
                f"deployment {self.deployment_id!r} has no direct action-session "
                "data plane; the SDK fell back to the central WebSocket path "
                "(base64-encoded frames). Refusing to run in that mode — enable "
                "action sessions on the deployment's provider first.",
                error_type="ActionSessionDowngrade",
            )
        self._client = client
        self._session = session
        # ``_jsonable`` so the identity survives the local frame header even if
        # a future SDK returns a model object for one of these fields.
        self.identity = _jsonable({
            "deployment_id": self.deployment_id,
            "session_id": session.session_id,
            "eval_run_id": session.eval_run_id,
            "model_ref": policy.model_ref,
            "embodiment_id": policy.embodiment_id,
            "generation_id": getattr(binding, "generation_id", None),
            "checkpoint_digest": getattr(binding, "checkpoint_digest", None),
            "manifest_hash": getattr(binding, "manifest_hash", None),
            "binding_revision": getattr(binding, "binding_revision", None),
            "base_url": credentials["base_url"],
            "key_id": credentials.get("key_id"),
            "advisory": getattr(deployment, "advisory", None),
        })
        return dict(self.identity)

    def act(
        self,
        images: Mapping[str, bytes],
        state: Sequence[float],
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run one action chunk over the open session."""
        if self._session is None:
            raise ServoBridgeError("Servo session is not open")
        missing = [key for key in CAMERA_KEYS if not images.get(key)]
        if missing:
            raise ServoBridgeError(f"Servo observation is missing camera bytes: {missing}")
        state_values = [float(value) for value in state]
        if len(state_values) != STATE_DIM:
            raise ServoBridgeError(
                f"BimanualYAM state must be {STATE_DIM} floats, got {len(state_values)}"
            )
        observation = {
            "images": {key: bytes(images[key]) for key in CAMERA_KEYS},
            "state": state_values,
            "instruction": instruction or self.instruction,
        }
        prediction = self._session.act(observation, instruction=instruction)
        return self._validated_prediction(prediction)

    def _validated_prediction(self, prediction: Any) -> Dict[str, Any]:
        actions = [[float(value) for value in row] for row in (prediction.actions or [])]
        if len(actions) != ACTION_HORIZON or any(len(row) != STATE_DIM for row in actions):
            raise ServoBridgeError(
                "Servo returned an action chunk that is not "
                f"{ACTION_HORIZON}x{STATE_DIM}: "
                f"{len(actions)}x{len(actions[0]) if actions else 0}"
            )
        if any(not math.isfinite(value) for row in actions for value in row):
            raise ServoBridgeError("Servo returned a non-finite action value")
        if prediction.action_space != ACTION_SPACE:
            raise ServoBridgeError(
                f"Servo action space is {prediction.action_space!r}, expected {ACTION_SPACE!r}"
            )
        if not prediction.decoded:
            raise ServoBridgeError("Servo returned an undecoded action chunk")
        binding = prediction.binding
        binding_view = (
            binding.model_dump(mode="json")
            if binding is not None and hasattr(binding, "model_dump")
            else binding
        )
        expected_generation = self.identity.get("generation_id")
        # ``ActiveBindingView`` (self.identity, above) names this field
        # ``generation_id``; the per-response ``ActionBindingIdentity`` names
        # the same fact ``deploy_generation`` — they are not the same schema.
        actual_generation = (
            (binding_view or {}).get("deploy_generation") if binding_view else None
        )
        if expected_generation and actual_generation and actual_generation != expected_generation:
            raise ServoBridgeError(
                "Servo served a different generation than the session bound: "
                f"{actual_generation} != {expected_generation}"
            )
        return {
            "actions": actions,
            "horizon": int(prediction.horizon),
            "action_space": prediction.action_space,
            "telemetry": _jsonable(prediction.telemetry),
            "binding": _jsonable(binding_view),
            "safety_signal": _jsonable(prediction.safety_signal),
        }

    def close(self, success: bool = True) -> None:
        session, self._session = self._session, None
        if session is None:
            return
        try:
            session.close(success=success, suppress_completion_error=True)
        finally:
            client, self._client = self._client, None
            closer = getattr(getattr(client, "_http", None), "close", None)
            if closer is not None:
                try:
                    closer()
                except Exception:  # noqa: BLE001 - teardown is best effort
                    pass


class ServoDirectHost(ServoSessionHost):
    """One self-hosted ``servo serve`` endpoint over the native action session.

    Grant-only and fallback-free by construction: the grant names exactly one
    endpoint, there is no control plane to consult, no SDK credentials, and no
    fallback URL — any failure surfaces as an error instead of a reroute.
    ``act``/``_validated_prediction`` are inherited unchanged: the direct
    policy's prediction carries the same fields the hosted session returns.
    """

    def __init__(
        self,
        *,
        grant: str,
        instruction: Optional[str] = None,
        timeout_sec: Optional[float] = 600.0,
    ):
        # Deliberately NOT calling super().__init__: this host has no
        # credentials file and no managed deployment id to require.
        self._grant_path = grant
        self.deployment_id = "self-hosted"
        self.instruction = instruction
        self._timeout_sec = timeout_sec
        self._client = None
        self._session = None
        self.identity: Dict[str, Any] = {}

    def open(self) -> Dict[str, Any]:
        """Attach to the granted endpoint; the grant is the entire identity."""
        if self._session is not None:
            return dict(self.identity)
        try:
            from servo.direct import attach
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ServoBridgeError(
                "the official servo SDK is not importable in this interpreter "
                f"({sys.executable}); install servo-client or point "
                f"{SERVO_PYTHON_ENV} at a Python >= 3.12 that has it"
            ) from exc
        grant_path = Path(self._grant_path).expanduser()
        try:
            mode = grant_path.stat().st_mode
        except OSError as exc:
            raise ServoBridgeError(f"Servo grant cannot be read: {grant_path}") from exc
        if mode & 0o077:
            raise ServoBridgeError(
                f"Servo grant {grant_path} is group/world readable; it holds a "
                "private key — chmod 600 it"
            )
        policy = attach(
            grant=grant_path.read_text().strip(),
            instruction=self.instruction,
            timeout=float(self._timeout_sec or 600.0),
        )
        # DirectPolicy.act(observation, instruction=...) matches the hosted
        # Session.act call shape, so the inherited act() drives it unchanged.
        self._session = policy
        self.identity = _jsonable({
            "deployment_id": "self-hosted",
            "backend": "servo-direct-action-session",
            "grant": str(grant_path),
            "cameras": sorted(dict(getattr(policy, "camera_inputs", None) or {})),
        })
        return dict(self.identity)

    def close(self, success: bool = True) -> None:
        del success  # a self-hosted endpoint keeps no central rollout record
        session, self._session = self._session, None
        if session is None:
            return
        try:
            session.close()
        except Exception:  # noqa: BLE001 - teardown is best effort
            pass


def _jsonable(value: Any) -> Any:
    """Reduce SDK payloads to JSON the 3.11 parent can read back verbatim.

    Non-finite floats (e.g. a runtime-reported ``inf`` rate) are flattened to
    ``None`` rather than left for ``json.dumps(allow_nan=False)`` to reject —
    telemetry must never be able to kill the bridge mid-run.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


# ---------------------------------------------------------------------------
# stdio server (child process)
# ---------------------------------------------------------------------------


def _handle(host_ref: Dict[str, Any], header: Dict[str, Any], buffers: List[bytes]) -> Dict[str, Any]:
    op = header.get("op")
    host: Optional[ServoSessionHost] = host_ref.get("host")
    if op == "open":
        if host is not None:
            raise ServoBridgeError("a Servo session is already open on this bridge")
        if header.get("grant"):
            host = ServoDirectHost(
                grant=header["grant"],
                instruction=header.get("instruction"),
                timeout_sec=header.get("timeout_sec"),
            )
        else:
            host = ServoSessionHost(
                credentials=header["credentials"],
                deployment_id=header["deployment_id"],
                instruction=header.get("instruction"),
                timeout_sec=header.get("timeout_sec"),
            )
        identity = host.open()
        host_ref["host"] = host
        return {"ok": True, "identity": identity}
    if op == "act":
        if host is None:
            raise ServoBridgeError("act requested before open")
        image_index = header.get("images") or {}
        images = {key: buffers[int(index)] for key, index in image_index.items()}
        return {
            "ok": True,
            "prediction": host.act(
                images,
                header.get("state") or [],
                instruction=header.get("instruction"),
            ),
        }
    if op == "close":
        if host is not None:
            host.close(success=bool(header.get("success", True)))
            host_ref["host"] = None
        return {"ok": True}
    if op == "ping":
        try:
            import servo  # noqa: F401

            servo_importable = True
        except ImportError:
            servo_importable = False
        return {
            "ok": True,
            "executable": sys.executable,
            "python_version": sys.version.split()[0],
            "servo_importable": servo_importable,
        }
    raise ServoBridgeError(f"unknown Servo bridge op {op!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Serve bridge requests on stdin/stdout until the parent closes the pipe."""
    del argv
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    host_ref: Dict[str, Any] = {"host": None}
    try:
        while True:
            try:
                header, buffers = read_frame(stdin)
            except EOFError:
                break
            except KeyboardInterrupt:
                # Ctrl-C reaches this subprocess directly (same process
                # group as the parent launcher). Exit quietly through the
                # same path as a closed pipe instead of an unhandled
                # traceback; ``finally`` below still closes the host.
                break
            try:
                response = _handle(host_ref, header, buffers)
            except KeyboardInterrupt:
                # Same as the idle-read case above: Ctrl-C can also land
                # mid-request (e.g. inside an in-flight network act()).
                # Exit quietly rather than propagate a raw traceback; the
                # parent already treats an interrupted request as failed.
                break
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                response = {
                    "ok": False,
                    "error": str(exc) or exc.__class__.__name__,
                    "error_type": exc.__class__.__name__,
                }
            # Echo the request id so the parent can never pair a late reply
            # with a later observation.
            response["request_id"] = header.get("request_id")
            try:
                write_frame(stdout, response)
            except (TypeError, ValueError) as exc:
                # A response payload that cannot be JSON-encoded (e.g. a
                # non-finite float that slipped past ``_jsonable``) must not
                # kill this process — the caller degrades to a failed request,
                # not a dead session.
                write_frame(
                    stdout,
                    {
                        "ok": False,
                        "error": f"Servo bridge could not encode its response: {exc}",
                        "error_type": exc.__class__.__name__,
                        "request_id": header.get("request_id"),
                    },
                )
    finally:
        host = host_ref.get("host")
        if host is not None:
            try:
                host.close(success=False)
            except Exception:  # noqa: BLE001 - teardown is best effort
                pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
