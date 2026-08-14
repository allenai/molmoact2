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
host), deliberately kept trivial: a JSON header plus binary buffers so camera
frames never take a base64 round trip between the two local processes.

Two payload shapes ride that framing, chosen by the wire the action session
negotiated:

* ``images`` -- one pre-encoded JPEG buffer per camera. The JPEG wire is
  stateless, so bytes minted anywhere are valid on any session; this is the
  original path and is unchanged.
* ``frames`` -- one raw ``HxWx3`` ``uint8`` buffer per camera plus its shape.
  The h264 wire is session-stateful: an access unit is only valid against the
  decoder state its predecessor left, so it is minted by the transport at SEND
  time inside the SDK and a pre-encoded payload is refused outright. Pixels are
  therefore what has to cross this pipe -- see
  ``servo.execution.action_session_transport._encode_observation``.

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

#: Observation wires an action session can negotiate. ``jpeg`` is the default
#: and the fallback; ``h264`` is the codec wire, available only on a direct
#: (``servo serve`` grant) session because the encoder is owned by that
#: session's transport.
OBSERVATION_ENCODINGS: Tuple[str, ...] = ("jpeg", "h264")

#: Raw bridge frames are always contiguous ``HxWx3`` ``uint8``: the one shape
#: every camera in this rig produces and the only one the SDK's capture path
#: takes without a conversion of its own.
RAW_FRAME_DTYPE = "uint8"

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


def frame_parts(header: Mapping[str, Any], buffers: Sequence[bytes] = ()) -> List[Any]:
    """The pieces of one frame in wire order, ready to be written in sequence.

    Returned as pieces rather than one joined blob because the raw-pixel wire
    moves ~2 MB per act (three 360x640x3 frames). Joining first would copy
    every one of those bytes an extra time, on the act hot path, to produce a
    buffer the pipe then copies again -- so the caller writes the pieces and
    the pixels are read straight out of the capture buffer.
    """
    if len(buffers) > _MAX_BUFFERS:
        raise ValueError(f"a bridge frame carries at most {_MAX_BUFFERS} buffers")
    for buffer in buffers:
        # memoryview is a first-class payload here: the raw-pixel wire hands
        # over a zero-copy view of the capture buffer rather than a copy of it.
        if not isinstance(buffer, (bytes, bytearray, memoryview)) or not len(buffer):
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
    return [
        _FRAME_PREFIX.pack(_FRAME_MAGIC, len(encoded_header), len(buffers)),
        lengths,
        encoded_header,
        *buffers,
    ]


def encode_frame(header: Mapping[str, Any], buffers: Sequence[bytes] = ()) -> bytes:
    """Encode one ``(JSON header, raw buffers)`` frame as a single blob."""
    return b"".join(bytes(part) for part in frame_parts(header, buffers))


def raw_frame_spec(buffer_index: int, height: int, width: int) -> Dict[str, Any]:
    """Describe one raw camera buffer in an ``act`` header."""
    return {
        "buffer": int(buffer_index),
        "height": int(height),
        "width": int(width),
        "dtype": RAW_FRAME_DTYPE,
    }


def raw_frame_array(data: bytes, spec: Mapping[str, Any]) -> Any:
    """Rebuild one ``HxWx3`` ``uint8`` frame from a bridge buffer, without a copy.

    ``numpy`` is imported lazily: this module must stay importable on the 3.11
    robot runtime with nothing but the standard library at module level.
    """
    import numpy as np

    dtype = str(spec.get("dtype", RAW_FRAME_DTYPE))
    if dtype != RAW_FRAME_DTYPE:
        raise ServoBridgeError(
            f"raw bridge frames must be {RAW_FRAME_DTYPE}, got {dtype!r}"
        )
    try:
        height = int(spec["height"])
        width = int(spec["width"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ServoBridgeError("raw bridge frame is missing its height/width") from exc
    expected = height * width * 3
    if len(data) != expected:
        # A truncated or mis-shaped buffer must never be silently reinterpreted:
        # a reshape that happens to fit would hand the model a sheared image and
        # nothing downstream could tell.
        raise ServoBridgeError(
            f"raw bridge frame carries {len(data)} bytes, expected {expected} "
            f"for {height}x{width}x3 {RAW_FRAME_DTYPE}"
        )
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)


def _frame_is_empty(value: Any) -> bool:
    """True for a camera slot carrying no pixels -- bytes or array alike."""
    if value is None:
        return True
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value) == 0
    size = getattr(value, "size", None)
    if size is not None:
        return int(size) == 0
    return not value


def missing_cameras(images: Mapping[str, Any]) -> List[str]:
    """Camera keys with no usable frame, for either payload shape.

    A plain ``not images.get(key)`` cannot do this job any more: a numpy array
    has no truth value, so the check that exists to *report* a stalled camera
    would itself raise instead.
    """
    return [key for key in CAMERA_KEYS if _frame_is_empty(images.get(key))]


def _observation_frame(value: Any) -> Any:
    """Hand the SDK what it must own.

    Pre-encoded bytes ride the stateless JPEG wire verbatim. Raw pixels are
    passed through untouched so ``capture_observation`` fits them and the
    session's own encoder mints the payload -- the only order in which a
    stateful wire is safe.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return value


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
    for part in frame_parts(header, buffers):
        stream.write(part)
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
        images: Mapping[str, Any],
        state: Sequence[float],
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run one action chunk over the open session.

        ``images`` carries pre-encoded JPEG bytes (jpeg wire) or raw ``HxWx3``
        ``uint8`` arrays (codec wire); the SDK decides what to do with each.
        """
        if self._session is None:
            raise ServoBridgeError("Servo session is not open")
        missing = missing_cameras(images)
        if missing:
            raise ServoBridgeError(f"Servo observation is missing camera bytes: {missing}")
        state_values = [float(value) for value in state]
        if len(state_values) != STATE_DIM:
            raise ServoBridgeError(
                f"BimanualYAM state must be {STATE_DIM} floats, got {len(state_values)}"
            )
        observation = {
            "images": {key: _observation_frame(images[key]) for key in CAMERA_KEYS},
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
        observation_encoding: str = "jpeg",
        h264_crf: Optional[int] = None,
    ):
        # Deliberately NOT calling super().__init__: this host has no
        # credentials file and no managed deployment id to require.
        if observation_encoding not in OBSERVATION_ENCODINGS:
            raise ServoBridgeError(
                f"observation_encoding must be one of {list(OBSERVATION_ENCODINGS)}, "
                f"got {observation_encoding!r}"
            )
        if h264_crf is not None and observation_encoding != "h264":
            raise ServoBridgeError(
                "h264_crf only applies to observation_encoding='h264'"
            )
        self._grant_path = grant
        self.deployment_id = "self-hosted"
        self.instruction = instruction
        self._timeout_sec = timeout_sec
        self.observation_encoding = observation_encoding
        self.h264_crf = int(h264_crf) if h264_crf is not None else None
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
        try:
            policy = attach(
                grant=grant_path.read_text().strip(),
                instruction=self.instruction,
                timeout=float(self._timeout_sec or 600.0),
                observation_encoding=self.observation_encoding,
                h264_crf=self.h264_crf,
            )
        except TypeError as exc:
            # An SDK that predates the codec wire. Say so by name rather than
            # letting the run continue on a silently different wire -- an
            # out-of-date interpreter behind this bridge is exactly how the
            # codec wire was unreachable in the first place.
            raise ServoBridgeError(
                "the servo SDK behind this bridge cannot negotiate an "
                f"observation encoding ({sys.executable}); it predates the "
                "codec wire -- update the interpreter's servo checkout"
            ) from exc
        # DirectPolicy is LAZY: constructing it touches no network at all, so
        # without the two round trips below a dead serve, a stale grant or a
        # revoked key opens "successfully" here and fails on the FIRST ACT --
        # which in a real run is AFTER the arms are live. The launcher's
        # contract is that a bad credential, deployment or lease fails with the
        # arms still cold, so this is the only place it can be honoured.
        served, expected = self._verify_endpoint(policy)
        transport = self._open_session(policy)
        lease = getattr(transport, "lease", None)
        # DirectPolicy.act(observation, instruction=...) matches the hosted
        # Session.act call shape, so the inherited act() drives it unchanged.
        self._session = policy
        identity_fields = dict(getattr(policy.grant, "identity", None) or {})
        self.identity = _jsonable({
            "deployment_id": identity_fields.get("deployment_id") or "self-hosted",
            "backend": "servo-direct-action-session",
            "grant": str(grant_path),
            # Both of these read ``None`` on the operator's banner until they
            # are populated here; the launcher prints them immediately before
            # the arms are enabled, so "session None (generation None)" is the
            # last thing read before motion.
            "session_id": getattr(lease, "session_id", None),
            "generation_id": identity_fields.get("deploy_generation"),
            "checkpoint_digest": identity_fields.get("checkpoint_digest"),
            "manifest_hash": served or expected,
            "base_url": getattr(policy.grant, "endpoint_url", None),
            "session_route": getattr(policy.grant, "session_route", None),
            "cameras": sorted(dict(getattr(policy, "camera_inputs", None) or {})),
            "camera_inputs": dict(getattr(policy, "camera_inputs", None) or {}),
            "observation_encoding": self.observation_encoding,
            "h264_crf": self.h264_crf,
            "control_profile": dict(getattr(policy.grant, "control_profile", None) or {}),
        })
        return dict(self.identity)

    def _verify_endpoint(self, policy: Any) -> Tuple[Optional[str], Optional[str]]:
        """Prove the endpoint answers, authenticates, and serves THIS manifest."""
        endpoint = getattr(policy.grant, "endpoint_url", "the endpoint")
        try:
            metadata = dict(policy.metadata() or {})
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            self._abandon(policy)
            raise ServoBridgeError(
                f"Servo endpoint {endpoint} refused the grant at open: {exc}. "
                "A `servo serve` restart invalidates every grant it issued -- "
                "re-copy the current one from the serve host."
            ) from exc
        served = metadata.get("manifest_hash")
        expected = dict(getattr(policy.grant, "identity", None) or {}).get("manifest_hash")
        if served and expected and served != expected:
            self._abandon(policy)
            raise ServoBridgeError(
                f"Servo endpoint {endpoint} serves manifest {served} but the grant "
                f"names {expected}; this grant belongs to a previous serve"
            )
        return served, expected

    def _open_session(self, policy: Any) -> Any:
        """Open the action session itself, with the arms cold.

        This is the leg a stale grant fails on (``session control call failed
        (401)``), and it is also where an instruction-specific capture is paid
        -- off the act path, where a multi-second stall would freeze an arm.
        """
        try:
            return policy.open_action_session_transport(instruction=self.instruction)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            self._abandon(policy)
            raise ServoBridgeError(
                "Servo action session refused to open on "
                f"{getattr(policy.grant, 'endpoint_url', 'the endpoint')}: {exc}"
            ) from exc

    @staticmethod
    def _abandon(policy: Any) -> None:
        """Release a policy that failed open-time validation, then report."""
        try:
            policy.close()
        except Exception:  # noqa: BLE001 - teardown must not mask the real fault
            pass

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
        observation_encoding = header.get("observation_encoding") or "jpeg"
        if header.get("grant"):
            host = ServoDirectHost(
                grant=header["grant"],
                instruction=header.get("instruction"),
                timeout_sec=header.get("timeout_sec"),
                observation_encoding=observation_encoding,
                h264_crf=header.get("h264_crf"),
            )
        else:
            if observation_encoding != "jpeg":
                raise ServoBridgeError(
                    "a managed control-plane session sends jpeg observations; the "
                    f"{observation_encoding!r} wire is negotiated per action session "
                    "by a self-hosted `servo serve` grant (direct mode)"
                )
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
        frame_index = header.get("frames")
        if frame_index:
            images = {
                key: raw_frame_array(buffers[int(spec["buffer"])], spec)
                for key, spec in frame_index.items()
            }
        else:
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
