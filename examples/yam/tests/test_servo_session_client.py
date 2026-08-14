"""Hardware-free coverage for the official Servo action-session client path.

Nothing here touches the network, a GPU, the robot, or a real ``servo``
install: the SDK is replaced by an injected fake module (in-process) and by a
fake ``servo`` package on ``PYTHONPATH`` (subprocess bridge). The point of the
suite is to fence the parts of the contract that a live run cannot re-check:

* the local parent/child framing survives pipe fragmentation and refuses
  malformed frames;
* the SDK machine-key file is validated (schema, secret, https, 0600);
* exactly ONE action session is opened and reused for every action chunk, and
  camera frames reach ``session.act`` as raw JPEG bytes -- never base64;
* every prediction is fenced (shape, finiteness, action space, decoded flag,
  generation) before the robot could ever execute it.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import logging
import os
import struct
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path
from types import ModuleType, SimpleNamespace

from servo_session_bridge import (
    ACTION_HORIZON,
    CAMERA_KEYS,
    OBSERVATION_ENCODINGS,
    SDK_CREDENTIAL_SCHEMA,
    SERVO_PYTHON_ENV,
    STATE_DIM,
    ServoBridgeError,
    ServoDirectHost,
    ServoSessionHost,
    encode_frame,
    frame_parts,
    missing_cameras,
    raw_frame_array,
    raw_frame_spec,
    read_frame,
    write_frame,
)

try:  # numpy is present in both runtimes; the bridge only needs it for pixels.
    import numpy as np

    _HAVE_NUMPY = True
except Exception:  # pragma: no cover - numpy is a hard dep of the robot runtime
    np = None  # type: ignore[assignment]
    _HAVE_NUMPY = False

try:  # ``molmoact_client`` pulls torch/transformers; the bridge itself does not.
    from molmoact_client import MolmoActServo, ServoSessionTransport

    _CLIENT_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - reported through skipUnless below
    MolmoActServo = None  # type: ignore[assignment]
    ServoSessionTransport = None  # type: ignore[assignment]
    _CLIENT_IMPORT_ERROR = exc

_HAVE_CLIENT = _CLIENT_IMPORT_ERROR is None
_CLIENT_SKIP = f"molmoact_client is not importable here: {_CLIENT_IMPORT_ERROR!r}"

# Keep the policy logger quiet; ``setup_logger`` never resets ``disabled``.
logging.getLogger("molmoact").disabled = True
_QUIET_LOGGER = logging.getLogger("servo_session_client_tests")
_QUIET_LOGGER.addHandler(logging.NullHandler())
_QUIET_LOGGER.propagate = False
_QUIET_LOGGER.setLevel(logging.CRITICAL)

# A syntactically real (tiny) JPEG payload: SOI ... EOI. The fakes never decode
# it, so the bridge tests stay stdlib-only.
_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + bytes(
    range(256)
) + b"\xff\xd9"

_FAKE_API_KEY = "sk_servo_FAKE_do_not_use_0000"
_FAKE_BASE_URL = "https://servo.invalid"


#: Sentinel for "leave this key out of the credential file entirely".
_OMIT = object()


def _camera_blobs() -> dict:
    """Three distinguishable JPEG payloads, one per Servo camera key."""
    return {key: _JPEG + key.encode("ascii") * 7 for key in CAMERA_KEYS}


def _write_credentials(directory: str, **overrides) -> str:
    payload = {
        "schema": SDK_CREDENTIAL_SCHEMA,
        "api_key": _FAKE_API_KEY,
        "base_url": _FAKE_BASE_URL,
        "key_id": "key_test_0001",
        "label": "molmoact2-yam-tests",
    }
    payload.update(overrides)
    payload = {key: value for key, value in payload.items() if value is not _OMIT}
    path = Path(directory) / "molmoact2-yam-sdk.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    return str(path)


def _iter_strings(value):
    """Yield every string (keys included) reachable inside ``value``."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _assert_no_base64_images(test, observation, blobs) -> None:
    """No camera frame may appear as base64/data-URI text anywhere."""
    encoded = [base64.b64encode(blob).decode("ascii") for blob in blobs]
    for text in _iter_strings(observation):
        test.assertFalse(
            text.startswith("data:"),
            msg=f"observation carries a data URI: {text[:40]!r}",
        )
        for candidate in encoded:
            test.assertNotIn(candidate[:48], text)
            test.assertNotIn(candidate, text)


# ---------------------------------------------------------------------------
# 1. Local parent/child framing
# ---------------------------------------------------------------------------


class _DribbleStream:
    """A stream that hands back exactly one byte per ``read`` (pipe worst case)."""

    def __init__(self, payload: bytes):
        self._payload = payload
        self._offset = 0
        self.reads = 0

    def read(self, size: int) -> bytes:
        self.reads += 1
        if size <= 0 or self._offset >= len(self._payload):
            return b""
        chunk = self._payload[self._offset : self._offset + 1]
        self._offset += 1
        return chunk


class FrameCodecTests(unittest.TestCase):
    def test_round_trip_without_buffers(self):
        header = {"op": "ping", "nested": {"list": [1, 2, 3]}, "unicode": "café"}
        decoded_header, buffers = read_frame(io.BytesIO(encode_frame(header)))
        self.assertEqual(decoded_header, header)
        self.assertEqual(buffers, [])

    def test_round_trip_with_three_binary_buffers(self):
        blobs = list(_camera_blobs().values())
        frame = encode_frame({"op": "act", "images": {"top": 0, "left": 1, "right": 2}}, blobs)
        decoded_header, decoded_buffers = read_frame(io.BytesIO(frame))
        self.assertEqual(decoded_header["op"], "act")
        self.assertEqual(decoded_buffers, blobs)
        for buffer in decoded_buffers:
            self.assertIsInstance(buffer, bytes)
            self.assertTrue(buffer.startswith(b"\xff\xd8"))
        # The frame is byte-transparent: the raw JPEGs are embedded verbatim.
        for blob in blobs:
            self.assertIn(blob, frame)
            self.assertNotIn(base64.b64encode(blob), frame)

    def test_short_reads_are_reassembled(self):
        blobs = list(_camera_blobs().values())
        stream = _DribbleStream(encode_frame({"op": "act", "n": 3}, blobs))
        decoded_header, decoded_buffers = read_frame(stream)
        self.assertEqual(decoded_header, {"op": "act", "n": 3})
        self.assertEqual(decoded_buffers, blobs)
        # One byte per read: the codec cannot have relied on a single read().
        self.assertGreater(stream.reads, len(blobs[0]))

    def test_encode_rejects_empty_and_non_bytes_buffers(self):
        with self.assertRaises(ValueError):
            encode_frame({"op": "act"}, [b""])
        with self.assertRaises(ValueError):
            encode_frame({"op": "act"}, ["not-bytes"])
        with self.assertRaises(ValueError):
            encode_frame({"op": "act"}, [_JPEG, b"", _JPEG])

    def test_encode_rejects_too_many_buffers(self):
        with self.assertRaises(ValueError):
            encode_frame({"op": "act"}, [b"x"] * 17)
        # 16 is the documented ceiling and must still encode.
        self.assertTrue(encode_frame({"op": "act"}, [b"x"] * 16))

    def test_encode_rejects_non_finite_header_values(self):
        with self.assertRaises(ValueError):
            encode_frame({"op": "act", "state": [float("nan")] * STATE_DIM})

    def test_read_rejects_bad_magic_and_oversized_buffer_count(self):
        with self.assertRaisesRegex(ValueError, "magic"):
            read_frame(io.BytesIO(struct.pack("<4sII", b"XXXX", 2, 0) + b"{}"))
        with self.assertRaisesRegex(ValueError, "too many buffers"):
            read_frame(io.BytesIO(struct.pack("<4sII", b"SVYB", 2, 17) + b"{}"))

    def test_read_rejects_a_non_object_header(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            read_frame(io.BytesIO(struct.pack("<4sII", b"SVYB", 2, 0) + b"[]"))

    def test_read_reports_eof_and_truncation(self):
        with self.assertRaises(EOFError):
            read_frame(io.BytesIO(b""))
        truncated = encode_frame({"op": "act"}, [_JPEG])[:-1]
        with self.assertRaises(EOFError):
            read_frame(io.BytesIO(truncated))
        with self.assertRaises(EOFError):
            read_frame(io.BytesIO(encode_frame({"op": "act"})[:5]))

    def test_write_frame_streams_a_readable_frame(self):
        buffer = io.BytesIO()
        write_frame(buffer, {"ok": True, "identity": {"session_id": "sess_1"}}, [_JPEG])
        buffer.seek(0)
        header, buffers = read_frame(buffer)
        self.assertTrue(header["ok"])
        self.assertEqual(buffers, [_JPEG])


# ---------------------------------------------------------------------------
# 2. SDK machine-key credentials
# ---------------------------------------------------------------------------


class SdkCredentialTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.directory = self._dir.name

    def _load(self, **overrides):
        from servo_session_bridge import load_sdk_credentials

        return load_sdk_credentials(_write_credentials(self.directory, **overrides))

    def test_accepts_a_valid_0600_bundle(self):
        payload = self._load()
        self.assertEqual(payload["schema"], SDK_CREDENTIAL_SCHEMA)
        self.assertEqual(payload["base_url"], _FAKE_BASE_URL)
        self.assertEqual(payload["key_id"], "key_test_0001")
        self.assertTrue(payload["api_key"])

    def test_rejects_the_browser_session_credentials(self):
        with self.assertRaisesRegex(ServoBridgeError, "schema"):
            self._load(schema="servo.cli-credentials.v1")
        with self.assertRaisesRegex(ServoBridgeError, "schema"):
            self._load(schema=_OMIT)

    def test_rejects_a_missing_or_empty_api_key(self):
        with self.assertRaisesRegex(ServoBridgeError, "api_key"):
            self._load(api_key=_OMIT)
        with self.assertRaisesRegex(ServoBridgeError, "api_key"):
            self._load(api_key="")
        with self.assertRaisesRegex(ServoBridgeError, "api_key"):
            self._load(api_key=1234)

    def test_rejects_a_non_https_base_url(self):
        with self.assertRaisesRegex(ServoBridgeError, "https"):
            self._load(base_url="http://servo.invalid")
        with self.assertRaisesRegex(ServoBridgeError, "https"):
            self._load(base_url=_OMIT)

    def test_rejects_a_group_or_world_readable_file(self):
        from servo_session_bridge import load_sdk_credentials

        path = _write_credentials(self.directory)
        Path(path).chmod(0o644)
        with self.assertRaisesRegex(ServoBridgeError, "group/world readable"):
            load_sdk_credentials(path)
        Path(path).chmod(0o604)
        with self.assertRaisesRegex(ServoBridgeError, "group/world readable"):
            load_sdk_credentials(path)
        Path(path).chmod(0o600)
        self.assertTrue(load_sdk_credentials(path))

    def test_rejects_missing_and_malformed_files(self):
        from servo_session_bridge import load_sdk_credentials

        missing = str(Path(self.directory) / "nope.json")
        with self.assertRaisesRegex(ServoBridgeError, "cannot be read"):
            load_sdk_credentials(missing)
        broken = Path(self.directory) / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        broken.chmod(0o600)
        with self.assertRaisesRegex(ServoBridgeError, "cannot be read"):
            load_sdk_credentials(str(broken))


# ---------------------------------------------------------------------------
# 3. ServoSessionHost against a fake official SDK
# ---------------------------------------------------------------------------


class _FakeBinding:
    """Doubles both real SDK shapes that carry a generation:

    * ``policy.active_binding`` (``ActiveBindingView``) is read by direct
      attribute access (``getattr(binding, "generation_id", ...)``) and its
      real field IS named ``generation_id``.
    * ``prediction.binding`` (``ActionBindingIdentity``) is read through
      ``.model_dump()`` and its real field is ``deploy_generation`` (a
      *different* name for the same fact) with no ``manifest_hash`` at all —
      see servo_contracts/inference.py:ActionBindingIdentity.
    """

    def __init__(
        self,
        generation_id="gen_test_1",
        checkpoint_digest="sha256:" + "a" * 64,
        manifest_hash="sha256:" + "b" * 64,
        binding_revision=3,
        deployment_id="dep_test",
    ):
        self.generation_id = generation_id
        self.checkpoint_digest = checkpoint_digest
        self.manifest_hash = manifest_hash
        self.binding_revision = binding_revision
        self.deployment_id = deployment_id

    def model_dump(self, mode="json"):
        return {
            "deployment_id": self.deployment_id,
            "deploy_generation": self.generation_id,
            "checkpoint_digest": self.checkpoint_digest,
            "binding_revision": self.binding_revision,
        }


def _fake_prediction(**overrides):
    fields = {
        "actions": [
            [float(row) + column / 100.0 for column in range(STATE_DIM)]
            for row in range(ACTION_HORIZON)
        ],
        "horizon": ACTION_HORIZON,
        "action_space": "joint_position",
        "decoded": True,
        # Flattened shape of the real SDK's InferenceTelemetry
        # (servo/types.py:_telemetry_view) — the key is server_infer_ms, not
        # a raw "infer_ms" passthrough.
        "telemetry": {"server_infer_ms": 12.5, "real": True},
        "binding": _FakeBinding(),
        "safety_signal": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _install_fake_servo(
    test, *, prediction_factory=None, advisory="p95 480 ms", direct_transport=True
):
    """Inject a fake ``servo`` module and return a recorder for what it saw."""

    state = SimpleNamespace(clients=[], sessions=[], deployments=[], policies=[])
    factory = prediction_factory or (lambda index: _fake_prediction())

    class _FakeHttp:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    class _FakePolicy:
        def __init__(self, deployment, instruction):
            self.model_ref = "molmoact2"
            self.embodiment_id = "emb_yam_test"
            self.instruction = instruction
            self.deployment_id = deployment.id
            self.active_binding = deployment.binding

    class _FakeDeployment:
        def __init__(self, deployment_id):
            self.id = deployment_id
            self.status = "active"
            self.advisory = advisory
            self.binding = _FakeBinding()
            self.policy_instructions = []

        def policy(self, robot=None, *, instruction=None):
            self.policy_instructions.append(instruction)
            policy = _FakePolicy(self, instruction)
            state.policies.append(policy)
            return policy

    class _FakeDeployments:
        def __init__(self):
            self.requested = []

        def get(self, deployment_id):
            self.requested.append(deployment_id)
            deployment = _FakeDeployment(deployment_id)
            state.deployments.append(deployment)
            return deployment

    class _FakeSession:
        def __init__(self, policy):
            self.policy = policy
            self.session_id = "sess_test_1"
            self.eval_run_id = "run_test_1"
            self.opens = 0
            self.acts = []
            self.closes = []
            # Mirrors the real Session: non-None only once the direct data
            # plane negotiates (servo/execution/session.py:72-90). ``None``
            # here is exactly the state that must make ServoSessionHost.open
            # refuse the central-WebSocket/base64 fallback.
            self._action_transport = "fake-direct-transport" if direct_transport else None

        def open(self):
            self.opens += 1
            return self

        def act(self, observation, instruction=None):
            self.acts.append((observation, instruction))
            return factory(len(self.acts) - 1)

        def close(self, success=True, suppress_completion_error=False):
            self.closes.append(
                {"success": success, "suppress_completion_error": suppress_completion_error}
            )

    class _FakeServo:
        def __init__(self, *, base_url, api_key, timeout=None):
            self.base_url = base_url
            self.api_key_present = bool(api_key)
            self.timeout = timeout
            self.deployments = _FakeDeployments()
            self._http = _FakeHttp()
            state.clients.append(self)

        def session(self, policy):
            session = _FakeSession(policy)
            state.sessions.append(session)
            return session

    module = ModuleType("servo")
    module.__spec__ = importlib.machinery.ModuleSpec("servo", loader=None)
    module.Servo = _FakeServo
    previous = sys.modules.get("servo")
    sys.modules["servo"] = module

    def _restore():
        if previous is None:
            sys.modules.pop("servo", None)
        else:  # pragma: no cover - only if a real servo was already imported
            sys.modules["servo"] = previous

    test.addCleanup(_restore)
    return state


class ServoSessionHostTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.credentials = _write_credentials(self._dir.name)

    def _host(self, **kwargs):
        kwargs.setdefault("credentials", self.credentials)
        kwargs.setdefault("deployment_id", "dep_test")
        kwargs.setdefault("instruction", "pick up the red cap")
        return ServoSessionHost(**kwargs)

    def test_a_deployment_id_is_required(self):
        with self.assertRaisesRegex(ServoBridgeError, "deployment id"):
            ServoSessionHost(credentials=self.credentials, deployment_id="")

    def test_one_session_is_opened_and_reused_for_every_chunk(self):
        state = _install_fake_servo(self)
        host = self._host()

        identity = host.open()
        self.assertEqual(identity["deployment_id"], "dep_test")
        self.assertEqual(identity["session_id"], "sess_test_1")
        self.assertEqual(identity["eval_run_id"], "run_test_1")
        self.assertEqual(identity["model_ref"], "molmoact2")
        self.assertEqual(identity["embodiment_id"], "emb_yam_test")
        self.assertEqual(identity["generation_id"], "gen_test_1")
        self.assertEqual(identity["checkpoint_digest"], "sha256:" + "a" * 64)
        self.assertEqual(identity["manifest_hash"], "sha256:" + "b" * 64)
        self.assertEqual(identity["binding_revision"], 3)
        self.assertEqual(identity["base_url"], _FAKE_BASE_URL)
        self.assertEqual(identity["key_id"], "key_test_0001")
        self.assertEqual(identity["advisory"], "p95 480 ms")
        # The identity is a copy: callers cannot mutate the host's record.
        identity["session_id"] = "tampered"
        self.assertEqual(host.identity["session_id"], "sess_test_1")

        # Re-opening is a no-op, not a second control-plane session.
        self.assertEqual(host.open()["session_id"], "sess_test_1")

        blobs = _camera_blobs()
        for index in range(3):
            result = host.act(blobs, [float(index)] * STATE_DIM)
            self.assertEqual(len(result["actions"]), ACTION_HORIZON)
            self.assertEqual(len(result["actions"][0]), STATE_DIM)
            self.assertEqual(result["action_space"], "joint_position")
            self.assertEqual(result["horizon"], ACTION_HORIZON)
            self.assertEqual(result["telemetry"], {"server_infer_ms": 12.5, "real": True})
            self.assertEqual(result["binding"]["deploy_generation"], "gen_test_1")

        self.assertEqual(len(state.clients), 1, "more than one SDK client was built")
        self.assertEqual(len(state.deployments), 1)
        self.assertEqual(state.deployments[0].policy_instructions, ["pick up the red cap"])
        self.assertEqual(len(state.sessions), 1, "the session was not reused")
        session = state.sessions[0]
        self.assertEqual(session.opens, 1)
        self.assertEqual(len(session.acts), 3)
        self.assertIs(session.policy, state.policies[0])
        self.assertTrue(state.clients[0].api_key_present)
        self.assertEqual(state.clients[0].base_url, _FAKE_BASE_URL)

    def test_a_transport_downgrade_to_the_central_websocket_is_refused(self):
        # Session.open() swallows CapabilityNotAvailable from the direct data
        # plane and silently falls back to the base64/msgpack central
        # WebSocket path (servo/execution/session.py:72-90). That fallback
        # defeats the entire point of this client and must fail closed.
        state = _install_fake_servo(self, direct_transport=False)
        host = self._host()
        with self.assertRaisesRegex(ServoBridgeError, "no direct action-session data plane"):
            host.open()
        session = state.sessions[0]
        self.assertEqual(session.opens, 1)
        # The half-open session must be completed as a failure, not abandoned.
        self.assertEqual(session.closes, [{"success": False, "suppress_completion_error": True}])
        self.assertEqual(state.clients[0]._http.closed, 1)

    def test_observations_carry_raw_jpeg_bytes_and_a_14_float_state(self):
        state = _install_fake_servo(self)
        host = self._host()
        host.open()
        blobs = _camera_blobs()
        host.act(blobs, [0.5] * STATE_DIM, instruction="close the drawer")

        observation, instruction = state.sessions[0].acts[0]
        self.assertEqual(instruction, "close the drawer")
        self.assertEqual(set(observation["images"]), set(CAMERA_KEYS))
        self.assertEqual(list(observation["images"]), list(CAMERA_KEYS))
        for key in CAMERA_KEYS:
            frame = observation["images"][key]
            self.assertIsInstance(frame, bytes)
            self.assertTrue(frame.startswith(b"\xff\xd8"))
            self.assertEqual(frame, blobs[key])
        self.assertEqual(observation["state"], [0.5] * STATE_DIM)
        self.assertEqual(len(observation["state"]), STATE_DIM)
        for value in observation["state"]:
            self.assertIsInstance(value, float)
        self.assertEqual(observation["instruction"], "close the drawer")
        _assert_no_base64_images(self, observation, blobs.values())

    def test_the_observation_instruction_falls_back_to_the_session_instruction(self):
        state = _install_fake_servo(self)
        host = self._host()
        host.open()
        host.act(_camera_blobs(), [0.0] * STATE_DIM)
        observation, instruction = state.sessions[0].acts[0]
        self.assertEqual(observation["instruction"], "pick up the red cap")
        self.assertIsNone(instruction)

    def test_state_is_coerced_from_any_real_sequence(self):
        state = _install_fake_servo(self)
        host = self._host()
        host.open()
        host.act(_camera_blobs(), tuple(range(STATE_DIM)))
        observation, _ = state.sessions[0].acts[0]
        self.assertEqual(observation["state"], [float(i) for i in range(STATE_DIM)])

    def test_close_completes_the_session_once_and_is_idempotent(self):
        state = _install_fake_servo(self)
        host = self._host()
        host.open()
        host.close(success=True)
        host.close(success=True)
        session = state.sessions[0]
        self.assertEqual(
            session.closes,
            [{"success": True, "suppress_completion_error": True}],
        )
        self.assertEqual(state.clients[0]._http.closed, 1)
        with self.assertRaisesRegex(ServoBridgeError, "not open"):
            host.act(_camera_blobs(), [0.0] * STATE_DIM)

    def test_close_reports_failure_outcomes(self):
        state = _install_fake_servo(self)
        host = self._host()
        host.open()
        host.close(success=False)
        self.assertEqual(state.sessions[0].closes[0]["success"], False)

    def test_close_without_open_is_a_no_op(self):
        _install_fake_servo(self)
        self._host().close()

    def test_act_before_open_is_refused(self):
        _install_fake_servo(self)
        with self.assertRaisesRegex(ServoBridgeError, "not open"):
            self._host().act(_camera_blobs(), [0.0] * STATE_DIM)

    # -- fenced failure paths -------------------------------------------------

    def _open_host_with(self, **prediction_overrides):
        _install_fake_servo(
            self, prediction_factory=lambda index: _fake_prediction(**prediction_overrides)
        )
        host = self._host()
        host.open()
        return host

    def test_a_short_action_chunk_is_refused(self):
        host = self._open_host_with(actions=[[0.0] * STATE_DIM for _ in range(29)])
        with self.assertRaisesRegex(ServoBridgeError, "30x14"):
            host.act(_camera_blobs(), [0.0] * STATE_DIM)

    def test_a_narrow_action_chunk_is_refused(self):
        host = self._open_host_with(
            actions=[[0.0] * 7 for _ in range(ACTION_HORIZON)]
        )
        with self.assertRaisesRegex(ServoBridgeError, "30x14"):
            host.act(_camera_blobs(), [0.0] * STATE_DIM)

    def test_an_empty_action_chunk_is_refused(self):
        host = self._open_host_with(actions=None)
        with self.assertRaisesRegex(ServoBridgeError, "30x14"):
            host.act(_camera_blobs(), [0.0] * STATE_DIM)

    def test_a_non_finite_action_is_refused(self):
        actions = [[0.0] * STATE_DIM for _ in range(ACTION_HORIZON)]
        actions[7][3] = float("nan")
        host = self._open_host_with(actions=actions)
        with self.assertRaisesRegex(ServoBridgeError, "non-finite"):
            host.act(_camera_blobs(), [0.0] * STATE_DIM)

        actions[7][3] = float("inf")
        host = self._open_host_with(actions=actions)
        with self.assertRaisesRegex(ServoBridgeError, "non-finite"):
            host.act(_camera_blobs(), [0.0] * STATE_DIM)

    def test_a_foreign_action_space_is_refused(self):
        host = self._open_host_with(action_space="end_effector_pose")
        with self.assertRaisesRegex(ServoBridgeError, "action space"):
            host.act(_camera_blobs(), [0.0] * STATE_DIM)

    def test_an_undecoded_chunk_is_refused(self):
        host = self._open_host_with(decoded=False)
        with self.assertRaisesRegex(ServoBridgeError, "undecoded"):
            host.act(_camera_blobs(), [0.0] * STATE_DIM)

    def test_a_generation_switch_under_the_session_is_refused(self):
        host = self._open_host_with(binding=_FakeBinding(generation_id="gen_other"))
        with self.assertRaisesRegex(ServoBridgeError, "different generation"):
            host.act(_camera_blobs(), [0.0] * STATE_DIM)

    def test_missing_camera_bytes_are_refused(self):
        _install_fake_servo(self)
        host = self._host()
        host.open()
        for broken in (
            {key: value for key, value in _camera_blobs().items() if key != "left"},
            {**_camera_blobs(), "right": b""},
            {},
        ):
            with self.assertRaisesRegex(ServoBridgeError, "missing camera bytes"):
                host.act(broken, [0.0] * STATE_DIM)

    def test_a_wrong_length_state_is_refused(self):
        _install_fake_servo(self)
        host = self._host()
        host.open()
        for state in ([0.0] * 13, [0.0] * 15, []):
            with self.assertRaisesRegex(ServoBridgeError, "14 floats"):
                host.act(_camera_blobs(), state)


# ---------------------------------------------------------------------------
# 4. The subprocess bridge, end to end
# ---------------------------------------------------------------------------

_FAKE_SERVO_PACKAGE = '''\
"""Deterministic stand-in for the official servo SDK (bridge subprocess test)."""

import hashlib
import json
import os
import time

_TRACE = os.environ.get("FAKE_SERVO_TRACE")
_SLEEP = float(os.environ.get("FAKE_SERVO_ACT_SLEEP", "0") or 0)


def _trace(event, **fields):
    if not _TRACE:
        return
    record = {"event": event, "pid": os.getpid()}
    record.update(fields)
    with open(_TRACE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\\n")


class _Binding:
    # ``generation_id`` is what ActiveBindingView (policy.active_binding)
    # really names this field (getattr, not model_dump); ActionBindingIdentity
    # (prediction.binding, via model_dump below) names the same fact
    # ``deploy_generation`` and carries no manifest_hash.
    deployment_id = "dep_bridge"
    generation_id = "gen_bridge_1"
    checkpoint_digest = "sha256:" + "c" * 64
    manifest_hash = "sha256:" + "d" * 64
    binding_revision = 11

    def model_dump(self, mode="json"):
        return {
            "deployment_id": self.deployment_id,
            "deploy_generation": self.generation_id,
            "checkpoint_digest": self.checkpoint_digest,
            "binding_revision": self.binding_revision,
        }


class _Policy:
    def __init__(self, deployment_id, instruction):
        self.model_ref = "molmoact2"
        self.embodiment_id = "emb_yam_bridge"
        self.deployment_id = deployment_id
        self.instruction = instruction
        self.active_binding = _Binding()


class _Deployment:
    def __init__(self, deployment_id):
        self.id = deployment_id
        self.status = "active"
        self.advisory = "bridge advisory"

    def policy(self, robot=None, *, instruction=None):
        _trace("policy", deployment_id=self.id, instruction=instruction)
        return _Policy(self.id, instruction)


class _Deployments:
    def get(self, deployment_id):
        _trace("deployments.get", deployment_id=deployment_id)
        return _Deployment(deployment_id)


class _Prediction:
    def __init__(self, actions, telemetry):
        self.actions = actions
        self.horizon = len(actions)
        self.action_space = "joint_position"
        self.decoded = True
        self.telemetry = telemetry
        self.binding = _Binding()
        self.safety_signal = None


class _Http:
    def close(self):
        _trace("http.close")


class _Session:
    def __init__(self, policy):
        self.policy = policy
        self.session_id = "sess_bridge_%d" % os.getpid()
        self.eval_run_id = "run_bridge"
        self._calls = 0
        # Non-None so ServoSessionHost.open() accepts this as a real direct
        # transport rather than refusing it as a WebSocket-fallback downgrade.
        self._action_transport = "fake-direct-transport"

    def open(self):
        _trace("session.open", session_id=self.session_id)
        return self

    def act(self, observation, instruction=None):
        if _SLEEP:
            time.sleep(_SLEEP)
        images = observation["images"]
        for name, blob in images.items():
            if not isinstance(blob, (bytes, bytearray)):
                raise AssertionError("camera %r arrived as %s" % (name, type(blob).__name__))
        digests = {
            name: hashlib.sha256(bytes(blob)).hexdigest() for name, blob in images.items()
        }
        state = [float(value) for value in observation["state"]]
        index = self._calls
        self._calls += 1
        _trace("session.act", session_id=self.session_id, index=index)
        actions = [
            [index + row / 100.0 + column / 10000.0 for column in range(14)]
            for row in range(30)
        ]
        telemetry = {
            "session_id": self.session_id,
            "call_index": index,
            "pid": os.getpid(),
            "server_infer_ms": 3.25,
            "image_keys": list(images),
            "image_digests": digests,
            "image_bytes": {name: len(images[name]) for name in images},
            "instruction": instruction,
            "observation_instruction": observation.get("instruction"),
            "state": state,
        }
        return _Prediction(actions, telemetry)

    def close(self, success=True, suppress_completion_error=False):
        _trace(
            "session.close",
            session_id=self.session_id,
            success=success,
            suppress_completion_error=suppress_completion_error,
        )


class Servo:
    def __init__(self, *, base_url, api_key, timeout=None):
        self.base_url = base_url
        self.deployments = _Deployments()
        self._http = _Http()
        _trace(
            "client",
            base_url=base_url,
            api_key_present=bool(api_key),
            api_key_len=len(api_key or ""),
            timeout=timeout,
        )

    def session(self, policy):
        return _Session(policy)
'''


@unittest.skipUnless(_HAVE_CLIENT, _CLIENT_SKIP)
class ServoSessionTransportTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = self._dir.name
        self.credentials = _write_credentials(self.root)
        package = Path(self.root) / "fakepkg" / "servo"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(_FAKE_SERVO_PACKAGE, encoding="utf-8")
        self.package_root = str(package.parent)
        self.trace = str(Path(self.root) / "trace.jsonl")

    def _transport(self, **kwargs):
        kwargs.setdefault("credentials", self.credentials)
        kwargs.setdefault("deployment_id", "dep_test")
        kwargs.setdefault("instruction", "pick up the red cap")
        kwargs.setdefault("servo_python", sys.executable)
        kwargs.setdefault("logger", _QUIET_LOGGER)
        env = {"PYTHONPATH": self.package_root, "FAKE_SERVO_TRACE": self.trace}
        env.update(kwargs.pop("extra_env", {}))
        kwargs.setdefault("bridge_env", env)
        transport = ServoSessionTransport(**kwargs)
        self.addCleanup(self._safe_close, transport)
        return transport

    @staticmethod
    def _safe_close(transport):
        try:
            transport.close(success=False)
        except Exception:  # noqa: BLE001 - cleanup only
            pass

    def _trace_events(self):
        path = Path(self.trace)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def test_one_session_serves_every_chunk_through_the_child_interpreter(self):
        transport = self._transport()
        self.assertEqual(transport.mode, "subprocess")
        started = time.perf_counter()

        identity = transport.open()
        self.assertEqual(identity["deployment_id"], "dep_test")
        self.assertTrue(identity["session_id"].startswith("sess_bridge_"))
        self.assertEqual(identity["generation_id"], "gen_bridge_1")
        self.assertEqual(identity["model_ref"], "molmoact2")
        self.assertEqual(identity["base_url"], _FAKE_BASE_URL)
        self.assertEqual(identity["key_id"], "key_test_0001")

        process = transport._process
        self.assertIsNotNone(process)
        self.assertNotEqual(process.pid, os.getpid())

        blobs = _camera_blobs()
        results = [
            transport.act(blobs, [float(step)] * STATE_DIM, instruction="step %d" % step)
            for step in range(2)
        ]

        for index, result in enumerate(results):
            expected = [
                [index + row / 100.0 + column / 10000.0 for column in range(STATE_DIM)]
                for row in range(ACTION_HORIZON)
            ]
            self.assertEqual(result["actions"], expected, "actions did not survive the pipe")
            self.assertEqual(result["action_space"], "joint_position")
            self.assertEqual(result["horizon"], ACTION_HORIZON)
            telemetry = result["telemetry"]
            self.assertEqual(telemetry["call_index"], index)
            self.assertEqual(telemetry["instruction"], "step %d" % index)
            self.assertEqual(telemetry["image_keys"], list(CAMERA_KEYS))
            self.assertEqual(
                telemetry["image_digests"],
                {key: hashlib.sha256(blobs[key]).hexdigest() for key in CAMERA_KEYS},
                "camera bytes were altered on the way to the SDK process",
            )
            self.assertEqual(
                telemetry["image_bytes"], {key: len(blobs[key]) for key in CAMERA_KEYS}
            )
            self.assertEqual(telemetry["state"], [float(index)] * STATE_DIM)
            self.assertEqual(telemetry["pid"], process.pid)
            self.assertNotEqual(telemetry["pid"], os.getpid())

        self.assertEqual(
            results[0]["telemetry"]["session_id"],
            results[1]["telemetry"]["session_id"],
            "the two chunks did not run on the same session",
        )
        self.assertEqual(results[0]["telemetry"]["session_id"], identity["session_id"])

        transport.close(success=True)
        self.assertIsNotNone(process.poll(), "the bridge subprocess did not exit")
        self.assertEqual(transport.identity, {})

        events = self._trace_events()
        kinds = [event["event"] for event in events]
        self.assertEqual(kinds.count("client"), 1)
        self.assertEqual(kinds.count("session.open"), 1, "more than one session was opened")
        self.assertEqual(kinds.count("session.act"), 2)
        self.assertEqual(kinds.count("session.close"), 1)
        self.assertEqual(kinds.count("http.close"), 1)
        client_event = events[kinds.index("client")]
        self.assertEqual(client_event["base_url"], _FAKE_BASE_URL)
        self.assertTrue(client_event["api_key_present"])
        self.assertEqual(client_event["api_key_len"], len(_FAKE_API_KEY))
        self.assertNotEqual(client_event["pid"], os.getpid())
        close_event = events[kinds.index("session.close")]
        self.assertTrue(close_event["success"])
        self.assertTrue(close_event["suppress_completion_error"])
        self.assertLess(time.perf_counter() - started, 30.0)

    def test_a_child_side_failure_surfaces_as_a_bridge_error(self):
        transport = self._transport()
        transport.open()
        with self.assertRaisesRegex(ServoBridgeError, "14 floats"):
            transport.act(_camera_blobs(), [0.0] * 13)
        transport.close(success=False)

    def test_a_stalled_child_times_out_instead_of_hanging(self):
        transport = self._transport(
            act_timeout_sec=0.4,
            extra_env={"FAKE_SERVO_ACT_SLEEP": "1.5"},
        )
        transport.open()
        started = time.perf_counter()
        with self.assertRaisesRegex(ServoBridgeError, "did not answer"):
            transport.act(_camera_blobs(), [0.0] * STATE_DIM)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 1.4, "the timeout did not fire promptly")
        self.assertGreaterEqual(elapsed, 0.4)

        # The late reply must never be handed to the NEXT observation: after a
        # timeout the transport fails closed instead of returning stale actions.
        time.sleep(1.3)
        retry_started = time.perf_counter()
        with self.assertRaises(ServoBridgeError):
            transport.act(_camera_blobs(), [1.0] * STATE_DIM)
        self.assertLess(
            time.perf_counter() - retry_started,
            0.4,
            "a timed-out transport must fail fast, not wait for another reply",
        )
        transport.close(success=False)

    def test_a_missing_interpreter_is_refused_up_front(self):
        with self.assertRaisesRegex(ServoBridgeError, "interpreter not found"):
            ServoSessionTransport(
                credentials=self.credentials,
                deployment_id="dep_test",
                servo_python=str(Path(self.root) / "no-such-python"),
                logger=_QUIET_LOGGER,
            )

    def test_a_non_positive_timeout_is_refused(self):
        for bad in (0, -1.0):
            for kwarg in ("open_timeout_sec", "act_timeout_sec"):
                with self.assertRaises(ValueError):
                    ServoSessionTransport(
                        credentials=self.credentials,
                        deployment_id="dep_test",
                        servo_python=sys.executable,
                        logger=_QUIET_LOGGER,
                        **{kwarg: bad},
                    )

    def test_the_interpreter_choice_is_explicit(self):
        environment = {key: value for key, value in os.environ.items()}
        environment.pop(SERVO_PYTHON_ENV, None)
        with unittest.mock.patch.dict(os.environ, environment, clear=True):
            if importlib.util.find_spec("servo") is not None:  # pragma: no cover
                transport = ServoSessionTransport(
                    credentials=self.credentials,
                    deployment_id="dep_test",
                    logger=_QUIET_LOGGER,
                )
                self.assertEqual(transport.mode, "in-process")
            else:
                with self.assertRaisesRegex(ServoBridgeError, "not importable"):
                    ServoSessionTransport(
                        credentials=self.credentials,
                        deployment_id="dep_test",
                        logger=_QUIET_LOGGER,
                    )

    def test_requests_before_spawn_are_refused(self):
        transport = self._transport()
        with self.assertRaisesRegex(ServoBridgeError, "not running"):
            transport._request({"op": "ping"}, timeout=1.0)


# ---------------------------------------------------------------------------
# 5. MolmoActServo over a stub transport
# ---------------------------------------------------------------------------


class _StubTransport:
    """Stands in for :class:`ServoSessionTransport` (no SDK, no subprocess)."""

    def __init__(self, prediction=None, deployment_id="dep_test"):
        self.deployment_id = deployment_id
        self.identity = {
            "deployment_id": deployment_id,
            "session_id": "sess_stub_1",
            "generation_id": "gen_stub_1",
            "checkpoint_digest": "sha256:" + "e" * 64,
            "manifest_hash": "sha256:" + "f" * 64,
            "base_url": _FAKE_BASE_URL,
        }
        self.calls = []
        self.opens = 0
        self.closes = []
        self._prediction = prediction

    def open(self):
        self.opens += 1
        return dict(self.identity)

    def act(self, images, state, instruction=None):
        self.calls.append(
            {"images": dict(images), "state": list(state), "instruction": instruction}
        )
        if self._prediction is not None:
            return self._prediction
        return {
            "actions": [
                [float(row) + column / 100.0 for column in range(STATE_DIM)]
                for row in range(ACTION_HORIZON)
            ],
            "horizon": ACTION_HORIZON,
            "action_space": "joint_position",
            "telemetry": {"server_infer_ms": 42.0},
            "binding": {"deploy_generation": "gen_stub_1"},
            "safety_signal": None,
        }

    def close(self, success=True):
        self.closes.append(success)


@unittest.skipUnless(_HAVE_CLIENT, _CLIENT_SKIP)
class MolmoActServoTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.np = np
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.credentials = _write_credentials(self._dir.name)

    def _policy(self, *, prediction=None, **kwargs):
        kwargs.setdefault("credentials", self.credentials)
        kwargs.setdefault("servo_python", sys.executable)
        kwargs.setdefault("instruction", "pick up the red cap")
        policy = MolmoActServo("dep_test", **kwargs)
        policy._transport = _StubTransport(prediction=prediction)
        return policy

    def _frame(self, height, width, color):
        frame = self.np.zeros((height, width, 3), dtype=self.np.uint8)
        frame[:, :, :] = color
        return frame

    def _observation(self):
        return {
            "front_camera_rgb": self._frame(48, 64, (220, 20, 20)),
            "left_camera_rgb": self._frame(24, 32, (20, 220, 20)),
            "right_camera_rgb": self._frame(16, 16, (20, 20, 220)),
            "joint_positions": self.np.arange(STATE_DIM, dtype=self.np.float32),
        }

    def test_a_deployment_id_is_mandatory(self):
        for missing in (None, ""):
            with self.assertRaisesRegex(ValueError, "deployment id"):
                MolmoActServo(missing, credentials=self.credentials)

    def test_encoder_settings_are_validated(self):
        with self.assertRaisesRegex(ValueError, "jpeg_quality"):
            MolmoActServo("dep_test", credentials=self.credentials, jpeg_quality=0)
        with self.assertRaisesRegex(ValueError, "jpeg_quality"):
            MolmoActServo("dep_test", credentials=self.credentials, jpeg_quality=101)
        with self.assertRaisesRegex(ValueError, "image_size"):
            MolmoActServo(
                "dep_test",
                credentials=self.credentials,
                servo_python=sys.executable,
                image_size=0,
            )

    def test_prepare_input_and_inference_produce_a_30x14_float32_chunk(self):
        policy = self._policy()
        self.assertEqual(policy.get_action_horizon(), ACTION_HORIZON)
        prepared = policy.prepare_input(self._observation(), "pick up the red cap")
        self.assertEqual(prepared["instruction"], "pick up the red cap")
        self.assertEqual(prepared["state"].shape, (STATE_DIM,))

        result = policy.inference(prepared)
        actions = result["actions"]
        self.assertEqual(actions.shape, (ACTION_HORIZON, STATE_DIM))
        self.assertEqual(actions.dtype, self.np.float32)
        self.assertTrue(self.np.isfinite(actions).all())
        self.assertEqual(result["transport"]["protocol"], "servo-action-session")
        self.assertEqual(result["transport"]["session_id"], "sess_stub_1")
        self.assertEqual(result["transport"]["action_space"], "joint_position")
        self.assertEqual(result["transport"]["server_inference_ms"], 42.0)
        self.assertGreater(result["transport"]["image_bytes"], 0)
        self.assertEqual(result["reproducibility"]["backend"], "servo-action-session")
        self.assertEqual(result["servo"]["action_space"], "joint_position")
        self.assertEqual(result["servo"]["telemetry"], {"server_infer_ms": 42.0})

        call = policy._transport.calls[0]
        self.assertEqual(call["instruction"], "pick up the red cap")
        self.assertEqual(call["state"], [float(i) for i in range(STATE_DIM)])
        self.assertEqual(
            result["transport"]["image_bytes"],
            sum(len(blob) for blob in call["images"].values()),
        )

    def test_cameras_map_onto_the_servo_keys(self):
        from PIL import Image

        policy = self._policy()
        observation = self._observation()
        policy.inference(policy.prepare_input(observation, "pick up the red cap"))
        images = policy._transport.calls[0]["images"]

        self.assertEqual(set(images), set(CAMERA_KEYS))
        expected = {
            "top": ((64, 48), 0),  # front camera, red
            "left": ((32, 24), 1),  # left wrist, green
            "right": ((16, 16), 2),  # right wrist, blue
        }
        for key, (size, channel) in expected.items():
            blob = images[key]
            self.assertIsInstance(blob, bytes)
            self.assertTrue(blob.startswith(b"\xff\xd8"), f"{key} is not a JPEG")
            with Image.open(io.BytesIO(blob)) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, size, f"{key} came from the wrong camera")
                pixels = self.np.asarray(image.convert("RGB"))
            self.assertEqual(
                int(pixels.mean(axis=(0, 1)).argmax()),
                channel,
                f"{key} came from the wrong camera",
            )
        _assert_no_base64_images(self, policy._transport.calls[0], images.values())

    def test_jpeg_quality_and_image_size_are_honoured(self):
        from PIL import Image

        rng = self.np.random.default_rng(1234)
        noisy = rng.integers(0, 256, size=(96, 128, 3), dtype=self.np.uint8)
        observation = {
            "front_camera_rgb": noisy,
            "left_camera_rgb": noisy,
            "right_camera_rgb": noisy,
            "joint_positions": self.np.zeros(STATE_DIM, dtype=self.np.float32),
        }

        low = self._policy(jpeg_quality=40)
        high = self._policy(jpeg_quality=95)
        low.inference(low.prepare_input(observation, "x"))
        high.inference(high.prepare_input(observation, "x"))
        low_bytes = len(low._transport.calls[0]["images"]["top"])
        high_bytes = len(high._transport.calls[0]["images"]["top"])
        self.assertLess(low_bytes, high_bytes)

        sized = self._policy(image_size=224)
        sized.inference(sized.prepare_input(observation, "x"))
        for blob in sized._transport.calls[0]["images"].values():
            with Image.open(io.BytesIO(blob)) as image:
                self.assertEqual(image.size, (224, 224))

        self.assertEqual(
            low.reproducibility_metadata()["image_encoding"], "jpeg-quality-40-size-source"
        )
        self.assertEqual(
            sized.reproducibility_metadata()["image_encoding"], "jpeg-quality-85-size-224"
        )

    def test_reproducibility_metadata_reports_the_session_identity(self):
        policy = self._policy()
        metadata = policy.reproducibility_metadata()
        self.assertEqual(metadata["backend"], "servo-action-session")
        self.assertEqual(metadata["deployment_id"], "dep_test")
        self.assertEqual(metadata["session_id"], "sess_stub_1")
        self.assertEqual(metadata["generation_id"], "gen_stub_1")
        self.assertEqual(metadata["checkpoint_digest"], "sha256:" + "e" * 64)
        self.assertEqual(metadata["manifest_hash"], "sha256:" + "f" * 64)
        self.assertEqual(metadata["base_url"], _FAKE_BASE_URL)
        self.assertFalse(metadata["deterministic_generator"])
        self.assertFalse(policy.begin_rollout(7)["deterministic_generator"])

    def test_open_and_close_delegate_to_the_single_session(self):
        policy = self._policy()
        self.assertEqual(policy.open()["session_id"], "sess_stub_1")
        self.assertEqual(policy.identity["session_id"], "sess_stub_1")
        policy.close(success=False)
        policy.close()
        self.assertEqual(policy._transport.opens, 1)
        self.assertEqual(policy._transport.closes, [False, True])

    def test_a_malformed_chunk_from_the_transport_is_refused(self):
        prepared_observation = self._observation()
        broken = [
            {
                "actions": [[0.0] * STATE_DIM for _ in range(29)],
                "action_space": "joint_position",
            },
            {
                "actions": [[0.0] * 7 for _ in range(ACTION_HORIZON)],
                "action_space": "joint_position",
            },
            {
                "actions": [
                    [float("nan")] * STATE_DIM for _ in range(ACTION_HORIZON)
                ],
                "action_space": "joint_position",
            },
        ]
        for prediction in broken:
            policy = self._policy(prediction=prediction)
            prepared = policy.prepare_input(prepared_observation, "x")
            with self.assertRaisesRegex(RuntimeError, "malformed action chunk"):
                policy.inference(prepared)

    def test_a_non_bimanual_state_is_refused_before_the_wire(self):
        policy = self._policy()
        observation = self._observation()
        observation["joint_positions"] = self.np.zeros(7, dtype=self.np.float32)
        with self.assertRaisesRegex(ValueError, "bimanual state"):
            policy.prepare_input(observation, "x")
        self.assertEqual(policy._transport.calls, [])

        prepared = policy.prepare_input(self._observation(), "x")
        prepared["state"] = self.np.full(STATE_DIM, self.np.nan, dtype=self.np.float32)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            policy.inference(prepared)
        self.assertEqual(policy._transport.calls, [])

    def test_a_non_rgb_camera_frame_is_refused(self):
        policy = self._policy()
        observation = self._observation()
        observation["front_camera_rgb"] = self.np.zeros((8, 8), dtype=self.np.uint8)
        with self.assertRaisesRegex(ValueError, "HxWx3"):
            policy.inference(policy.prepare_input(observation, "x"))
        self.assertEqual(policy._transport.calls, [])


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# 6. The codec wire: raw pixels across the bridge, and a cold-failing open
# ---------------------------------------------------------------------------


def _install_fake_servo_direct(test, *, attach_hook=None, metadata=None,
                               metadata_error=None, session_error=None):
    """Inject a fake ``servo.direct`` and record what ``attach`` was given."""

    state = SimpleNamespace(attach_calls=[], policies=[], closed=0, sessions=0)

    class _FakeGrant:
        endpoint_url = "https://fake.endpoint:8443"
        session_route = "quic://fake.endpoint:9443"
        control_profile = {"max_jump": None, "max_relative_target": None,
                           "fps": 30.0, "exec_steps": 30}
        camera_inputs = {}
        identity = {
            "deployment_id": "dep_fake",
            # Matches _fake_prediction's binding: the direct host now fences
            # every response against the grant's generation, so a fixture that
            # disagreed would be testing the fence, not the wire.
            "deploy_generation": "gen_test_1",
            "checkpoint_digest": "sha256:fake",
            "manifest_hash": "sha256:manifest-a",
        }

    class _FakeTransport:
        lease = SimpleNamespace(session_id="actsession_fake_1")

    class _FakePolicy:
        def __init__(self, **kwargs):
            self.grant = _FakeGrant()
            self.camera_inputs = {}
            self.kwargs = kwargs
            self.acts = []

        def metadata(self):
            if metadata_error is not None:
                raise metadata_error
            return dict(metadata or {"manifest_hash": "sha256:manifest-a"})

        def open_action_session_transport(self, *, instruction=None):
            if session_error is not None:
                raise session_error
            state.sessions += 1
            return _FakeTransport()

        def act(self, observation, instruction=None):
            self.acts.append((observation, instruction))
            return _fake_prediction()

        def close(self):
            state.closed += 1

    def _attach(**kwargs):
        state.attach_calls.append(dict(kwargs))
        if attach_hook is not None:
            attach_hook(kwargs)
        policy = _FakePolicy(**kwargs)
        state.policies.append(policy)
        return policy

    package = ModuleType("servo")
    package.__spec__ = importlib.machinery.ModuleSpec("servo", loader=None)
    package.__path__ = []  # type: ignore[attr-defined]
    direct = ModuleType("servo.direct")
    direct.__spec__ = importlib.machinery.ModuleSpec("servo.direct", loader=None)
    direct.attach = _attach
    package.direct = direct

    previous = {name: sys.modules.get(name) for name in ("servo", "servo.direct")}
    sys.modules["servo"] = package
    sys.modules["servo.direct"] = direct

    def _restore():
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:  # pragma: no cover
                sys.modules[name] = module

    test.addCleanup(_restore)
    return state


def _write_grant(directory: str) -> str:
    path = Path(directory) / "grant.txt"
    path.write_text("fake-grant-blob")
    path.chmod(0o600)
    return str(path)


@unittest.skipUnless(_HAVE_NUMPY, "numpy is required for the raw-pixel wire")
class RawFrameFramingTests(unittest.TestCase):
    """The pipe must carry pixels intact -- a wrong shape is undetectable later."""

    def test_a_raw_frame_survives_the_pipe_byte_for_byte(self):
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        buffer = memoryview(np.ascontiguousarray(image)).cast("B")
        header = {"op": "act", "frames": {"top": raw_frame_spec(0, 4, 5)}}
        stream = io.BytesIO(b"".join(bytes(p) for p in frame_parts(header, [buffer])))
        decoded_header, buffers = read_frame(stream)
        restored = raw_frame_array(buffers[0], decoded_header["frames"]["top"])
        self.assertEqual(restored.shape, (4, 5, 3))
        self.assertEqual(restored.dtype, np.uint8)
        self.assertTrue(np.array_equal(restored, image))

    def test_a_memoryview_is_framed_by_its_byte_count_not_its_row_count(self):
        # ``len(memoryview(hxwx3))`` is the ROW count. Framing that would
        # declare a 4-byte buffer for a 60-byte frame and truncate silently.
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        buffer = memoryview(image).cast("B")
        self.assertEqual(len(buffer), image.nbytes)
        parts = frame_parts({"op": "act"}, [buffer])
        self.assertEqual(sum(len(bytes(p)) for p in parts[3:]), image.nbytes)

    def test_a_truncated_raw_frame_is_refused_not_reshaped(self):
        with self.assertRaisesRegex(ServoBridgeError, "expected 60"):
            raw_frame_array(b"\x00" * 59, raw_frame_spec(0, 4, 5))

    def test_a_foreign_dtype_is_refused(self):
        spec = {**raw_frame_spec(0, 4, 5), "dtype": "float32"}
        with self.assertRaisesRegex(ServoBridgeError, "must be uint8"):
            raw_frame_array(b"\x00" * 60, spec)

    def test_missing_cameras_reports_arrays_without_raising(self):
        full = {key: np.zeros((2, 2, 3), dtype=np.uint8) for key in CAMERA_KEYS}
        self.assertEqual(missing_cameras(full), [])
        empty = {**full, "left": np.zeros((0, 2, 3), dtype=np.uint8)}
        self.assertEqual(missing_cameras(empty), ["left"])
        self.assertEqual(missing_cameras({}), list(CAMERA_KEYS))


class ServoDirectHostOpenTests(unittest.TestCase):
    """``open`` must prove the endpoint with the arms COLD.

    ``DirectPolicy`` is lazy -- it touches the network on its first act -- so
    without a forced round trip a stale grant opens clean and fails once the
    motors are live. That is the failure this class exists to prevent.
    """

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.grant = _write_grant(self._dir.name)

    def test_open_forces_the_endpoint_and_session_round_trips(self):
        state = _install_fake_servo_direct(self)
        host = ServoDirectHost(grant=self.grant, instruction="pick up the lid")
        identity = host.open()
        self.assertEqual(state.sessions, 1)
        self.assertEqual(identity["session_id"], "actsession_fake_1")
        self.assertEqual(identity["generation_id"], "gen_test_1")
        self.assertEqual(identity["deployment_id"], "dep_fake")
        self.assertEqual(identity["observation_encoding"], "jpeg")

    def test_a_stale_grant_fails_at_open_not_at_act(self):
        state = _install_fake_servo_direct(
            self, metadata_error=RuntimeError("401 endpoint token kid is not configured")
        )
        host = ServoDirectHost(grant=self.grant)
        with self.assertRaisesRegex(ServoBridgeError, "refused the grant at open"):
            host.open()
        # The failed policy is released rather than left holding a connection.
        self.assertEqual(state.closed, 1)

    def test_a_grant_for_a_previous_serve_fails_at_open(self):
        _install_fake_servo_direct(self, metadata={"manifest_hash": "sha256:manifest-b"})
        host = ServoDirectHost(grant=self.grant)
        with self.assertRaisesRegex(ServoBridgeError, "previous serve"):
            host.open()

    def test_a_refused_session_fails_at_open(self):
        _install_fake_servo_direct(self, session_error=RuntimeError("lease refused"))
        host = ServoDirectHost(grant=self.grant)
        with self.assertRaisesRegex(ServoBridgeError, "refused to open"):
            host.open()

    def test_the_negotiated_encoding_reaches_attach(self):
        state = _install_fake_servo_direct(self)
        host = ServoDirectHost(
            grant=self.grant, observation_encoding="h264", h264_crf=27
        )
        identity = host.open()
        self.assertEqual(state.attach_calls[0]["observation_encoding"], "h264")
        self.assertEqual(state.attach_calls[0]["h264_crf"], 27)
        self.assertEqual(identity["observation_encoding"], "h264")
        self.assertEqual(identity["h264_crf"], 27)

    def test_an_sdk_without_the_codec_wire_is_named_not_silently_downgraded(self):
        def _reject(kwargs):
            if "observation_encoding" in kwargs:
                raise TypeError("attach() got an unexpected keyword argument")

        _install_fake_servo_direct(self, attach_hook=_reject)
        host = ServoDirectHost(grant=self.grant, observation_encoding="h264")
        with self.assertRaisesRegex(ServoBridgeError, "predates the codec wire"):
            host.open()

    def test_an_unknown_encoding_is_refused_up_front(self):
        with self.assertRaisesRegex(ServoBridgeError, "observation_encoding must be"):
            ServoDirectHost(grant=self.grant, observation_encoding="av1")

    def test_a_crf_without_the_codec_wire_is_refused(self):
        with self.assertRaisesRegex(ServoBridgeError, "h264_crf only applies"):
            ServoDirectHost(grant=self.grant, h264_crf=27)

    def test_a_generation_switch_under_a_direct_session_is_refused(self):
        # Populating generation_id at open is what arms the inherited fence
        # for self-hosted endpoints; before that it was silently inert here.
        state = _install_fake_servo_direct(self)
        state_grant = sys.modules["servo.direct"]
        host = ServoDirectHost(grant=self.grant)
        host.open()
        host.identity["generation_id"] = "gen_other"
        with self.assertRaisesRegex(ServoBridgeError, "different generation"):
            host.act({key: b"\xff\xd8jpeg" for key in CAMERA_KEYS}, [0.0] * STATE_DIM)
        del state, state_grant

    @unittest.skipUnless(_HAVE_NUMPY, "numpy is required for the raw-pixel wire")
    def test_raw_pixels_reach_the_sdk_unencoded(self):
        state = _install_fake_servo_direct(self)
        host = ServoDirectHost(grant=self.grant, observation_encoding="h264")
        host.open()
        images = {
            key: np.full((2, 3, 3), index + 1, dtype=np.uint8)
            for index, key in enumerate(CAMERA_KEYS)
        }
        host.act(images, [0.0] * STATE_DIM)
        observation, _ = state.policies[0].acts[0]
        for key in CAMERA_KEYS:
            frame = observation["images"][key]
            self.assertFalse(isinstance(frame, (bytes, bytearray)))
            self.assertTrue(np.array_equal(frame, images[key]))


@unittest.skipUnless(_HAVE_CLIENT, _CLIENT_SKIP)
@unittest.skipUnless(_HAVE_NUMPY, "numpy is required for the raw-pixel wire")
class MolmoActServoWireTests(unittest.TestCase):
    """Which payload the policy builds is decided by the negotiated wire."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.grant = _write_grant(self._dir.name)

    def _policy(self, **kwargs):
        return MolmoActServo(
            grant=self.grant,
            servo_python=sys.executable,
            instruction="pick up the red lid",
            **kwargs,
        )

    def _observation(self):
        return {
            "left_camera_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "front_camera_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "right_camera_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "joint_positions": np.zeros(STATE_DIM, dtype=np.float32),
        }

    def _sent_images(self, policy):
        captured = {}

        def _act(images, state, instruction=None):
            captured["images"] = images
            return _fake_prediction().__dict__ if hasattr(_fake_prediction(), "__dict__") else {}

        return captured, _act

    def test_the_jpeg_wire_still_mints_jpeg_bytes(self):
        policy = self._policy()
        prepared = policy.prepare_input(self._observation(), "pick up the red lid")
        captured = {}
        with unittest.mock.patch.object(
            policy._transport, "act",
            side_effect=lambda images, state, instruction=None: (
                captured.update(images=images) or _fake_prediction_dict()
            ),
        ):
            policy.inference(prepared)
        for key in CAMERA_KEYS:
            self.assertIsInstance(captured["images"][key], bytes)
            self.assertTrue(captured["images"][key].startswith(b"\xff\xd8"))

    def test_the_codec_wire_sends_raw_contiguous_uint8_pixels(self):
        policy = self._policy(observation_encoding="h264")
        prepared = policy.prepare_input(self._observation(), "pick up the red lid")
        captured = {}
        with unittest.mock.patch.object(
            policy._transport, "act",
            side_effect=lambda images, state, instruction=None: (
                captured.update(images=images) or _fake_prediction_dict()
            ),
        ):
            policy.inference(prepared)
        for key in CAMERA_KEYS:
            frame = captured["images"][key]
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(frame.dtype, np.uint8)
            self.assertEqual(frame.shape, (8, 8, 3))
            self.assertTrue(frame.flags["C_CONTIGUOUS"])

    def test_the_codec_wire_needs_a_self_hosted_grant(self):
        with self.assertRaisesRegex(ServoBridgeError, "self-hosted|grant="):
            MolmoActServo(
                deployment="dep_hosted",
                credentials=_write_credentials(self._dir.name),
                servo_python=sys.executable,
                observation_encoding="h264",
            )

    def test_every_declared_encoding_is_constructible(self):
        for encoding in OBSERVATION_ENCODINGS:
            self._policy(observation_encoding=encoding)


def _fake_prediction_dict():
    """The bridge-shaped prediction mapping ``MolmoActServo.inference`` consumes."""
    return {
        "actions": [[0.0] * STATE_DIM for _ in range(ACTION_HORIZON)],
        "action_space": "joint_position",
        "telemetry": {"server_infer_ms": 12.0, "request_bytes": 4096},
    }
