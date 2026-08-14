# Bimanual YAM — MolmoAct2 closed-loop eval

This directory holds the **robot/client side** of MolmoAct2 on a bimanual YAM
setup. Inference itself runs either on Servo (a managed hosted deployment of
the [`allenai/MolmoAct2-BimanualYAM`](https://huggingface.co/allenai/MolmoAct2-BimanualYAM)
checkpoint, reached over one official Servo action session), in a self-hosted
`host_server_yam.py` process on this LAN, or in-process from the checkpoint.
The eval launcher here drives the two YAM arms, captures the 3-camera
observation, queries a policy, executes the returned action chunk, records each
rollout, and (optionally) converts a labeled session into a LeRobot v3.0
dataset.

It is vendored and trimmed from the reference YAM implementation at
<https://github.com/williamtsai726/YAM> — only the eval-relevant pieces are
kept (teleop, data collection, and the Gello leader-arm code are omitted).

> Hardware-coupled example: it talks to real YAM arms over CAN (via `i2rt`) and
> Intel RealSense cameras. It is meant to run on the workstation wired to the
> robot, not in the dependency-light server environment.

## Layout

```
examples/yam/
├── host_server_yam.py            # self-hosted inference server (separate; see top-level README §5)
├── host_servo_yam.py             # superseded self-hosted Servo endpoint (no client speaks it)
├── launch_yaml_eval_molmoact.py  # eval launcher — main entry point
├── molmoact_client.py            # MolmoActServo (Servo session) + MolmoActHTTP + MolmoActLocal policies
├── servo_session_bridge.py       # official Servo SDK session host + Python 3.11 -> 3.12 bridge
├── camera_server.py              # long-lived ZMQ server owning the 3 RealSense cams
├── camera_client.py              # ZMQ client + standalone live viewer
├── eval_utils.py                 # per-rollout saver, cv2 viewer, labeling, conversion
├── rerun_rollout.py               # saved rollout -> offline Rerun .rrd playback
├── rerun_export_watchdog.py       # detached crash-resilient .rrd exporter
├── view_rollout.sh                # open an .rrd on a network-accessible web viewer
├── lerobot_convert.py            # raw rollouts -> LeRobot v3.0 dataset
├── start_camera_server.sh        # convenience launcher for camera_server.py
├── requirements.txt
├── configs/
│   ├── yam_left.yaml             # cameras, storage, eval, lerobot + left arm
│   └── yam_right.yaml            # right arm only
└── gello_min/                    # trimmed YAM runtime (robot/env/camera drivers)
```

## Install

```bash
pip install -r examples/yam/requirements.txt
# Plus the two non-PyPI deps (see requirements.txt):
#   i2rt    — YAM CAN/motor driver (required)
#   lerobot — only for the optional dataset conversion
```

`server` mode additionally needs the official `servo-client` SDK, which
requires Python >= 3.12 and therefore does **not** go into the 3.11 robot
runtime — see [Servo action sessions](#servo-action-sessions) below.

Run every command below **from the molmoact2 repo root** (the scripts add
`examples/yam/` to `sys.path`, so `gello_min` and the sibling modules resolve).

## Inference modes

Set `eval.mode` in the rollout YAML, or override it per run with
`--policy-mode`:

- **`server`** — hosted policy over **one official Servo action session**. The
  SDK resolves a managed Servo deployment, opens a single action session
  (signed offer + lease) for the whole run, and every action chunk rides that
  session's protobuf transport. Requires a managed deployment id
  (`--servo-deployment dep_...` or `eval.server.deployment`). This is the
  launcher's default when `eval.mode` is absent from the config.
- **`http`** — POST observations to a running `host_server_yam.py` on this LAN
  using the legacy `json_numpy` protocol. Point `eval.molmoact_server` (or
  `--molmoact-server`) at it (`host:port` or full URL; `/act` is appended).
  Start the server per the top-level README §5, e.g.
  `uv run python examples/yam/host_server_yam.py --port 8202`. Development path
  only: no authentication, no generation fencing, no control-plane record.
- **`local`** — load the checkpoint in-process via `transformers` (no server).
  Configure under `eval.local`. bf16 needs ~10–14 GB VRAM, fp32 ~26 GB.

The three policies are interchangeable behind the same
`prepare_input` -> `inference` interface; the launcher picks one from the
config/CLI. `configs/yam_left.yaml` ships `mode: http`; the physical-arm
configs ship `mode: local`. Pass `--policy-mode server` to take the hosted path
without editing a rollout YAML.

### Servo action sessions

`server` mode is pure official Servo SDK: `Servo(base_url, api_key)` ->
`deployments.get(...)` -> `deployment.policy(...)` -> `sv.session(policy)`.
There is no custom endpoint, no bespoke wire format, and no client-side token
signing — the control plane owns authentication, the signed offer/lease, and
generation fencing.

**Credentials.** The client reads a machine (SDK) API key from
`~/.config/servo/molmoact2-yam-sdk.json`, mode `0600`, schema
`servo.sdk-credentials.v1` (`api_key`, `base_url`, `key_id`, `label`). Never
commit it, never print its `api_key`, and keep the file owner-only — the client
refuses a group/world-readable bundle and refuses the browser/CLI session token
in `~/.config/servo/credentials.json`, which is not a machine API key.
`SERVO_UNKEY_ROOT_KEY` (exported from `~/.bashrc`) is an Unkey
provider/operator secret; it is **not** a robot client credential and must not
be used here.

**Interpreter split.** The robot runtime (`/home/npow/molmoact2-venv`) is
Python 3.11 because of i2rt/pyrealsense2/torch; the official SDK requires
Python >= 3.12. `servo_session_bridge.py` resolves that locally: it holds the
SDK session code (`ServoSessionHost`) and can also run as a small stdio helper
process under a 3.12 interpreter. If `servo` is importable in the running
interpreter, the session runs in-process; otherwise you must name the
interpreter explicitly with `--servo-python`, `eval.server.servo_python`, or
the `SERVO_PYTHON` environment variable — on this workstation
`/home/npow/code/servo/.venv/bin/python`. An explicitly named interpreter is
always used, even when `servo` is importable here. There is no silent
fallback: if none of the three is set and `servo` is not importable, `server`
mode fails immediately instead of choosing for you. The parent/child hop is a
local pipe only; everything that leaves the machine is SDK transport.

**One session per run.** The launcher opens the session right after policy
construction and *before* any motor can be enabled, so a bad credential,
deployment, or lease fails with the arms still cold. That same session serves
every action chunk of every rollout in the session, and is completed on exit —
`_shutdown_runtime()` releases the robot first and closes the policy second, so
the session's closing network round trip never delays motor release.

**What rides the wire.** Each request carries the three camera frames as raw
JPEG bytes through the SDK's `EncodedObservation` (Servo accepts JPEG or PNG;
there is **no base64 step**) plus a 14-float state vector
`[left arm (7), right arm (7)]`. Camera roles, where "left"/"right" mean
standing behind the arms and facing into the workspace:

| Device alias | Servo camera key | View |
| --- | --- | --- |
| `/dev/yam-cameras/middle` | `top` | middle/static workspace camera |
| `/dev/yam-cameras/left` | `left` | left wrist |
| `/dev/yam-cameras/right` | `right` | right wrist |

The instruction you type at the stdin prompt rides with every action request.
Responses are validated client-side: a 30x14 chunk, action space
`joint_position`, decoded, finite, and served by the same binding generation the
session opened against.

Measured three-camera payload from this rig:

| Encoding | Bytes for all three frames | Encode time | Notes |
| --- | ---: | ---: | --- |
| JPEG quality 85 (default) | ~87.3 KiB | ~14 ms | `eval.server.jpeg_quality: 85` |
| JPEG quality 75 | ~64.8 KiB | ~14 ms | smaller, lower visual quality |
| WebP quality 85 | ~51.4 KiB | ~182 ms | **not accepted** by Servo action sessions |

WebP is smaller but costs an order of magnitude more CPU per query and Servo
action sessions take JPEG or PNG only — stay on JPEG and tune
`eval.server.jpeg_quality` if you need fewer bytes. `eval.server.image_size`
stays `null` by default: frames go at source resolution because the Servo
runtime owns model-specific resize/padding; it is a diagnostic override only.

**Launch a hosted run:**

```bash
PYTHONPATH=examples/yam:/home/npow/code/i2rt \
/home/npow/molmoact2-venv/bin/python \
examples/yam/launch_yaml_eval_molmoact.py \
  --config-path examples/yam/configs/yam_left_physical.yaml \
  --right-config-path examples/yam/configs/yam_right_primary.yaml \
  --policy-mode server \
  --servo-deployment dep_xxxxxxxxxxxx \
  --servo-credentials ~/.config/servo/molmoact2-yam-sdk.json \
  --servo-python /home/npow/code/servo/.venv/bin/python \
  --active-arm-side left \
  --execution-mode active_arm_hold \
  --num-rollouts 1
```

`--servo-python` can be dropped if `SERVO_PYTHON` is exported or the launcher
already runs on a 3.12 interpreter that has `servo-client`.
`--servo-credentials` can be dropped to accept the default path above.

**State of play (2026-08-02).** The client side is complete, but a live run
also needs the *remote* Servo control plane to expose a managed MolmoAct2
deployment for the YAM embodiment with action sessions enabled on its runtime
artifact. That deployment does not exist yet: the catalog reachable with this
machine key still advertises the single-arm/SO-101 view, and runtime
`action_sessions_enabled` defaults to false. Until it is created, `server` mode
fails at session open (before any motor is enabled, by design). Use `http` or
`local` mode in the meantime.

### Superseded: the self-hosted Servo endpoint (`host_servo_yam.py`)

`host_servo_yam.py` and its Odin systemd/observability units are **superseded**
and no client in this tree speaks their protocol any more. They served a custom
authenticated `/act` endpoint with a locally signed endpoint token, replaced by
the official action-session path above. The client-side endpoint code and its
token-signing dependency are gone from `molmoact_client.py` and
`requirements.txt`; the endpoint credential/verifier bundles still sitting in
`~/.config/servo/` are no longer read by anything here — the only Servo
credential this client uses is `molmoact2-yam-sdk.json`.

The file is still on disk and its long-lived user unit may still be running on
Odin. If you find that unit serving traffic, it has no client here — inspect or
retire it with:

```bash
ssh odin
systemctl --user status molmoact2-yam-servo.service
journalctl --user -u molmoact2-yam-servo.service -n 50
# stop it once nothing depends on it:
systemctl --user disable --now molmoact2-yam-servo.service
```

Its Prometheus/Grafana sidecar (root unit
`molmoact2-yam-observability.service`, tailnet-only Grafana at
`https://odin.tail17f7a4.ts.net/grafana/`, credentials in the mode-`0600` file
`~/.config/servo/molmoact2-grafana.env`, which must not be committed) only ever
scraped that endpoint and has no consumer in the current path.

## Hardware setup

1. Both YAM arms powered, e-stop released.
2. The 3 RealSense cameras plugged into USB 3, listed in the rollout config
   under `sensors.cameras`. **Order matters** — the model was trained on
   `[top, left, right]`; here `front_camera` plays the `top` role. The
   physical-arm configs address the cameras through the stable
   `/dev/yam-cameras/{middle,left,right}` aliases (udev rule in
   `examples/yam/udev/`, installed to `/etc/udev/rules.d/`) rather than
   `/dev/video*`; replug or hub changes are then harmless, but replacing a
   camera means updating that rule's serial.
3. Bring up CAN and set the camera/CAN interface names in the configs
   (`channel:` — find them with `ip link show`). Disable the motor watchdog so
   the arms don't collapse during long sessions (see the `i2rt` docs / your
   YAM bring-up scripts).
4. Some right-arm motors (4 and the gripper) boot with a latched DaMiao
   status `0x3` ("output-shaft calibration") even though their encoders are
   fine (verified 2026-08-13: positions stable through clear/enable,
   registers identical to healthy motors). The local `i2rt` driver now clears
   this automatically at startup — at most once per motor, requiring a clean
   enable reply and a position that is continuous across the clear, so a
   genuinely lost calibration still fails closed. If an arm refuses to start
   with a calibration error despite this, the fault is real: inspect the
   joint before retrying. `~/code/i2rt/scripts/yam_motor_preflight.py` checks
   all motors without starting a rollout.

## Run a session

Two terminals when the camera server is enabled (the default).

**Terminal A — camera server (long-lived):**

```bash
bash examples/yam/start_camera_server.sh
# or: python examples/yam/camera_server.py --config examples/yam/configs/yam_left.yaml
```

Wait for `REP bound on tcp://127.0.0.1:5555` / `PUB bound on tcp://127.0.0.1:5556`.
It holds the cameras warm across sessions and feeds the live viewer's PUB
stream so the cv2 window keeps repainting during inference.

**Terminal B — eval:**

```bash
python examples/yam/launch_yaml_eval_molmoact.py \
    --config_path       examples/yam/configs/yam_left.yaml \
    --right-config-path examples/yam/configs/yam_right.yaml \
    -n 10
```

`-n 10` runs 10 rollouts. Set `eval.camera_server.enabled: false` to open the
cameras in-process instead (one fewer terminal, but the viewer freezes during
inference).

## What happens per rollout

1. Arms interpolate to `agent.start_joints` — your cue to reset the workspace.
2. Stdin prompts for the task instruction (Enter reuses the previous one).
3. The rollout runs; a 3-pane cv2 window (`YAM Eval`) shows LEFT / FRONT / RIGHT.
4. End it by pressing a key **in the cv2 window**:
   - `y` → success, `n` → failure, `q` → quit (kept unlabeled under `eval/`)
   - or let it hit `max_steps` → you're prompted on stdin afterwards.

`Ctrl-C` is handled: the in-progress rollout is flushed with an `err.md`
marker and any rollouts already labeled this session are still converted.

## Where files land

Under `{storage.base_dir}/data/{storage.task_directory}/`:

```
eval/<ts>/                       # quit / unlabeled rollouts
success/<YYYY-MM-DD>/<ts>/
failure/<YYYY-MM-DD>/<ts>/
eval_lerobot_v30/<session_ts>/   # LeRobot v3.0 dataset (labeled rollouts, end of session)
```

Each rollout has `episode.h5` (joint trajectory + instruction) and one PNG per
camera per frame under `left_rgb/`, `front_rgb/`, `right_rgb/`.

## Post-rollout playback in Rerun

Rerun is an **offline** diagnostic artifact, not a live control dependency.
With `eval.rerun.enabled: true` (enabled in both physical-arm configs), the
launcher starts and verifies a detached exporter *before* it creates any robot
or camera resources. After the launcher exits—normally or due to an
exception—the exporter scans raw rollout directories and atomically publishes
`rollout.rrd` alongside each one. It never opens a camera, CAN interface, or
robot, so export cannot affect motion timing.

Normally a replay includes camera frames, encoder feedback, policy targets,
and action chunks from `episode.h5`. If the launcher dies before HDF5 was
durably written, the exporter still creates an explicitly marked camera-only
recovery RRD from synchronized PNGs; it does not invent lost telemetry. Its
per-rollout state is recorded in `rerun_export.status.json`.

Open the recording after the rollout finishes:

```bash
/home/npow/molmoact2-venv/bin/rerun \
  yam_eval_runs/data/red_lid_left_arm/eval/<timestamp>/rollout.rrd
```

Or use the helper to bind the browser viewer and recording server to every
network interface. It prints a complete LAN URL that opens the saved recording
directly (opening the bare port 9090 URL only shows Rerun's file picker):

```bash
examples/yam/view_rollout.sh \
  yam_eval_runs/data/red_lid_left_arm/eval/<timestamp>/rollout.rrd
```

The helper uses ports 9090 (viewer) and 9876 (recording) by default. Override
them with `RERUN_WEB_PORT` and `RERUN_GRPC_PORT`, or override the executable
with `RERUN_BIN`. Set `RERUN_PUBLIC_IP` when the browser reaches the machine
through a different hostname or address, such as Tailscale.

The recording starts with a three-camera timeline and includes:

- the instruction;
- encoder feedback before and after every command;
- the exact absolute-joint target sent to the arm and target-minus-feedback;
- the full 30-action plan at each policy replan, plus inference timing.

For a historical rollout made before Rerun was enabled, convert it after the
fact (it will show camera and encoder feedback; policy targets are shown only
when that older `episode.h5` recorded them):

```bash
PYTHONPATH=examples/yam /home/npow/molmoact2-venv/bin/python \
  examples/yam/rerun_rollout.py \
  yam_eval_runs/data/red_lid_left_arm/eval/<timestamp>
```

## Key config knobs (`configs/yam_left.yaml`)

| Key | Meaning |
|---|---|
| `eval.mode` | `server` (one Servo action session), `http` (self-hosted `host_server_yam.py`), or `local` (in-process). Defaults to `server` when unset. CLI: `--policy-mode`. |
| `eval.server.deployment` | Managed Servo deployment id (`dep_...`) for `mode: server`; required. CLI: `--servo-deployment`. |
| `eval.server.credentials` | Servo SDK machine-key file (mode `0600`); default `~/.config/servo/molmoact2-yam-sdk.json`. CLI: `--servo-credentials`. |
| `eval.server.servo_python` | Python >= 3.12 interpreter with `servo-client`, used when this runtime cannot import `servo`. CLI: `--servo-python`; env `SERVO_PYTHON`. |
| `eval.server.jpeg_quality` | Camera JPEG quality for the action session; default `85` (~87 KiB for three frames; `75` gives ~65 KiB). |
| `eval.server.image_size` | Optional client-side padded resize; default `null` so the Servo runtime owns preprocessing. |
| `eval.molmoact_server` | `http` mode only: address of the self-hosted `host_server_yam.py`. CLI: `--molmoact-server`. |
| `eval.local.*` | Checkpoint / device / dtype for `mode: local`. |
| `eval.camera_server.enabled` | `true` uses the ZMQ camera server; `false` opens cameras in-process. |
| `eval.live_view_enabled` | `false` disables the cv2 window (headless runs). |
| `eval.rerun.*` | Post-rollout Rerun export (`image_stride`, JPEG quality, policy chunk size). |
| `max_steps` | Per-rollout timeout in control steps. |
| `storage.*` | Output location, instruction, PNG save settings. |
| `lerobot.*` | End-of-session dataset conversion knobs. |

## Camera server, standalone

Sanity-check the cameras independently of the eval loop:

```bash
python examples/yam/camera_client.py --mode sub      # subscribe to the PUB stream
```
