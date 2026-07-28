# Bimanual YAM — MolmoAct2 closed-loop eval

This directory holds the **robot/client side** of MolmoAct2 on a bimanual YAM
setup. The inference **server** (`host_server_yam.py`) loads the
[`allenai/MolmoAct2-BimanualYAM`](https://huggingface.co/allenai/MolmoAct2-BimanualYAM)
checkpoint and serves actions over HTTP; the eval launcher here drives the two
YAM arms, captures the 3-camera observation, queries a policy, executes the
returned action chunk, records each rollout, and (optionally) converts a
labeled session into a LeRobot v3.0 dataset.

It is vendored and trimmed from the reference YAM implementation at
<https://github.com/williamtsai726/YAM> — only the eval-relevant pieces are
kept (teleop, data collection, and the Gello leader-arm code are omitted).

> Hardware-coupled example: it talks to real YAM arms over CAN (via `i2rt`) and
> Intel RealSense cameras. It is meant to run on the workstation wired to the
> robot, not in the dependency-light server environment.

## Layout

```
examples/yam/
├── host_server_yam.py            # inference server (separate; see top-level README §5)
├── launch_yaml_eval_molmoact.py  # eval launcher — main entry point
├── molmoact_client.py            # MolmoAct (HTTP) + MolmoActLocal (in-process) policies
├── camera_server.py              # long-lived ZMQ server owning the 3 RealSense cams
├── camera_client.py              # ZMQ client + standalone live viewer
├── eval_utils.py                 # per-rollout saver, cv2 viewer, labeling, conversion
├── rerun_rollout.py               # saved rollout -> offline Rerun .rrd playback
├── rerun_export_watchdog.py       # detached crash-resilient .rrd exporter
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

Run every command below **from the molmoact2 repo root** (the scripts add
`examples/yam/` to `sys.path`, so `gello_min` and the sibling modules resolve).

## Inference modes

Set `eval.mode` in `configs/yam_left.yaml`:

- **`server`** (default) — POST observations to a running `host_server_yam.py`
  using the `json_numpy` wire protocol. Point `eval.molmoact_server` at the
  server (`host:port`, full URL, or ngrok hostname; `/act` is appended).
  Start the server per the top-level README §5, e.g.
  `uv run python examples/yam/host_server_yam.py --port 8202`.
- **`local`** — load the checkpoint in-process via `transformers` (no server).
  Configure under `eval.local`. bf16 needs ~10–14 GB VRAM, fp32 ~26 GB.

The two policies are interchangeable; the launcher picks one from the config.

## Hardware setup

1. Both YAM arms powered, e-stop released.
2. The 3 RealSense cameras plugged into USB 3; put their serials in
   `configs/yam_left.yaml` under `sensors.cameras`. **Order matters** — the
   model was trained on `[top, left, right]`; here `front_camera` plays the
   `top` role.
3. Bring up CAN and set the camera/CAN interface names in the configs
   (`channel:` — find them with `ip link show`). Disable the motor watchdog so
   the arms don't collapse during long sessions (see the `i2rt` docs / your
   YAM bring-up scripts).

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

Or use Rerun's browser viewer and open the URL it prints (normally
`http://127.0.0.1:9090`):

```bash
/home/npow/molmoact2-venv/bin/rerun --web-viewer --web-viewer-port 9090 \
  yam_eval_runs/data/red_lid_left_arm/eval/<timestamp>/rollout.rrd
```

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
| `eval.mode` | `server` (HTTP) or `local` (in-process). |
| `eval.molmoact_server` | Server address for `mode: server`. |
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
