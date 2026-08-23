<div align="center">
  <img src="assets/MolmoAct2.svg" alt="MolmoAct2 Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <br>
  <h1>MolmoAct2: Action Reasoning Models for Real-world Deployment</h1>
</div>

<p align="center">
  <a href="https://github.com/allenai/molmoact2/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/molmoact2">
  </a>
  <a href="https://allenai.org/blog/molmoact2">
    <img alt="Blog Post" src="https://img.shields.io/badge/Blog-Post-F0529C">
  </a>
  <a href="https://arxiv.org/abs/2605.02881">
    <img alt="Paper URL" src="https://img.shields.io/badge/arXiv-2605.02881-red?logo=arxiv">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmoact2-models-69f81e05242e2499606b1be6">
    <img alt="Base Models" src="https://img.shields.io/badge/HF-Base%20Models-yellow?logo=huggingface">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmoact2-finetuned-models-69f81e23d5a7b34fde34f2ce">
    <img alt="Finetuned Models" src="https://img.shields.io/badge/HF-Finetuned%20Models-yellow?logo=huggingface">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmoact2-bimanualyam-dataset-69f81e17b140ec34f430a35e">
    <img alt="MolmoAct2-BimanualYAM Dataset" src="https://img.shields.io/badge/HF-MolmoAct2--BimanualYAM%20Dataset-yellow?logo=huggingface">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmoact2-datasets-69f81e316ec3daafe3f9555c">
    <img alt="Robotics Datasets" src="https://img.shields.io/badge/HF-Robotics%20Datasets-yellow?logo=huggingface">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmo2-er-datasets-69f8d605d92d46a5fc24ced2">
    <img alt="ER Datasets" src="https://img.shields.io/badge/HF-ER%20Datasets-yellow?logo=huggingface">
  </a>
  <a href="https://molmospaces.allen.ai/leaderboard">
    <img alt="1st VLA on MolmoSpace" src="https://img.shields.io/badge/MolmoSpace-1st%20VLA-success?logo=trophy&logoColor=gold">
  </a>
</p>

MolmoAct2 is Ai2's open family of action reasoning models for robot control and real-world deployment. It builds on the Molmo2-ER embodied-reasoning vision-language backbone, adds robot state and action modeling, and connects the VLM to a flow-matching continuous action expert for closed-loop manipulation. The release includes base checkpoints for continued training, fine-tuned robot policies for evaluation and deployment, and the datasets used to build MolmoAct2 and Molmo2-ER.

---
### Updates
- **[2026/08/22]** 🔥 We fixed some finetuning issues in LeRobot's MolmoAct2 training scripts: [**huggingface/lerobot**](https://github.com/huggingface/lerobot).
- **[2026/06/13]** 🔥 We have released pre-training and post-training code and full experimental details for MolmoAct2, get started [**Here**](https://github.com/allenai/molmoact2/tree/main/experiments).
- **[2026/06/10]** 🔥 We have setup zero-shot evaluation for MolmoAct2 (DROID and Bimanual YAM) on Maniskill simulation, get started [**Here**](https://github.com/allenai/molmoact2/tree/main/sim_eval).
- **[2026/05/28]** 🔥 MolmoAct2 has been fully integrated into Huggingface, LeRobot official repo at [**MolmoAct2**](https://huggingface.co/docs/lerobot/main/en/molmoact2).
- **[2026/05/19]** 🔥 We've also released MolmoAct2-Cortex evaluation rollouts on YAM bimanual setups (useful for failure annotation and reward model training) at [**Policy Rollouts**](https://huggingface.co/collections/allenai/molmoact2-eval-rollouts).
- **[2026/05/17]** 🔥 We have released FastAPI inference servers for MolmoAct2 using DROID and YAM setups at [**Inference Servers**](#5-inference-servers) (implemented by [Jie Wang](https://github.com/Everloom-129)).
- **[2026/05/14]** 🔥 We have released MolmoAct2 lerobot workflow for fine-tuning and inference. [**Check it out**](https://github.com/allenai/lerobot/tree/molmoact2-policy). 
- **[2025/05/06]** 🔥 Detail implementation and setup for Franka, SO-100/101, and bimanual YAM have been released at  [**Real-world Deployment**](#4-real-world-deployment).
- **[2026/05/05] 🔥 [MolmoAct2]([https://huggingface.co/collections/allenai/molmoact-689697591a3936fba38174d7](https://allenai.org/blog/molmoact2))** has been released!


## Intel XPU Support

MolmoAct2 inference has been validated on Intel XPU (Intel GPUs) and runs **without any code changes** to this repository. Install a PyTorch build with Intel XPU support and select the `xpu` device at runtime.

## 1. Models

### Base Models

We provide base checkpoints at every training stage for continued MolmoAct2 training and robot fine-tuning. These are foundation checkpoints rather than one-size-fits-all deployment policies.

| Model | Use Case | Description | Checkpoint Path |
| --- | --- | --- | --- |
| MolmoAct2 | Fine-tuning | Post-trained MolmoAct2 model with a continuous flow-matching action expert. Use as the default foundation checkpoint for adapting to a target robot embodiment or benchmark. | https://huggingface.co/allenai/MolmoAct2 |
| MolmoAct2-Think | Fine-tuning | MolmoAct2 foundation checkpoint with depth-token reasoning. Use when downstream policies should reason over compact depth predictions before acting. | https://huggingface.co/allenai/MolmoAct2-Think |
| MolmoAct2-Pretrain | Post-training | Pre-trained discrete autoregressive VLA backbone before the continuous action expert is attached. Intended for continuing MolmoAct2 training stages, not direct continuous-control inference. | https://huggingface.co/allenai/MolmoAct2-Pretrain |
| Molmo2-ER | Pre-training | Embodied-reasoning VLM backbone used as the starting point for MolmoAct2 action models. | https://huggingface.co/allenai/Molmo2-ER |

### Finetuned Models

We also provide fine-tuned checkpoints for common robot platforms and benchmarks. These models are intended to run directly in their target setting, or to serve as a stronger starting point for closely related robots. As with any robot policy, performance depends on hardware, cameras, calibration, action conventions, and language/task distribution.

| Model | Use Case | Description | Checkpoint Path |
| --- | --- | --- | --- |
| MolmoAct2-DROID | Inference / Fine-tuning | MolmoAct2 fine-tuned on the filtered DROID Franka mixture with absolute joint-pose control. Intended for DROID-style policy inference or further fine-tuning. | https://huggingface.co/allenai/MolmoAct2-DROID |
| MolmoAct2-BimanualYAM | Inference / Fine-tuning | MolmoAct2 fine-tuned on the bimanual YAM mixture with absolute joint-pose control and annotated language instructions. | https://huggingface.co/allenai/MolmoAct2-BimanualYAM |
| MolmoAct2-SO100_101 | Inference / Fine-tuning | MolmoAct2 fine-tuned on SO-100/SO-101 datasets with absolute joint-pose control and annotated language instructions. | https://huggingface.co/allenai/MolmoAct2-SO100_101 |
| MolmoAct2-LIBERO | Inference / Fine-tuning | MolmoAct2 fine-tuned on the full LIBERO training mixture, combining Spatial, Object, Goal, and Long suites. | https://huggingface.co/allenai/MolmoAct2-LIBERO |
| MolmoAct2-Think-LIBERO | Inference / Fine-tuning | MolmoAct2-Think fine-tuned on LIBERO with depth-and-action examples and adaptive depth reasoning. | https://huggingface.co/allenai/MolmoAct2-Think-LIBERO |

## 2. Datasets

| Data | Description | Dataset Path |
| --- | --- | --- |
| MolmoAct2-BimanualYAM Dataset | Collection of bimanual YAM datasets and related resources used for MolmoAct2 bimanual training and evaluation. | https://huggingface.co/collections/allenai/molmoact2-bimanualyam-dataset-69f81e17b140ec34f430a35e |
| MolmoAct2 Robotics Datasets | Robotics datasets for MolmoAct2 training and fine-tuning, including SO-100/SO-101, DROID, MolmoAct Dataset, BC-Z, Bridge, and RT-1. | https://huggingface.co/collections/allenai/molmoact2-datasets-69f81e316ec3daafe3f9555c |
| Molmo2-ER Datasets | Embodied reasoning datasets used for Molmo2-ER and MolmoAct2 backbone training, including spatial, 3D, robotics, and visual reasoning data. | https://huggingface.co/collections/allenai/molmo2-er-datasets-69f8d605d92d46a5fc24ced2 |

Note that all of the robotics datasets for pre-training and post-training are in LeRobot v3.0 format, paired with extra language annotations.

## 3. LeRobot Integration

MolmoAct2 is integrated into LeRobot as a policy implementation, so users can train, evaluate, and deploy MolmoAct2 with standard LeRobot datasets and workflows. This repository includes the LeRobot integration as a Git submodule at `lerobot/`, pinned to the branch [`allenai/lerobot:molmoact2-policy`](https://github.com/allenai/lerobot/tree/molmoact2-policy).

For training, although all of our experiments start from the base checkpoint [`allenai/MolmoAct2`](https://huggingface.co/allenai/MolmoAct2), we recommend starting from the fine-tuned checkpoints listed in the [Finetuned Models](#finetuned-models) section above if your embodiment is similar to [Bimanual YAM](https://huggingface.co/allenai/MolmoAct2-BimanualYAM), [DROID Franka](https://huggingface.co/allenai/MolmoAct2-DROID), or [SO-100/SO-101](https://huggingface.co/allenai/MolmoAct2-SO100_101), as they can provide better initialization and downstream performance. For generic use, use the base checkpoint.

After cloning this repository, initialize the submodule from the repo root:

```bash
git submodule update --init --recursive
cd lerobot
```

For training, evaluation, and deployment instructions, see the MolmoAct2 LeRobot documentation at [`docs/source/molmoact2.mdx`](https://github.com/allenai/lerobot/blob/molmoact2-policy/docs/source/molmoact2.mdx). To reproduce the original LIBERO benchmark results exactly with the v0.5.1 evaluation stack, use the pinned inference branch [`allenai/lerobot:molmoact2-hf-inference`](https://github.com/allenai/lerobot/tree/molmoact2-hf-inference) with instructions in [MolmoAct2 README](https://github.com/allenai/lerobot/tree/molmoact2-hf-inference#molmoact2).

We also open-source the original MolmoAct2 experiment scripts under [`experiments/`](experiments/). These cover training and evaluation replication, depth annotation, Hugging Face checkpoint conversion, and fine-tuning on new LeRobot datasets. See [`experiments/README.md`](experiments/README.md) for setup and commands.

## 4. Real-world Deployment

> [!WARNING]
> **Disclaimer:** Out-of-the-box deployment is intended for simple tasks within the training task distribution (e.g., Pick-and-Place, opening, closing and etc). Performance has only been empirically verified on the **SO-100** and **Franka DROID** embodiments. Results on other embodiments and tasks are not guaranteed.

MolmoAct2 supports out-of-the-box deployment on three robot embodiments:

- **SO-100**
- **Bimanual YAMs**
- **Franka DROID setup**

### SO-100/101 Setup

For the best performance, we recommend using an **SO-100 with the standard wrist configuration** and a **third-person camera**. Here is an open implementation by Irene Grace. [Code](https://github.com/irenegracekp/molmoact2-so101)

### Bimanual YAM Setup

For the best performance, please build your Bimanual YAM setup following the reference design below:

![Bimanual YAM setup](assets/m.png)

All required components can be purchased using this [Bimanual YAM parts list](https://docs.google.com/spreadsheets/d/10bg4XJoeIqnuOBLpUlkhJV6QEYn_oK5IZVm5C7_kdbo/edit?usp=sharing).

Implementation code for setting up, data collection, and inference for Bimanual YAM is [here](https://github.com/williamtsai726/YAM)

Standardize evaluation implementation for zero-shot by Cortex AI [here](https://gist.github.com/SuveenE/6bc2b822ac44807565729c2b0ebb1cb2)

### Franka Setup

For the Franka setup, we recommend following the official [DROID implementation](https://github.com/droid-dataset/droid) for best results.

## 5. Inference Servers

This repository ships two FastAPI inference servers under `examples/`, one per fine-tuned checkpoint. Each server exposes the same `/act` wire protocol — `json_numpy`-encoded request/response — but with an embodiment-specific schema (camera count, state dimension, normalisation tag).

| Server | Checkpoint | Default port | State dim | Cameras |
| --- | --- | --- | --- | --- |
| [`examples/droid/host_server_droid.py`](examples/droid/host_server_droid.py) | [`allenai/MolmoAct2-DROID`](https://huggingface.co/allenai/MolmoAct2-DROID) | `8000` | `(8,) = [q1..q7, gripper]` | `external`, `wrist` |
| [`examples/yam/host_server_yam.py`](examples/yam/host_server_yam.py) | [`allenai/MolmoAct2-BimanualYAM`](https://huggingface.co/allenai/MolmoAct2-BimanualYAM) | `8202` | `(14,)` (per-arm 7-D × 2 arms) | `top`, `left`, `right` (order matters) |

### 1. Install [uv](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL          # reload PATH so the `uv` binary is picked up
uv --version
```

### 2. Create the project environment

The pinned dependencies (CUDA-12.8 PyTorch wheels, `transformers`, `fastapi`, `json-numpy`, …) live in `pyproject.toml`. CUDA 12.8 supports Blackwell GPUs, including RTX 50-series cards; it requires NVIDIA driver 570.26 or newer on Linux and 570.65 or newer on Windows. The CUDA toolkit itself does not need to be installed separately because the PyTorch wheels include the CUDA runtime.

From the repo root:

```bash
uv sync                  # creates .venv/ and installs all deps
uv run python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"
# expected: torch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8, and your GPU
```

The project supports Python 3.11 and 3.12; `uv` selects a compatible interpreter and downloads one if needed. Re-run `uv sync` after pulling new commits.

For an RTX 50-series GPU, confirm that the final architecture list contains `sm_120`. If inference fails with `no kernel image is available for execution on the device`, the environment is still using an older PyTorch wheel; run `uv sync --reinstall-package torch --reinstall-package torchvision` from the repository root and repeat the version check above.

### 3. Download the checkpoint (~22 GB each)

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1                       # fast parallel download
uv run hf download allenai/MolmoAct2-DROID               # for the DROID server
uv run hf download allenai/MolmoAct2-BimanualYAM         # for the YAM server
```

To put the cache on a different disk, set `HF_HOME=/path/to/cache` before the download (and when starting the server).

### 4. Start a server

```bash
# DROID (Franka)
uv run python examples/droid/host_server_droid.py --host 0.0.0.0 --port 8000 --dtype bfloat16

# Bimanual YAM
uv run python examples/yam/host_server_yam.py --host 0.0.0.0 --port 8202 --dtype bfloat16
```

Useful flags (both servers):

- `--dtype bfloat16|float16|float32` — default `bfloat16`. The DROID model card uses `float32` (~88 GB), which only fits on ~96 GB of free VRAM. The YAM model card reports `float32` at ~26 GB (fits on a single A6000), `bfloat16` under 16 GB. `bfloat16` is the safe default for both.
- `--device cuda:0`
- `--cuda-graph` — enables CUDA-graph capture for the action expert (~2× faster per call, ~2 GB extra VRAM). Disabled by default so the server coexists with other GPU workloads.
- `--no-warmup` — skip the dummy forward pass at startup.

#### bf16 patches

Loading in `bfloat16` is not officially supported by the upstream MolmoAct2 code; each server applies two idempotent patches to the cached `modeling_molmoact2.py` at startup:

1. flow-matching trajectory uses the model dtype instead of hardcoded `float32` (otherwise the action expert errors with `mat1 and mat2 must have the same dtype`),
2. `_to_array` casts to `float32` before `.numpy()` (numpy has no bf16 dtype).

Both are marked with `# patched_bf16_*` comments and re-applied on every server start, so re-downloading the checkpoint won't break things. Newer snapshot revisions (e.g. YAM) have already fixed both upstream; the server will log "needle not found" warnings, which are harmless.

### 5. Reach it from the LAN

Bound to `0.0.0.0`, the server is reachable on every interface of this host. Health check:

```bash
curl http://<lan-ip>:8000/act
# DROID: {"status":"ok","repo_id":"allenai/MolmoAct2-DROID","norm_tag":"franka_droid",...}

curl http://<lan-ip>:8202/act
# YAM:   {"status":"ok","repo_id":"allenai/MolmoAct2-BimanualYAM","norm_tag":"yam_dual_molmoact2","num_cameras":3,"state_dim":14,...}
```

The wire format (`json_numpy`-encoded request) is documented in the docstring at the top of each server file. The DROID server expects `external_cam`, `wrist_cam`, `instruction`, `state`; the YAM server expects `top_cam`, `left_cam`, `right_cam`, `instruction`, `state`. Both return `actions` (`(N, D)` float32) and `dt_ms`.

### Firewall / port

If clients on the LAN can't connect, open the port locally:

```bash
sudo ufw allow from <subnet> to any port 8000 proto tcp   # DROID
sudo ufw allow from <subnet> to any port 8202 proto tcp   # YAM
```

## 6. Pre-training and Post-training

[Full code](https://github.com/allenai/molmoact2/tree/main/experiments)

## 7. License

This model is licensed under Apache 2.0. It is intended for research and educational use in accordance with Ai2's [Responsible Use Guidelines](https://allenai.org/responsible-use).

## 8. Model and Hardware Safety
MolmoAct2 generate robot actions from visual observations and language instructions, but their behavior may vary across embodiments, environments, and hardware configurations. Users should carefully validate model outputs before deployment, especially when operating physical robots or other actuated systems. Where possible, actions should be monitored through interpretable intermediate outputs (adaptive depth map), simulation rollouts, action limits, or other safety checks before execution on hardware. The model’s action space should be bounded by the training data, robot controller limits, and task-specific safety constraints, including limits on speed, workspace, torque, and contact force. Users should follow the hardware manufacturer’s safety guidelines, use appropriate emergency-stop mechanisms, and operate the system only in a safely configured environment with human supervision.

## 9. Contacts

For questions, collaborations, or support, please contact with:
```
{hqfang,duanj1}@cs.washington.edu 
```
Found a bug or have a feature request? Please open a GitHub issue.

## 10. Citation

```bibtex
@misc{fang2026molmoact2actionreasoningmodels,
      title={MolmoAct2: Action Reasoning Models for Real-world Deployment}, 
      author={Haoquan Fang and Jiafei Duan and Donovan Clay and Sam Wang and Shuo Liu and Weikai Huang and Xiang Fan and Wei-Chuan Tsai and Shirui Chen and Yi Ru Wang and Shanli Xing and Jaemin Cho and Jae Sung Park and Ainaz Eftekhar and Peter Sushko and Karen Farley and Angad Wadhwa and Cole Harrison and Winson Han and Ying-Chun Lee and Eli VanderBilt and Rose Hendrix and Suveen Ellawela and Lucas Ngoo and Joyce Chai and Zhongzheng Ren and Ali Farhadi and Dieter Fox and Ranjay Krishna},
      year={2026},
      eprint={2605.02881},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.02881}, 
}
```
