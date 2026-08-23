# MolmoAct2 Experiments

This directory contains the open-sourced MolmoAct2 training and evaluation code used to replicate our experiments and fine-tune on new LeRobot datasets.

The experiments package is organized for release use: training commands are written directly below, Beaker specs and local shell wrappers are not part of the tracked source tree, and deployment documentation lives in the vendored LeRobot docs.

## Setup

Install the repository in editable mode:

```bash
git clone https://github.com/allenai/molmoact2.git
cd molmoact2/experiments
pip install -e ".[all]"
pip install -e "./lerobot[async,libero]"
```

Use `./lerobot[all]` instead of `./lerobot[async,libero]` when you need the full LeRobot hardware and simulator dependency set.

Configure local paths for your machine or scheduler:

```bash
export LEROBOT_DATA_ROOT=/path/to/lerobot/data
export LEROBOT_DEPTH_DATA_ROOT=/path/to/lerobot/depth_data
export MOLMO_DATA_DIR=/path/to/molmo/data
export SPATIAL_DATA_HOME=/path/to/spatial/embodied_training_data
export HF_HOME=/path/to/huggingface/cache
export LEROBOT_VIDEO_BACKEND=pyav
export PYTHONPATH="$PWD:$PWD/lerobot/src:${PYTHONPATH:-}"
```

Set credentials through environment variables or scheduler secrets:

```bash
export HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
```

## Downloading VLM Data

The VLM data pipeline uses both the Hugging Face cache and processed datasets under
`MOLMO_DATA_DIR`. Set both locations before running the downloader:

```bash
export MOLMO_DATA_DIR=/path/to/molmo/data
export HF_HOME=/path/to/molmo/data/huggingface
mkdir -p "${MOLMO_DATA_DIR}" "${HF_HOME}"
```

From the `experiments` directory, download one dataset, several datasets, or a built-in group:

```bash
python scripts/download_datasets.py pixmo_points_train --n-procs 8
python scripts/download_datasets.py pixmo_points_train cosyn_point --n-procs 8
python scripts/download_datasets.py pixmo
python scripts/download_datasets.py image_pointing
python scripts/download_datasets.py demo
```

`all` downloads the union of the built-in `pixmo`, `image_pointing`, `video_pointing`,
`video_tracking`, and `demo` groups. Duplicate entries are downloaded only once, and the
command reports every failed dataset before exiting nonzero:

```bash
python scripts/download_datasets.py all --n-procs 8
```

Public image data uses the same released sources as Molmo2. In particular,
`cosyn_point`, `pixmo_multi_points`, and `pixmo_multi_image_qa` are loaded from
`allenai/CoSyn-point`, `allenai/molmo2-pixmo-multi-points`, and
`allenai/Molmo2-MultiImageQA`. They do not require private metadata JSON files.

Processed image datasets are written below `${MOLMO_DATA_DIR}/torch_datasets`, shared
PixMo images below `${MOLMO_DATA_DIR}/torch_datasets/pixmo_images`, and Hugging Face
artifacts below `${HF_HOME}`. Downloads can be resumed safely because existing completed
datasets and image files are reused.

Some video and tracking datasets still require accepting a Hugging Face agreement or a
manual download because of their upstream licenses. Those datasets are not replaced by
the public PixMo download path; follow the error from the corresponding class in
`olmo/data/video_datasets.py` or `olmo/data/video_object_tracking_datasets.py`.

## Checkpoints

`launch_scripts/train_lerobot.py` accepts local checkpoint paths, URLs, and Hugging Face model IDs.

| Stage | Start checkpoint |
| --- | --- |
| ER training | `https://storage.googleapis.com/oe-training-public/Molmo2-1225/Molmo2-4B.tar` |
| MolmoAct2 pretraining | `allenai/Molmo2-ER` |
| MolmoAct2 post-training | `allenai/MolmoAct2-Pretrain` |
| Standard fine-tuning | `allenai/MolmoAct2` |
| Depth fine-tuning | `allenai/MolmoAct2-Think` |
| Bimanual YAM fine-tuning | `allenai/MolmoAct2-BimanualYAM` |
| DROID fine-tuning | `allenai/MolmoAct2-DROID` |
| SO100/SO101 fine-tuning | `allenai/MolmoAct2-SO100_101` |

## Depth Annotations

Generate depth companion datasets before depth post-training or depth fine-tuning:

```bash
python scripts/generate_depth_annotation.py \
  "${LEROBOT_DATA_ROOT}/<repo_id>" \
  --camera-key observation.images.primary
```

When `LEROBOT_DEPTH_DATA_ROOT` is set, the generator writes to `${LEROBOT_DEPTH_DATA_ROOT}/<repo_id>` by default, matching the training lookup. It writes `buffer_codes` with 100 entries by default, matching training's default `--num_depth_tokens_per_image 100`.

## Fine-Tuning A New Dataset

The primary use case for this repo is adapting MolmoAct2 to a new LeRobot dataset. Add a mixture in `launch_scripts/data_mixtures.py`, then train with full fine-tuning, LoRA, or action-expert-only tuning.

### Register A Mixture

For a single LeRobot dataset, use `build_single_lerobot_mixture` and register the builder in `MOLMOACT2_LEROBOT_MIXTURES`. `libero_goal` is the minimal built-in example:

```python
def build_molmoact2_libero_goal():
    return build_single_lerobot_mixture(
        name="libero_goal",
        tag="libero",
        repo_ids=["allenai/MolmoAct2-LIBERO-Dataset"],
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=[
            "observation.images.image",
            "observation.images.wrist_image",
        ],
        normalize_gripper=False,
        setup_type="single franka robotic arm in libero",
        control_mode="delta end-effector pose",
        action_horizon=10,
        n_action_steps=10,
    )


MOLMOACT2_LEROBOT_MIXTURES["libero_goal"] = build_molmoact2_libero_goal
```

For a new embodiment, create a new tag and describe the robot/action convention literally:

```python
def build_molmoact2_my_robot():
    return build_single_lerobot_mixture(
        name="my_robot",
        tag="my_robot",
        repo_ids=["my-org/my-lerobot-dataset"],
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=[
            "observation.images.front",
            "observation.images.wrist",
        ],
        normalize_gripper=False,
        setup_type="single robot arm in my workspace",
        control_mode="delta end-effector pose",
        action_horizon=30,
        n_action_steps=30,
    )


MOLMOACT2_LEROBOT_MIXTURES["my_robot"] = build_molmoact2_my_robot
```

Use the same `tag` only for datasets that share robot semantics, action/state normalization, camera conventions, `setup_type`, and `control_mode`.

### Packing And Training Budget

For custom-data fine-tuning, start with `--packing=false --dynamic_seq_len=true`, as in the recipes below. Packing is not required for fine-tuning, and the unpacked setup makes `global_batch_size` and `max_duration` straightforward to interpret: each batch entry is one dataset sample.

The release-reproduction recipes later in this document keep `--packing=true` because packing was part of those original training setups. With packing enabled, one fixed-length packed sequence can contain multiple source samples and action chunks, so a global batch size of 64 means 64 packed sequences per optimizer step, not necessarily 64 source samples. Packed and unpacked runs therefore should not be compared only by batch size and step count; also compare consumed samples, tokens, and action chunks. The trainer reports packing statistics such as `batch/n_packed` and `batch/packed_action_chunks_mean_actual` for this purpose.

`--packing=true` and `--dynamic_seq_len=true` are mutually exclusive. Only enable packing for release reproduction or after profiling it as a deliberate throughput optimization for a custom workload.

### Smoke Test

For a quick single-dataset validation, disable packing and use dynamic sequence lengths:

```bash
export EXP_NAME="molmoact2-my-robot-smoke"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --standalone --nproc-per-node=1 \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 \
  my_robot \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --max_duration=20 \
  --device_batch_size=1 \
  --global_batch_size=1 \
  --num_workers=0 --pin_memory=false \
  --save_folder="checkpoints/smoke/${EXP_NAME}" \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=false \
  --ft_action_expert=true \
  --ft_embedding=none
```

### Full Fine-Tuning

Full fine-tuning updates the VLM, vision tower, connector, LM head, and action expert. Use it for larger datasets or substantial embodiment changes.

```bash
export EXP_NAME="molmoact2-my-robot-fft"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes="${NNODES:-1}" --nproc-per-node=8 \
  --node_rank="${RANK:-0}" --master_addr="${ADDR:-127.0.0.1}" --master_port="${PORT:-29415}" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 \
  my_robot \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --max_duration=50000 \
  --device_batch_size=2 \
  --global_batch_size=64 \
  --num_workers=4 --pin_memory=true \
  --data.timeout=900 \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=20 \
  --save_folder="checkpoints/finetune/${EXP_NAME}" \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=true \
  --ft_action_expert=true \
  --ft_embedding=lm_head \
  --lora_enable=false \
  --llm_learning_rate=1e-5 \
  --vit_learning_rate=5e-6 \
  --connector_learning_rate=5e-6 \
  --action_expert_learning_rate=5e-5
```

### LoRA Fine-Tuning

LoRA fine-tuning injects adapters only into linear layers in the VLM's LLM and ViT. It does not add LoRA adapters to the connector or action expert. With the recipe below, the connector and action expert are instead trained as regular full parameters because `--ft_vlm=true` and `--ft_action_expert=true`; set the corresponding flag to `false` to freeze them. Use LoRA for smaller datasets or similar embodiments.

```bash
export EXP_NAME="molmoact2-my-robot-lora"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes="${NNODES:-1}" --nproc-per-node=8 \
  --node_rank="${RANK:-0}" --master_addr="${ADDR:-127.0.0.1}" --master_port="${PORT:-29415}" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 \
  my_robot \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --max_duration=50000 \
  --device_batch_size=2 \
  --global_batch_size=64 \
  --num_workers=4 --pin_memory=true \
  --data.timeout=900 \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=20 \
  --save_folder="checkpoints/finetune/${EXP_NAME}" \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=true \
  --ft_action_expert=true \
  --ft_embedding=lm_head \
  --lora_enable=true \
  --lora_rank=64 \
  --llm_learning_rate=5e-5 \
  --vit_learning_rate=5e-5 \
  --connector_learning_rate=5e-5 \
  --action_expert_learning_rate=5e-5
```

#### LoRA Checkpoint Layout

Each LoRA save produces several directories with different purposes:

| Directory | Contents and intended use |
| --- | --- |
| `stepX` | Sharded model, optimizer, and trainer state. Use this directory to resume training at step X. It is not the deployment checkpoint. |
| `stepX-lora-llm` | LLM PEFT adapter files used as an intermediate input when building the merged checkpoint. This is not a complete MolmoAct2 checkpoint. |
| `stepX-lora-vision` | ViT PEFT adapter files used as an intermediate input when building the merged checkpoint. This is not a complete MolmoAct2 checkpoint. |
| `stepX-merged` | Complete unsharded model with the LLM and ViT adapters merged and all trained non-LoRA components, including the action expert. This is the final model checkpoint for inference or export. |

Convert the merged checkpoint to Hugging Face format with:

```bash
python -m olmo.hf_model.convert_molmoact2_to_hf \
  path/to/stepX-merged \
  path/to/stepX-hf
```

Alternatively, use `stepX-merged` directly with the vendored LeRobot inference, evaluation, or async serving tools:

```bash
--policy.type=molmoact2 \
--policy.checkpoint_path=path/to/stepX-merged
```

See [issue #30](https://github.com/allenai/molmoact2/issues/30) for the original checkpoint-layout question.

### Action-Expert-Only Fine-Tuning

Action-expert-only fine-tuning freezes the VLM, vision tower, connector, embeddings, and LM head. Use it when the vision-language behavior should remain fixed and the new dataset mainly changes the continuous control head.

```bash
export EXP_NAME="molmoact2-my-robot-ae-only"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes="${NNODES:-1}" --nproc-per-node=8 \
  --node_rank="${RANK:-0}" --master_addr="${ADDR:-127.0.0.1}" --master_port="${PORT:-29415}" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2 \
  my_robot \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --max_duration=50000 \
  --device_batch_size=2 \
  --global_batch_size=64 \
  --num_workers=4 --pin_memory=true \
  --data.timeout=900 \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=20 \
  --save_folder="checkpoints/finetune/${EXP_NAME}" \
  --packing=false \
  --dynamic_seq_len=true \
  --ft_vlm=false \
  --ft_action_expert=true \
  --ft_embedding=none \
  --lora_enable=false \
  --action_expert_learning_rate=5e-5
```

### Depth Fine-Tuning

For depth-reasoning fine-tuning, start from `allenai/MolmoAct2-Think` and add:

```bash
--enable_depth_reasoning=true \
--num_depth_tokens=128 \
--num_depth_tokens_per_image=100 \
--depth_code_input_noise_rate=0.1 \
--style_robot_action=1.0 \
--style_robot_depth=0.0 \
--style_robot_depth_action=1.0
```

## Application

The vendored `lerobot` package is the supported inference and deployment path for MolmoAct2. It covers simulator rollout, real-robot rollout/recording, async inference, CUDA graph inference, and the MolmoAct2 policy configuration surface.

See `lerobot/docs/source/molmoact2.mdx` for deployment commands and LeRobot-specific details.

## Reproducing Released Training Stages

The commands below reproduce the major MolmoAct2 training stages. They are references for release replication; for new projects, start with the new-dataset fine-tuning section above.

These commands intentionally retain the packed data settings used by the released runs. Do not copy their packed batch size and step budget directly to an unpacked custom-data fine-tune; use the custom fine-tuning defaults and budget guidance above instead.

### Molmo2-ER Training

Molmo2-ER starts from the public Molmo2-4B checkpoint archive, not from a Hugging Face model. The launcher downloads and extracts this archive into `${MOLMOACT2_CHECKPOINT_CACHE:-${HF_HOME}/molmoact2/checkpoints}` before loading it.

```bash
export EXP_NAME="molmo2-er"
export SPATIAL_DATA_HOME="${SPATIAL_DATA_HOME:-/path/to/spatial/embodied_training_data}"

WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes=2 --nproc-per-node=8 \
  --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
  launch_scripts/sft.py \
  https://storage.googleapis.com/oe-training-public/Molmo2-1225/Molmo2-4B.tar \
  molmo2_embodied_spatial_mix_50 \
  --seq_len=16384 \
  --device_batch_size=1 \
  --max_duration=20000 \
  --save_interval=1000 \
  --save_num_checkpoints_to_keep=20 \
  --num_workers=2 \
  --data.persistent_workers=true \
  --save_folder="checkpoints/er/${EXP_NAME}" \
  --wandb.name="${EXP_NAME}" \
  --wandb.entity=<wandb-entity> \
  --wandb.project=<wandb-project> \
  --model.vision_backbone.compile_connector=null \
  --model.mm_preprocessor.max_frames=128 \
  --model.mm_preprocessor.use_frame_special_tokens=true \
  --model.mm_preprocessor.max_subtitle_tokens=null
```

Use `spatial-all-v4` for the spatial-only stage, or `molmo2_embodied_spatial_mix_30`, `molmo2_embodied_spatial_mix_50`, `molmo2_embodied_spatial_mix_70`, and `molmo2_embodied_spatial_mix_90` for general/spatial recovery ablations.

### MolmoAct2 Pretraining

```bash
export EXP_NAME="molmoact2-pretrain"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes=8 --nproc-per-node=8 \
  --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
  launch_scripts/train_lerobot.py \
  allenai/Molmo2-ER \
  pre_post_train \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --seq_len=4200 \
  --max_duration=200000 \
  --device_batch_size=2 \
  --global_batch_size=128 \
  --log_interval=20 \
  --num_workers=4 --pin_memory=true \
  --data.timeout=900 \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=20 \
  --save_folder="checkpoints/pretrain/${EXP_NAME}" \
  --packing=true \
  --crop_mode=resize \
  --ft_embedding=added_tokens \
  --add_action_expert=false \
  --action_format=discrete \
  --ft_vlm=true \
  --connector_learning_rate=5e-6 \
  --vit_learning_rate=5e-6 \
  --llm_learning_rate=1e-5 \
  --random_camera_order=episode \
  --frame_loading_backend=torchcodec_exact \
  --use_annotated_task=true \
  --sample_annotated_task=false
```

### Post-Training

```bash
export EXP_NAME="molmoact2-posttrain"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes=8 --nproc-per-node=8 \
  --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2-Pretrain \
  pre_post_train \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --seq_len=2100 --vlm_seq_len=4200 \
  --max_duration=100000 \
  --device_batch_size=2 \
  --global_batch_size=128 \
  --num_workers=4 --pin_memory=true \
  --data.timeout=900 \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=20 \
  --save_folder="checkpoints/posttrain/${EXP_NAME}" \
  --separate_vlm_dataloader=true \
  --packing=true \
  --pad_packed_action_chunks=true \
  --packed_action_chunk_cap=5 \
  --crop_mode=resize \
  --ft_embedding=added_tokens \
  --ft_vlm=true \
  --ft_action_expert=true \
  --action_expert_learning_rate=5e-5 \
  --num_flow_timesteps=4 \
  --mask_action_dim_padding=true \
  --random_camera_order=episode \
  --frame_loading_backend=torchcodec_exact \
  --use_annotated_task=true \
  --sample_annotated_task=false
```

Depth post-training:

```bash
export EXP_NAME="molmoact2-posttrain-depth"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes=8 --nproc-per-node=8 \
  --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2-Pretrain \
  pre_post_train \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --seq_len=2100 --vlm_seq_len=4200 \
  --max_duration=100000 \
  --device_batch_size=2 \
  --global_batch_size=128 \
  --num_workers=4 --pin_memory=true \
  --data.timeout=900 \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=20 \
  --save_folder="checkpoints/posttrain/${EXP_NAME}" \
  --separate_vlm_dataloader=true \
  --skip_missing_vlm_examples=true \
  --packing=true \
  --pad_packed_action_chunks=true \
  --packed_action_chunk_cap=5 \
  --crop_mode=resize \
  --ft_embedding=added_tokens \
  --ft_vlm=true \
  --ft_action_expert=true \
  --action_expert_learning_rate=5e-5 \
  --num_flow_timesteps=4 \
  --mask_action_dim_padding=true \
  --random_camera_order=episode \
  --frame_loading_backend=torchcodec_exact \
  --use_annotated_task=true \
  --sample_annotated_task=false \
  --num_depth_tokens=128 \
  --enable_depth_reasoning=true \
  --num_depth_tokens_per_image=100 \
  --style_robot_action=1.0 \
  --style_robot_depth=1.0 \
  --style_robot_depth_action=1.0
```

### Fine-Tuning Existing Mixtures

Use the closest released checkpoint for the target embodiment:

- General LIBERO or Franka tabletop tasks: `allenai/MolmoAct2`
- Depth reasoning: `allenai/MolmoAct2-Think`
- DROID: `allenai/MolmoAct2-DROID`
- Bimanual YAM: `allenai/MolmoAct2-BimanualYAM`
- SO100/SO101: `allenai/MolmoAct2-SO100_101`

DROID full fine-tuning:

```bash
export EXP_NAME="molmoact2-droid"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes=4 --nproc-per-node=8 \
  --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2-DROID \
  droid \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --seq_len=2100 \
  --max_duration=50000 \
  --device_batch_size=2 \
  --global_batch_size=64 \
  --num_workers=4 --pin_memory=true \
  --data.timeout=900 \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=20 \
  --save_folder="checkpoints/finetune/${EXP_NAME}" \
  --packing=true \
  --crop_mode=resize \
  --ft_vlm=true \
  --ft_action_expert=true \
  --action_expert_learning_rate=5e-5 \
  --num_flow_timesteps=8 \
  --mask_action_dim_padding=true \
  --random_camera_order=none \
  --frame_loading_backend=torchcodec_exact \
  --use_annotated_task=false \
  --sample_annotated_task=false
```

Depth fine-tuning uses the same mixture names and `--enable_depth_reasoning=true`:

```bash
export EXP_NAME="molmoact2-droid-depth"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" WANDB_API_KEY="${WANDB_API_KEY:-}" torchrun \
  --nnodes=4 --nproc-per-node=8 \
  --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2-Think \
  droid \
  --wandb.name="${EXP_NAME}" --wandb.entity=<wandb-entity> --wandb.project=<wandb-project> \
  --seq_len=2100 \
  --max_duration=50000 \
  --device_batch_size=2 \
  --global_batch_size=64 \
  --num_workers=4 --pin_memory=true \
  --data.timeout=900 \
  --save_interval=10000 \
  --save_num_checkpoints_to_keep=20 \
  --save_folder="checkpoints/finetune/${EXP_NAME}" \
  --packing=true \
  --crop_mode=resize \
  --ft_vlm=true \
  --ft_action_expert=true \
  --action_expert_learning_rate=5e-5 \
  --num_flow_timesteps=8 \
  --mask_action_dim_padding=true \
  --random_camera_order=none \
  --frame_loading_backend=torchcodec_exact \
  --use_annotated_task=false \
  --sample_annotated_task=false \
  --num_depth_tokens=128 \
  --enable_depth_reasoning=true \
  --num_depth_tokens_per_image=100 \
  --depth_code_input_noise_rate=0.1 \
  --style_robot_action=1.0 \
  --style_robot_depth=0.0 \
  --style_robot_depth_action=1.0
```
