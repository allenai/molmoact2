import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, fields, replace
from os.path import join
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import debugpy

from omegaconf import omegaconf, OmegaConf
from launch_scripts.data_mixtures import TAG_METADATA_BY_TAG
from launch_scripts.lerobot_utils.env import (
    _prepare_subprocess_environment_for_training,
    _run_torchcodec_preflight,
    _set_lerobot_environment_from_args,
    _store_env_json_in_file,
)
from launch_scripts.lerobot_utils.hf import (
    _apply_lerobot_molmoact2_defaults,
    get_hf_model_config,
)
from launch_scripts.lerobot_utils.train_plan import (
    ACTION_TOKENIZER_MAX_ACTION_DIM,
    _dedupe_tokens_preserve_order,
    _parse_bool_arg,
    _normalize_registered_lerobot_tag_metadata,
    _normalize_style_sampling_rates,
    _reject_removed_action_training_flags,
    _validate_action_training_args,
    _sync_vlm_data_cfg_with_primary,
    _validate_lerobot_tag_temporal_metadata,
    _validate_packed_action_chunk_padding_args,
    _validate_separate_vlm_dataloader_args,
    _build_vlm_data_cfg,
    get_lerobot_training_data_plan,
    infer_max_action_dim_from_lerobot_metadata,
    infer_max_action_horizon_from_lerobot_metadata,
)
from launch_scripts.lerobot_utils.stats import (
    _apply_tag_metadata_masks,
    _collect_tagged_stats,
)
from olmo.data.data_loader import DataLoaderConfig
from olmo.data.dynamic_packer import PackingConfig
from olmo.data.robot_processing import RobotProcessorConfig
from olmo.extra_tokens import (
    ACTION_TOKENS,
    DEFAULT_NUM_DEPTH_TOKENS,
    DEFAULT_NUM_STATE_TOKENS,
    DEPTH_TOKENS,
    SUPPORTED_STATE_FORMATS,
    STATE_TOKENS,
    build_action_added_tokens,
    build_depth_added_tokens,
    build_state_added_tokens,
    build_setup_added_tokens,
    build_control_added_tokens,
    style_uses_depth_output,
)
from olmo.io import file_exists
from olmo.model_configs import VISION_BACKBONES, LLMS
from olmo.models.molmo.data_formatter import DataFormatter
from olmo.models.molmo.molmo import MolmoConfig
from olmo.models.molmo.molmo_preprocessor import MolmoPreprocessorConfig
from olmo.models.molmoact2.molmoact2 import MolmoAct2Config
from olmo.models.molmo2.molmo2_preprocessor import Molmo2PreprocessorConfig
from olmo.nn.vision_backbone import MolmoVisionBackboneConfig
from olmo.preprocessing.multicrop_preprocessor import MultiCropConfig
from olmo.preprocessing.image_preprocessor import SUPPORTED_IMAGE_AUGMENTATION_MODES
from olmo.tokenizer import DEFAULT_PAD_MULTIPLE
from olmo.torch_util import get_world_size
from olmo.train.optim import OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType
from olmo.train.run_trainer import run_trainer
from olmo.train.trainer_config import TrainConfig, CompilerConfig, FSDPConfig, BatchDivisor, \
    SpeedMonitorConfig, WandbConfig
from olmo.util import (
    clean_opt,
    is_hf_checkpoint_ref,
    prepare_torchrun_environment,
    select_checkpoint,
)

log = logging.getLogger(__name__)


def get_model(checkpoint, model, frame_loading_backend: str = "torchcodec_exact"):
    if checkpoint == "8b":
        image_vit = VISION_BACKBONES["siglip2"]
        model_cfg = MolmoConfig(
            llm=replace(
                LLMS["qwen3_8b"],
                residual_dropout=0.0,
                response_residual_dropout=0.1,
                additional_vocab_size=128,
            ),
            vision_backbone=MolmoVisionBackboneConfig(
                vit=image_vit,
                vit_layers=[-3, -9],
                image_padding_embed=None
            ),
            data_formatter=DataFormatter(
                system_prompt='style_and_length_v2',
                message_format="qwen3",
                pointing_format="html-v1",
                always_start_with_space=False,
            ),
            mm_preprocessor=MolmoPreprocessorConfig(
                crop_mode="overlap-and-resize-c2",
                max_crops=8,
                overlap_margins=(4, 4)
            )
        )
    elif is_hf_checkpoint_ref(checkpoint):
        return _apply_lerobot_molmoact2_defaults(
            get_hf_model_config(checkpoint, frame_loading_backend=frame_loading_backend)
        )
    elif file_exists(join(checkpoint, "model.yaml")):
        model_cfg = MolmoConfig.load(join(checkpoint, "model.yaml"))
    else:
        model_cfg = MolmoConfig.load(join(checkpoint, "config.yaml"), key="model")

    video_pre_processor_cfg = Molmo2PreprocessorConfig(
        use_col_tokens=False,
        max_crops=1,
        pooling_h=3,
        pooling_w=3,
        high_res_pooling_h=None,
        high_res_pooling_w=None,
        periodic_high_res_frame=None,
        time_mode="per-frame-compact",

        max_frames=64,
        time_sampling=True,
        loading_method=frame_loading_backend,
        frame_sample_mode="uniform_last_frame",
        max_fps=[2],
    )
    if isinstance(model_cfg.mm_preprocessor, MultiCropConfig):
        image_preprocessor_args = asdict(model_cfg.mm_preprocessor)
        image_preprocessor = MultiCropConfig(**{
            k.name: image_preprocessor_args[k.name] for k in
            fields(MultiCropConfig)
        })
        image_preprocessor.high_res_max_crops = 24
        image_preprocessor.p_high_res = 0
        video_pre_processor_cfg.image = image_preprocessor
    else:
        video_pre_processor_cfg.image = model_cfg.mm_preprocessor.image

    model_cfg = MolmoAct2Config(
        llm=model_cfg.llm,
        vision_backbone=model_cfg.vision_backbone,
        data_formatter=model_cfg.data_formatter,
        mm_preprocessor=video_pre_processor_cfg,
        bi_directional_attn=model_cfg.bi_directional_attn
    )

    return _apply_lerobot_molmoact2_defaults(model_cfg)



def main():
    argv_tokens = sys.argv[1:]
    parser = argparse.ArgumentParser(prog="Train a multitask model")
    parser.add_argument("checkpoint", help="Path to checkpoint to start from")
    parser.add_argument("mixture", nargs="?", default="pre_post_train")
    parser.add_argument("--debug", type=_parse_bool_arg, default=False, metavar="BOOL")
    parser.add_argument("--debugger", type=_parse_bool_arg, default=False, metavar="BOOL")
    parser.add_argument("--model", default="video")
    parser.add_argument(
        "--frame_loading_backend",
        choices=["decord_with_av_fallback", "torchcodec_exact", "torchcodec_approx", "av"],
        default="torchcodec_exact",
    )
    parser.add_argument(
        "--pin_memory",
        type=_parse_bool_arg,
        default=False,
        metavar="BOOL",
    )
    parser.add_argument("--seq_len", type=int)
    parser.add_argument(
        "--separate_vlm_dataloader",
        type=_parse_bool_arg,
        default=False,
        metavar="BOOL",
        help=(
            "If true, build a dedicated VLM dataloader for non-LeRobot datasets and interleave it "
            "with the primary robot dataloader using the mixture's top-level regular/action weights."
        ),
    )
    parser.add_argument(
        "--vlm_seq_len",
        type=int,
        default=None,
        help="Sequence length for the dedicated VLM dataloader when --separate_vlm_dataloader=true.",
    )
    parser.add_argument("--device_batch_size", default=2, type=int)
    parser.add_argument("--global_batch_size", default=128, type=int)
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--max_loss_examples", default=2048, type=int)
    parser.add_argument("--max_inf_eval_examples", default=1280, type=int)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--connector_learning_rate", default=5e-6, type=float)
    parser.add_argument("--vit_learning_rate", default=5e-6, type=float)
    parser.add_argument("--llm_learning_rate", default=1e-5, type=float)
    parser.add_argument("--action_expert_learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--max_action_dim",
        default=None,
        type=int,
        help=(
            "Maximum action dimension used for model/action padding. Defaults to the maximum "
            "action_dim declared by the selected LeRobot mixture metadata."
        ),
    )
    parser.add_argument(
        "--action_dim",
        dest="legacy_action_dim",
        default=None,
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max_action_horizon",
        default=None,
        type=int,
        help=(
            "Maximum action horizon used for model/action padding. Defaults to the maximum "
            "action_horizon declared by the selected LeRobot mixture metadata."
        ),
    )
    parser.add_argument("--n_obs_steps", default=1, type=int)
    parser.add_argument("--num_flow_timesteps", default=1, type=int)
    parser.add_argument("--flow_matching_beta_alpha", default=1.0, type=float)
    parser.add_argument("--flow_matching_beta_beta", default=1.5, type=float)
    parser.add_argument("--flow_matching_cutoff", default=1.0, type=float)
    parser.add_argument("--flow_matching_time_offset", default=0.001, type=float)
    parser.add_argument("--flow_matching_time_scale", default=0.999, type=float)
    parser.add_argument(
        "--add_action_expert",
        type=_parse_bool_arg,
        default=True,
        metavar="BOOL",
        help="If true, build the MolmoAct2 action expert branch. Disable for pure autoregressive pretraining.",
    )
    parser.add_argument(
        "--mask_action_dim_padding",
        type=_parse_bool_arg,
        default=True,
        metavar="BOOL",
        help=(
            "If true, exclude right-padded action dimensions from flow-matching dynamics and loss. "
            "If false, train on the padded suffix as dense zero targets."
        ),
    )
    parser.add_argument(
        "--action_expert_depth_gate",
        type=_parse_bool_arg,
        default=False,
        metavar="BOOL",
        help="If true, learn a scalar gate that scales depth-token conditioning before the action expert.",
    )
    parser.add_argument(
        "--action_expert_depth_gate_per_layer",
        type=_parse_bool_arg,
        default=False,
        metavar="BOOL",
        help="If true, learn a separate depth gate for each selected action-expert conditioning layer.",
    )
    parser.add_argument("--action_expert_depth_gate_init_bias", default=-4.0, type=float)
    parser.add_argument("--action_expert_causal_attn", type=_parse_bool_arg, default=False, metavar="BOOL")
    parser.add_argument("--action_expert_detach_vlm", type=_parse_bool_arg, default=False, metavar="BOOL")
    parser.add_argument("--ft_vlm", type=_parse_bool_arg, default=False, metavar="BOOL")
    parser.add_argument("--ft_embedding", default="lm_head", type=str)
    parser.add_argument("--ft_action_expert", type=_parse_bool_arg, default=False, metavar="BOOL")
    parser.add_argument(
        "--img_aug",
        nargs="?",
        const="photometric",
        default="full",
        choices=list(SUPPORTED_IMAGE_AUGMENTATION_MODES),
        type=str,
        help=(
            "Image augmentation mode. Use 'photometric' for non-spatial appearance-only augmentation, "
            "'full' to include spatial distortions such as crop/rotation, or choose 'none' to disable."
        ),
    )
    parser.add_argument(
        "--norm_mode",
        default="q01_q99",
        choices=["mean_std", "min_max", "q01_q99", "q10_q90", "none"],
        type=str,
        help="Robot state/action normalization mode. Use 'none' to disable normalization.",
    )
    parser.add_argument(
        "--action_format",
        default="continuous",
        choices=["continuous", "discrete", "both"],
        type=str,
        help="LeRobot action supervision format: continuous, discrete, or both.",
    )
    parser.add_argument(
        "--state_format",
        default="discrete",
        choices=sorted(SUPPORTED_STATE_FORMATS),
        type=str,
        help="LeRobot state conditioning format: continuous, discrete, or both.",
    )
    parser.add_argument(
        "--discrete_action_tokenizer",
        default=None,
        type=str,
        help="Action tokenizer used by discrete and both action supervision.",
    )
    parser.add_argument("--img_resize", default=None, type=str)
    parser.add_argument("--crop_mode", default="resize", type=str)
    parser.add_argument("--lora_enable", type=_parse_bool_arg, default=False, metavar="BOOL")
    parser.add_argument("--lora_rank", default=64, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_bias", default="none", type=str)
    parser.add_argument("--add_action_tokens", type=_parse_bool_arg, default=True, metavar="BOOL")
    parser.add_argument("--num_action_tokens", default=2048, type=int)
    parser.add_argument("--add_state_tokens", type=_parse_bool_arg, default=True, metavar="BOOL")
    parser.add_argument("--num_state_tokens", default=DEFAULT_NUM_STATE_TOKENS, type=int)
    parser.add_argument("--add_depth_tokens", type=_parse_bool_arg, default=True, metavar="BOOL")
    parser.add_argument("--num_depth_tokens", default=DEFAULT_NUM_DEPTH_TOKENS, type=int)
    parser.add_argument("--enable_depth_reasoning", type=_parse_bool_arg, default=False, metavar="BOOL")
    parser.add_argument("--style_robot_action", default=1.0, type=float)
    parser.add_argument("--style_robot_depth", default=0.0, type=float)
    parser.add_argument("--style_robot_depth_action", default=0.0, type=float)
    parser.add_argument("--num_depth_tokens_per_image", default=100, type=int)
    parser.add_argument(
        "--depth_code_input_noise_rate",
        default=0.0,
        type=float,
        help=(
            "Training-only fraction of teacher-forced depth code input tokens to replace "
            "with random depth code tokens. Labels stay unchanged."
        ),
    )
    parser.add_argument("--add_setup_tokens", type=_parse_bool_arg, default=True, metavar="BOOL")
    parser.add_argument("--add_control_tokens", type=_parse_bool_arg, default=True, metavar="BOOL")
    parser.add_argument(
        "--random_camera_order",
        default="none",
        choices=["none", "episode", "all"],
        type=str,
        help="Optional camera-order augmentation for multi-camera LeRobot inputs.",
    )
    parser.add_argument(
        "--use_annotated_task",
        type=_parse_bool_arg,
        default=False,
        metavar="BOOL",
        help="If true, use per-episode tasks from meta/tasks_annotated.parquet when available.",
    )
    parser.add_argument(
        "--sample_annotated_task",
        type=_parse_bool_arg,
        default=False,
        metavar="BOOL",
        help="If true, sample annotated tasks from meta/task_to_episode.parquet by task_index before falling back.",
    )
    parser.add_argument("--packing", type=_parse_bool_arg, default=False, metavar="BOOL")
    parser.add_argument(
        "--dynamic_seq_len",
        type=_parse_bool_arg,
        default=False,
        metavar="BOOL",
        help=(
            "If true, disable fixed to_max padding for non-packing training and pad each batch to "
            "its observed max sequence length. Without this flag, --seq_len is required."
        ),
    )
    parser.add_argument(
        "--pad_packed_action_chunks",
        type=_parse_bool_arg,
        default=False,
        metavar="BOOL",
        help=(
            "If true, pad each packed sample to a fixed number of action chunks in the collator "
            "to stabilize the action-expert batch shape."
        ),
    )
    parser.add_argument(
        "--packed_action_chunk_cap",
        default=None,
        type=int,
        help=(
            "Maximum number of packed action chunks selected per packed sample when "
            "--pad_packed_action_chunks=true. The collator pads up to this cap."
        ),
    )
    parser.add_argument(
        "--skip_overlong_examples",
        type=_parse_bool_arg,
        default=True,
        metavar="BOOL",
        help="If true, skip and resample examples whose supervised tokens fall beyond the configured sequence length.",
    )
    parser.add_argument(
        "--skip_missing_vlm_examples",
        type=_parse_bool_arg,
        default=False,
        metavar="BOOL",
        help=(
            "If true, skip and resample non-LeRobot VLM examples whose image/video files are missing, "
            "logging a warning for every skipped sample."
        ),
    )
    parser.add_argument(
        "--skip_all_loss_truncated_examples",
        dest="skip_overlong_examples",
        type=_parse_bool_arg,
        metavar="BOOL",
        help=argparse.SUPPRESS,
    )
    _reject_removed_action_training_flags(argv_tokens)
    args, other_args = parser.parse_known_args()

    explicit_max_action_dim = any(
        token == "--max_action_dim" or token.startswith("--max_action_dim=")
        for token in argv_tokens
    )
    if args.legacy_action_dim is not None:
        if explicit_max_action_dim and int(args.legacy_action_dim) != int(args.max_action_dim):
            raise ValueError(
                "Received conflicting action dimension arguments: "
                f"--action_dim={int(args.legacy_action_dim)} vs --max_action_dim={int(args.max_action_dim)}."
            )
        args.max_action_dim = int(args.legacy_action_dim)
    _validate_packed_action_chunk_padding_args(args)
    prepare_torchrun_environment()

    if args.debugger:
        if os.environ.get("RANK", "0") == "0":
            # Listen on port 5678 (adjust if needed)
            debugpy.listen(("172.17.0.1", 5678))
            print("Debugger is listening on port 5678. Waiting for client to attach...")
            debugpy.wait_for_client()

    training_data_plan = get_lerobot_training_data_plan(
        args.mixture,
        style_robot_action=float(args.style_robot_action),
        style_robot_depth=float(args.style_robot_depth),
        style_robot_depth_action=float(args.style_robot_depth_action),
    )
    inferred_max_action_horizon = infer_max_action_horizon_from_lerobot_metadata(
        training_data_plan.robot_mixture,
        tag_metadata_by_tag=TAG_METADATA_BY_TAG,
    )
    inferred_max_action_dim = infer_max_action_dim_from_lerobot_metadata(
        training_data_plan.robot_mixture,
        tag_metadata_by_tag=TAG_METADATA_BY_TAG,
    )
    if args.max_action_horizon is None:
        args.max_action_horizon = inferred_max_action_horizon
    elif int(args.max_action_horizon) != int(inferred_max_action_horizon):
        raise ValueError(
            "--max_action_horizon is derived from the selected LeRobot mixture and should not be "
            "overridden. "
            f"Mixture '{args.mixture}' requires max_action_horizon={inferred_max_action_horizon}, "
            f"but got {int(args.max_action_horizon)}."
        )
    if args.max_action_dim is None:
        args.max_action_dim = inferred_max_action_dim
    elif int(args.max_action_dim) < int(inferred_max_action_dim):
        raise ValueError(
            "--max_action_dim must be at least the maximum action_dim declared by the selected "
            "LeRobot mixture metadata. "
            f"Mixture '{args.mixture}' requires max_action_dim={inferred_max_action_dim}, "
            f"but got {int(args.max_action_dim)}."
        )
    if int(args.max_action_dim) < 1:
        raise ValueError("--max_action_dim must be >= 1")
    if int(args.max_action_dim) > ACTION_TOKENIZER_MAX_ACTION_DIM:
        log.info(
            "Training with max_action_dim=%d (> %d). Continuous action training is enabled; "
            "checkpoint loading will adjust the action expert action-dimension weights.",
            int(args.max_action_dim),
            ACTION_TOKENIZER_MAX_ACTION_DIM,
        )
    _validate_action_training_args(
        args.action_format,
        max_action_dim=int(args.max_action_dim),
        discrete_action_tokenizer=args.discrete_action_tokenizer,
    )
    _validate_lerobot_tag_temporal_metadata(
        training_data_plan.robot_mixture,
        tag_metadata_by_tag=TAG_METADATA_BY_TAG,
        max_action_horizon=int(args.max_action_horizon),
    )
    _validate_separate_vlm_dataloader_args(args, training_data_plan)
    _set_lerobot_environment_from_args(args)

    if args.separate_vlm_dataloader:
        train_mixture = training_data_plan.robot_mixture
        vlm_mixture = training_data_plan.vlm_mixture
        vlm_loader_rate = training_data_plan.vlm_loader_rate
    else:
        train_mixture = training_data_plan.combined_mixture
        vlm_mixture = None
        vlm_loader_rate = None

    normalized_style_rates = _normalize_style_sampling_rates(training_data_plan.robot_style_mixture)
    os.environ["LEROBOT_STYLE_SAMPLING_RATES"] = json.dumps(normalized_style_rates)
    if not train_mixture:
        raise ValueError("At least one training mixture must be provided.")
    seq_len = args.seq_len
    if seq_len is not None and seq_len < 1:
        raise ValueError("--seq_len must be >= 1 when provided.")
    if args.dynamic_seq_len and args.packing:
        raise ValueError("--dynamic_seq_len=true is only supported when --packing=false.")
    if seq_len is None and not args.dynamic_seq_len:
        raise ValueError("--seq_len is required unless --dynamic_seq_len=true.")

    if args.checkpoint == "8b":
        checkpoint = None
        model_cfg = get_model("8b", args.model, frame_loading_backend=args.frame_loading_backend)
    else:
        checkpoint = select_checkpoint(args.checkpoint)
        model_cfg = get_model(checkpoint, args.model, frame_loading_backend=args.frame_loading_backend)
    model_cfg.mm_preprocessor.max_subtitle_tokens = None
    model_cfg.mm_preprocessor.use_frame_special_tokens = True
    model_cfg.mm_preprocessor.max_frames = 8
    model_cfg.mm_preprocessor.max_fps = [2]
    model_cfg.mm_preprocessor.image.crop_mode = args.crop_mode
    model_cfg.max_action_dim = args.max_action_dim
    model_cfg.action_expert.max_action_dim = args.max_action_dim
    model_cfg.action_horizon = args.max_action_horizon
    if hasattr(model_cfg, "n_action_steps"):
        model_cfg.n_action_steps = None
    model_cfg.n_obs_steps = args.n_obs_steps
    if args.num_flow_timesteps is not None:
        if args.num_flow_timesteps < 1:
            raise ValueError("--num_flow_timesteps must be >= 1")
        model_cfg.num_flow_timesteps = args.num_flow_timesteps
    if args.flow_matching_beta_alpha is not None:
        if args.flow_matching_beta_alpha <= 0:
            raise ValueError("--flow_matching_beta_alpha must be > 0")
        model_cfg.flow_matching_beta_alpha = args.flow_matching_beta_alpha
    if args.flow_matching_beta_beta is not None:
        if args.flow_matching_beta_beta <= 0:
            raise ValueError("--flow_matching_beta_beta must be > 0")
        model_cfg.flow_matching_beta_beta = args.flow_matching_beta_beta
    if args.flow_matching_cutoff is not None:
        if not 0 < args.flow_matching_cutoff <= 1:
            raise ValueError("--flow_matching_cutoff must be in (0, 1]")
        model_cfg.flow_matching_cutoff = args.flow_matching_cutoff
    if args.flow_matching_time_offset is not None:
        if not 0 <= args.flow_matching_time_offset < 1:
            raise ValueError("--flow_matching_time_offset must be in [0, 1)")
        model_cfg.flow_matching_time_offset = args.flow_matching_time_offset
    if args.flow_matching_time_scale is not None:
        if args.flow_matching_time_scale <= 0:
            raise ValueError("--flow_matching_time_scale must be > 0")
        model_cfg.flow_matching_time_scale = args.flow_matching_time_scale
    if model_cfg.flow_matching_time_offset > model_cfg.flow_matching_cutoff:
        raise ValueError(
            "flow_matching_time_offset must be <= flow_matching_cutoff "
            f"(got {model_cfg.flow_matching_time_offset} > {model_cfg.flow_matching_cutoff})"
        )
    model_cfg.add_action_expert = bool(args.add_action_expert)
    model_cfg.mask_action_chunk_padding = True
    model_cfg.mask_action_dim_padding = bool(args.mask_action_dim_padding)
    model_cfg.action_expert.max_horizon = args.max_action_horizon
    model_cfg.action_format = args.action_format
    model_cfg.state_format = args.state_format
    model_cfg.enable_depth_reasoning = bool(args.enable_depth_reasoning)
    model_cfg.num_depth_codes = int(args.num_depth_tokens_per_image)
    if not 0.0 <= float(args.depth_code_input_noise_rate) <= 1.0:
        raise ValueError("--depth_code_input_noise_rate must be in [0, 1].")
    model_cfg.depth_code_input_noise_rate = float(args.depth_code_input_noise_rate)
    model_cfg.data_formatter.add_setup_tokens = bool(args.add_setup_tokens)
    model_cfg.data_formatter.add_control_tokens = bool(args.add_control_tokens)
    model_cfg.vision_backbone.use_image_augmentation = args.img_aug
    model_cfg.action_expert_depth_gate = bool(args.action_expert_depth_gate)
    model_cfg.action_expert_depth_gate_per_layer = bool(args.action_expert_depth_gate_per_layer)
    model_cfg.action_expert_depth_gate_init_bias = float(args.action_expert_depth_gate_init_bias)
    model_cfg.action_expert.causal_attn = bool(args.action_expert_causal_attn)
    model_cfg.action_expert_detach_vlm = bool(args.action_expert_detach_vlm)

    # handling added tokens
    existing_added_tokens = list(model_cfg.llm.tokenizer.resolve_new_tokens_for_both_input_and_output())
    existing_added_tokens = _dedupe_tokens_preserve_order(existing_added_tokens)
    target_added_tokens = list(existing_added_tokens)
    if args.add_setup_tokens:
        target_added_tokens.extend(build_setup_added_tokens())
    if args.add_control_tokens:
        target_added_tokens.extend(build_control_added_tokens())
    if args.add_state_tokens:
        if args.num_state_tokens <= 0:
            raise ValueError("--num_state_tokens must be > 0 when --add_state_tokens is set")
        target_added_tokens.extend(build_state_added_tokens(args.num_state_tokens))
    if args.add_action_tokens:
        if args.num_action_tokens <= 0:
            raise ValueError("--num_action_tokens must be > 0 when --add_action_tokens is set")
        target_added_tokens.extend(build_action_added_tokens(args.num_action_tokens))
    if args.add_depth_tokens:
        if args.num_depth_tokens <= 0:
            raise ValueError("--num_depth_tokens must be > 0 when --add_depth_tokens is set")
        target_added_tokens.extend(build_depth_added_tokens(args.num_depth_tokens))
    target_added_tokens = _dedupe_tokens_preserve_order(target_added_tokens)

    target_discrete_action_bins = ACTION_TOKENS.count_bins(target_added_tokens)
    target_discrete_state_bins = STATE_TOKENS.count_bins(target_added_tokens)
    target_depth_bins = DEPTH_TOKENS.count_bins(target_added_tokens)
    has_action_boundaries = ACTION_TOKENS.has_boundaries(target_added_tokens)
    has_state_boundaries = STATE_TOKENS.has_boundaries(target_added_tokens)
    has_depth_boundaries = DEPTH_TOKENS.has_boundaries(target_added_tokens)
    if args.state_format in {"discrete", "both"}:
        if target_discrete_state_bins <= 0 or not has_state_boundaries:
            raise ValueError(
                "--state_format in {discrete,both} requires added state tokens "
                "(<state_start>, <state_end>, and at least one <state_i>) in the tokenizer."
            )
    if args.action_format in {"discrete", "both"}:
        if target_discrete_action_bins <= 0 or not has_action_boundaries:
            raise ValueError(
                "--action_format in {discrete,both} requires added action tokens "
                "(<action_start>, <action_end>, and at least one <action_i>) in the tokenizer."
            )
    if args.add_depth_tokens:
        if target_depth_bins <= 0 or not has_depth_boundaries:
            raise ValueError(
                "--add_depth_tokens requires added depth tokens "
                "(<depth_start>, <depth_end>, and at least one <depth_i>) in the tokenizer."
            )
    if args.num_depth_tokens_per_image <= 0:
        raise ValueError("--num_depth_tokens_per_image must be > 0")
    uses_depth_styles = any(
        rate > 0.0 and style_uses_depth_output(style_name)
        for style_name, rate in normalized_style_rates.items()
    )
    if args.enable_depth_reasoning:
        if not args.add_depth_tokens:
            raise ValueError("--enable_depth_reasoning requires --add_depth_tokens")
    if target_discrete_action_bins > 0:
        os.environ["LEROBOT_NUM_ACTION_TOKENS"] = str(target_discrete_action_bins)
    else:
        os.environ.pop("LEROBOT_NUM_ACTION_TOKENS", None)
    if target_discrete_state_bins > 0:
        os.environ["LEROBOT_NUM_STATE_TOKENS"] = str(target_discrete_state_bins)
    else:
        os.environ.pop("LEROBOT_NUM_STATE_TOKENS", None)
    if target_depth_bins > 0:
        os.environ["LEROBOT_NUM_DEPTH_TOKENS"] = str(target_depth_bins)
    else:
        os.environ.pop("LEROBOT_NUM_DEPTH_TOKENS", None)

    model_cfg.lora_enable = args.lora_enable
    model_cfg.lora_rank = args.lora_rank
    model_cfg.lora_alpha = args.lora_alpha
    model_cfg.lora_dropout = args.lora_dropout
    model_cfg.lora_bias = args.lora_bias

    base_vocab_size = int(model_cfg.llm.vocab_size or 0)
    current_embedding_size = int(model_cfg.llm.embedding_size or base_vocab_size)
    current_added_count = len(existing_added_tokens)

    # Probe tokenizer layout only when padding/added rows might already exist.
    needs_layout_probe = bool(
        model_cfg.llm.fix_pad_tokenizer or current_added_count > 0 or current_embedding_size > base_vocab_size
    )
    if needs_layout_probe:
        try:
            current_tokenizer = model_cfg.llm.build_tokenizer()
            # IM_START is the first EXTRA token; its id equals base+added+padding.
            current_core_size = int(getattr(current_tokenizer, "image_start_token_id"))
        except Exception as exc:
            current_core_size = current_embedding_size if model_cfg.llm.fix_pad_tokenizer else (
                base_vocab_size + current_added_count
            )
            log.warning(
                "Failed to infer tokenizer core size from build_tokenizer(), "
                "falling back to config sizes. Error: %s",
                exc,
            )
    else:
        current_core_size = base_vocab_size + current_added_count

    current_padding_tokens = max(current_core_size - base_vocab_size - current_added_count, 0)
    model_cfg.llm.tokenizer.new_tokens_for_both_input_and_output = target_added_tokens

    target_added_count = len(target_added_tokens)
    if target_added_count > 0:
        required_embedding_size = base_vocab_size + target_added_count + current_padding_tokens
        requested_embedding_size = max(current_embedding_size, required_embedding_size)
        if DEFAULT_PAD_MULTIPLE > 1:
            requested_embedding_size = (
                ((requested_embedding_size + DEFAULT_PAD_MULTIPLE - 1) // DEFAULT_PAD_MULTIPLE)
                * DEFAULT_PAD_MULTIPLE
            )
        model_cfg.llm.embedding_size = requested_embedding_size
        if not model_cfg.llm.fix_pad_tokenizer:
            model_cfg.llm.fix_pad_tokenizer = True
            log.info("Enabled llm.fix_pad_tokenizer=True for added-token insertion.")
        log.info(
            "Configured %d added tokens. embedding_size: %d -> %d "
            "(vocab_size=%d, additional_vocab_size=%d, previous_added_tokens=%d, "
            "preserved_padding_tokens=%d, inferred_core_size=%d, pad_multiple=%d, "
            "discrete_action_bins=%d, discrete_state_bins=%d)",
            target_added_count,
            current_embedding_size,
            int(model_cfg.llm.embedding_size),
            base_vocab_size,
            int(model_cfg.llm.additional_vocab_size or 0),
            current_added_count,
            current_padding_tokens,
            current_core_size,
            DEFAULT_PAD_MULTIPLE,
            target_discrete_action_bins,
            target_discrete_state_bins,
        )

    # Preserve action normalization metadata in the saved config so inference can rebuild processors.
    root_base = os.environ.get("LEROBOT_DATA_ROOT")
    lerobot_tag_metadata_by_tag = _normalize_registered_lerobot_tag_metadata(TAG_METADATA_BY_TAG)
    stats_by_tag, repo_to_tag, _default_tag = _collect_tagged_stats(
        train_mixture,
        root_base=root_base,
        tag_metadata_by_tag=TAG_METADATA_BY_TAG,
    )
    if stats_by_tag:
        _apply_tag_metadata_masks(
            stats_by_tag,
            lerobot_tag_metadata_by_tag,
        )
    if stats_by_tag:
        _store_env_json_in_file("LEROBOT_STATS_BY_TAG", stats_by_tag)
        _store_env_json_in_file("LEROBOT_REPO_TO_TAG", repo_to_tag)
        _store_env_json_in_file("LEROBOT_TAG_METADATA", lerobot_tag_metadata_by_tag)
        proc_cfg = RobotProcessorConfig.from_stats(
            stats_by_tag=stats_by_tag,
            tag_metadata=lerobot_tag_metadata_by_tag,
            repo_to_tag=repo_to_tag,
            norm_mode=args.norm_mode,
            data_formatter_add_setup_tokens=args.add_setup_tokens,
            data_formatter_add_control_tokens=args.add_control_tokens,
        )
        model_cfg.robot_processor = proc_cfg

    _prepare_subprocess_environment_for_training()
    _run_torchcodec_preflight(model_cfg)

    if args.debug:
        checkpoint = None
        model_cfg.llm.init_path = None
        model_cfg.llm.n_layers = 4
        model_cfg.vision_backbone.vit.init_path = None
        model_cfg.vision_backbone.vit.image_num_layers = 2
        model_cfg.vision_backbone.vit_layers = [-1, -2]
        model_cfg.action_expert.num_layers = model_cfg.llm.n_layers
        args.num_workers = 2
        args.prefetch_factor = 2

    num_workers = args.num_workers
    evaluations = []
    loss_evaluations = []

    log_interval = 1 if args.debug else args.log_interval
    dynamic_sequence_length = bool(args.dynamic_seq_len)
    preprocessor_sequence_length = seq_len
    if dynamic_sequence_length:
        preprocessor_sequence_length = model_cfg.max_sequence_length
        model_cfg.action_expert.compile = None
        log.info(
            "Using dynamic non-packing sequence lengths with preprocessor max_sequence_length=%s.",
            preprocessor_sequence_length,
        )

    primary_data_cfg = DataLoaderConfig(
        kwargs_mixture=train_mixture,
        shuffle=True,
        split="train",
        drop_last=True,
        sequence_length=preprocessor_sequence_length,
        max_text_seq_len=None,
        num_workers=num_workers,
        pad=None if dynamic_sequence_length else "to_max",
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        skip_overlong_examples=bool(args.skip_overlong_examples),
        skip_missing_vlm_examples=bool(args.skip_missing_vlm_examples),
        seed=50189,
        packing=PackingConfig(
            buffer_size=48,
            image_weight=30,
            shortcut_max_len_images=False,
            pad_action_chunks=bool(args.pad_packed_action_chunks),
            action_chunk_cap=args.packed_action_chunk_cap,
        ) if args.packing else None,
    )
    vlm_data_cfg = None
    if args.separate_vlm_dataloader:
        assert vlm_mixture is not None
        assert vlm_loader_rate is not None
        vlm_data_cfg = _build_vlm_data_cfg(
            primary_data_cfg,
            vlm_mixture=vlm_mixture,
            args=args,
        )

    cfg = TrainConfig(
        run_name="multitask_train",
        save_folder=omegaconf.MISSING,
        seed=6198,
        dry_run=False,

        wandb=None if args.debug else WandbConfig(
            name="${run_name}",
            project="${oc.env:WANDB_PROJECT}",
            group=None,
            entity="${oc.env:WANDB_ENTITY}",
            log_interval=log_interval,
            allow_resume=True,
            finish_on_sigterm=True
        ),
        compile=None if dynamic_sequence_length else CompilerConfig(mode="default", dynamic=False),
        fused_loss=False,
        allow_resume=True,
        model=model_cfg,
        save_overwrite=True,
        data=primary_data_cfg,
        vlm_data=vlm_data_cfg,
        vlm_loader_rate=vlm_loader_rate,
        ft_connector=args.ft_vlm,
        ft_llm=args.ft_vlm,
        ft_vit=args.ft_vlm,
        ft_embedding=args.ft_embedding,
        ft_action_expert=args.ft_action_expert,
        optimizer=OptimizerConfig(
            name=OptimizerType.adamw,
            connector_learning_rate=args.connector_learning_rate,
            vit_learning_rate=args.vit_learning_rate,
            llm_learning_rate=args.llm_learning_rate,
            action_expert_learning_rate=args.action_expert_learning_rate,
            connector_weight_decay=0.0,
            vit_weight_decay=0.0,
            llm_weight_decay=0.0,
            action_expert_weight_decay=0.0,
            connector_betas=[0.9, 0.95],
            vit_betas=[0.9, 0.95],
            llm_betas=[0.9, 0.95],
            action_expert_betas=[0.9, 0.95],
            connector_eps=1e-6,
            vit_eps=1e-6,
            llm_eps=1e-6,
            action_expert_eps=1e-6,
        ),
        scheduler=SchedulerConfig(
            name=SchedulerType.multimodal,
            connector_t_warmup=200,
            vit_t_warmup=200,
            llm_t_warmup=200,
            action_expert_t_warmup=200,
            alpha_f=0.1,
            warmup_min_lr=0.0
        ),
        fsdp=FSDPConfig(fsdp2=True),
        load_path=None,
        initial_model_checkpoint=checkpoint,
        save_interval=2000,
        save_num_checkpoints_to_keep=1,
        global_train_batch_size=get_world_size() if args.debug else args.global_batch_size,
        device_train_microbatch_size=args.device_batch_size,
        time_limit=None,
        max_duration=24000,
        stop_at="${max_duration}",
        max_grad_norm=1,
        batch_divisor=BatchDivisor.global_batch,
        precision="amp_bf16",
        console_log_interval=log_interval,
        compile_loss=True,
        speed_monitor=SpeedMonitorConfig(window_size=20),
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        inf_evaluators=evaluations,
        evaluators=loss_evaluations,
        inf_eval_interval=2000,
        eval_interval=2000,
        save_final_unsharded_checkpoint=False,
        save_merged_lora_checkpoint=True,
        save_final_optim=True,
        response_logits_only=True
    )
    os.environ["LEROBOT_RANDOM_CAMERA_ORDER_SEED"] = str(cfg.data.seed)

    conf = OmegaConf.create(cfg)
    conf.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    conf = OmegaConf.to_object(conf)
    conf = _sync_vlm_data_cfg_with_primary(conf)
    run_trainer(conf)


if __name__ == '__main__':
    main()
