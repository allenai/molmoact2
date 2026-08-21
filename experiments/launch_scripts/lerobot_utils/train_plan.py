from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, fields, replace
from typing import Dict, List, Optional, Sequence

from launch_scripts.data_mixtures import (
    MOLMOACT2_LEROBOT_MIXTURES,
    TAG_METADATA_BY_TAG,
    is_lerobot_tag,
    reset_tag_metadata,
    strip_lerobot_tag_prefix,
)
from olmo.data.data_loader import DatasetWithArgs, KwargsMixture, DataLoaderConfig
from olmo.extra_tokens import ROBOT_OUTPUT_STYLES
from olmo.train.trainer_config import TrainConfig

log = logging.getLogger(__name__)
DEFAULT_DISCRETE_ACTION_TOKENIZER = "allenai/MolmoAct2-FAST-Tokenizer"
ACTION_TOKENIZER_MAX_ACTION_DIM = 32

_REMOVED_TRAINING_ACTION_FLAGS = {
    "--action_horizon": "Use --max_action_horizon instead.",
    "--n_action_steps": "n_action_steps is now required per tag metadata and inference-only.",
    "--mask_action_chunk_padding": "Action time padding is now always loss-masked for tag-to-max-horizon padding.",
    "--action_expert_impl": "MolmoAct2 only supports the modern action expert.",
    "--action_expert_pi05_variant": "The PI05 action expert implementation has been removed.",
    "--action_expert_gr00t_select_layer": "The GR00T action expert implementation has been removed.",
    "--action_expert_layer_mode": "MolmoAct2 always uses per-layer action expert conditioning.",
    "--action_expert_condition_source": "MolmoAct2 always conditions the action expert from VLM KV cache states.",
    "--state_token_value_encoding": "State token value encoding has been removed; state tokens are learned normally.",
    "--state_token_value_encoding_scale": "State token value encoding has been removed; this scale is unused.",
}


@dataclass
class LeRobotTrainingDataPlan:
    combined_mixture: List[KwargsMixture]
    robot_mixture: List[KwargsMixture]
    vlm_mixture: Optional[List[KwargsMixture]]
    vlm_loader_rate: Optional[float]
    robot_style_mixture: Dict[str, float]


def _parse_bool_arg(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'.")


def _reject_removed_action_training_flags(argv_tokens: Sequence[str]) -> None:
    for token in argv_tokens:
        for flag, guidance in _REMOVED_TRAINING_ACTION_FLAGS.items():
            if token == flag or token.startswith(f"{flag}="):
                raise ValueError(f"{flag} is no longer supported. {guidance}")


def _require_positive_tag_metadata_int(tag: str, metadata: Dict[str, object], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"LeRobot tag metadata for tag '{tag}' must define integer {key} >= 1."
        )
    value = int(value)
    if value < 1:
        raise ValueError(
            f"LeRobot tag metadata for tag '{tag}' must define integer {key} >= 1."
        )
    return value


def _get_used_lerobot_tags(robot_mixture: Sequence[KwargsMixture]) -> set[str]:
    used_tags = set()
    for mix in robot_mixture:
        raw_tag = str(mix.name or "default")
        if not any(ds_args.dataset_name.startswith("lerobot:") for ds_args in mix.datasets):
            continue
        if not is_lerobot_tag(raw_tag):
            continue
        used_tags.add(strip_lerobot_tag_prefix(raw_tag))
    return used_tags


def infer_max_action_horizon_from_lerobot_metadata(
    robot_mixture: Sequence[KwargsMixture],
    *,
    tag_metadata_by_tag: Dict[str, Dict[str, object]],
) -> int:
    normalized_metadata = _normalize_registered_lerobot_tag_metadata(tag_metadata_by_tag)
    used_tags = _get_used_lerobot_tags(robot_mixture)
    if not used_tags:
        raise ValueError("Cannot infer max_action_horizon because the training mixture has no LeRobot tags.")

    max_action_horizon = 0
    for tag in sorted(used_tags):
        metadata = normalized_metadata.get(tag)
        if not metadata:
            raise ValueError(f"Missing required LeRobot tag metadata for tag '{tag}'.")
        action_horizon = _require_positive_tag_metadata_int(tag, metadata, "action_horizon")
        n_action_steps = _require_positive_tag_metadata_int(tag, metadata, "n_action_steps")
        if n_action_steps > action_horizon:
            raise ValueError(
                f"LeRobot tag '{tag}' has n_action_steps={n_action_steps}, which exceeds "
                f"action_horizon={action_horizon}."
            )
        max_action_horizon = max(max_action_horizon, action_horizon)

    return max_action_horizon


def infer_max_action_dim_from_lerobot_metadata(
    robot_mixture: Sequence[KwargsMixture],
    *,
    tag_metadata_by_tag: Dict[str, Dict[str, object]],
) -> int:
    normalized_metadata = _normalize_registered_lerobot_tag_metadata(tag_metadata_by_tag)
    used_tags = _get_used_lerobot_tags(robot_mixture)
    if not used_tags:
        raise ValueError("Cannot infer max_action_dim because the training mixture has no LeRobot tags.")

    max_action_dim = 0
    for tag in sorted(used_tags):
        metadata = normalized_metadata.get(tag)
        if not metadata:
            raise ValueError(f"Missing required LeRobot tag metadata for tag '{tag}'.")
        max_action_dim = max(
            max_action_dim,
            _require_positive_tag_metadata_int(tag, metadata, "action_dim"),
        )

    return max_action_dim


def _validate_lerobot_tag_temporal_metadata(
    robot_mixture: Sequence[KwargsMixture],
    *,
    tag_metadata_by_tag: Dict[str, Dict[str, object]],
    max_action_horizon: int,
) -> None:
    normalized_metadata = _normalize_registered_lerobot_tag_metadata(tag_metadata_by_tag)
    used_tags = _get_used_lerobot_tags(robot_mixture)

    for tag in sorted(used_tags):
        metadata = normalized_metadata.get(tag)
        if not metadata:
            raise ValueError(f"Missing required LeRobot tag metadata for tag '{tag}'.")
        action_horizon = _require_positive_tag_metadata_int(tag, metadata, "action_horizon")
        if action_horizon > int(max_action_horizon):
            raise ValueError(
                f"LeRobot tag '{tag}' has action_horizon={action_horizon}, which exceeds "
                f"--max_action_horizon={int(max_action_horizon)}."
            )


def _validate_packed_action_chunk_padding_args(args) -> None:
    pad_packed_action_chunks = bool(getattr(args, "pad_packed_action_chunks", False))
    packed_action_chunk_cap = getattr(args, "packed_action_chunk_cap", None)
    if not pad_packed_action_chunks:
        if packed_action_chunk_cap is not None:
            raise ValueError(
                "--packed_action_chunk_cap may only be set when --pad_packed_action_chunks=true."
            )
        return
    if not bool(getattr(args, "packing", False)):
        raise ValueError("--pad_packed_action_chunks requires --packing.")
    if not bool(getattr(args, "add_action_expert", False)):
        raise ValueError("--pad_packed_action_chunks requires --add_action_expert=true.")
    if str(getattr(args, "action_format", "")) != "continuous":
        raise ValueError(
            "--pad_packed_action_chunks currently supports --action_format=continuous."
        )
    if str(getattr(args, "state_format", "")) != "discrete":
        raise ValueError("--pad_packed_action_chunks currently requires --state_format=discrete.")
    if packed_action_chunk_cap is None or int(packed_action_chunk_cap) < 1:
        raise ValueError(
            "--pad_packed_action_chunks requires --packed_action_chunk_cap to be set to an integer >= 1."
        )


def _validate_action_training_args(
    action_format: str,
    *,
    max_action_dim: int,
    discrete_action_tokenizer: Optional[str],
) -> None:
    normalized = str(action_format).strip().lower()
    if normalized not in {"continuous", "discrete", "both"}:
        raise ValueError(
            f"Unsupported --action_format={action_format!r}. "
            "Expected one of: continuous, discrete, both."
        )
    if normalized in {"discrete", "both"} and not str(discrete_action_tokenizer or "").strip():
        raise ValueError(
            f"--action_format={normalized} requires --discrete_action_tokenizer."
        )
    if normalized in {"discrete", "both"} and int(max_action_dim) > ACTION_TOKENIZER_MAX_ACTION_DIM:
        raise ValueError(
            "The action tokenizer is only trained with action dim max to 32; "
            f"--action_format={normalized} is not supported when max_action_dim={int(max_action_dim)}. "
            "Use --action_format=continuous."
        )


def _validate_separate_vlm_dataloader_args(args, data_plan: LeRobotTrainingDataPlan) -> None:
    separate_vlm_dataloader = bool(getattr(args, "separate_vlm_dataloader", False))
    vlm_seq_len = getattr(args, "vlm_seq_len", None)
    if not separate_vlm_dataloader:
        if vlm_seq_len is not None:
            raise ValueError("--vlm_seq_len may only be set when --separate_vlm_dataloader=true.")
        return

    if vlm_seq_len is None or int(vlm_seq_len) < 1:
        raise ValueError("--separate_vlm_dataloader=true requires --vlm_seq_len to be set to an integer >= 1.")
    if not data_plan.robot_mixture:
        raise ValueError(
            f"Mixture '{args.mixture}' has no LeRobot robot subset, so it cannot use --separate_vlm_dataloader."
        )
    if not data_plan.vlm_mixture:
        raise ValueError(
            f"Mixture '{args.mixture}' has no non-LeRobot VLM subset, so it cannot use --separate_vlm_dataloader."
        )
    if data_plan.vlm_loader_rate is None:
        raise ValueError(
            f"Mixture '{args.mixture}' does not define a top-level VLM sampling weight, "
            "so it cannot use --separate_vlm_dataloader."
        )


def _build_vlm_data_cfg(
    primary_data_cfg: DataLoaderConfig,
    *,
    vlm_mixture: List[KwargsMixture],
    args,
) -> DataLoaderConfig:
    return replace(
        primary_data_cfg,
        kwargs_mixture=vlm_mixture,
        sequence_length=int(args.vlm_seq_len),
    )


def _sync_vlm_data_cfg_with_primary(conf: TrainConfig) -> TrainConfig:
    if conf.vlm_data is None:
        return conf

    for field_name in (field.name for field in fields(DataLoaderConfig)):
        if field_name in {"kwargs_mixture", "sequence_length"}:
            continue
        setattr(conf.vlm_data, field_name, getattr(conf.data, field_name))

    return conf


def _build_kwargs_mixture_from_raw_mixture(raw_mixture) -> List[KwargsMixture]:
    kwargs_mixture: List[KwargsMixture] = []
    for mixture_name, datasets, rate in raw_mixture:
        submixture = get_training_mixture(datasets)
        kwargs_mixture.append(KwargsMixture(rate, submixture, mixture_name))
    return kwargs_mixture


def get_training_mixture(submixture):
    datasets = []
    for task_name in submixture:
        size, weight = None, None
        if isinstance(task_name, DatasetWithArgs):
            datasets.append(task_name)
            continue
        if isinstance(task_name, tuple):
            if len(task_name) == 3:
                task_name, size, weight = task_name
            else:
                task_name, size = task_name
        datasets.append(DatasetWithArgs(task_name, None, size, weight))
    return datasets


def _validate_raw_mixture_family(raw_mixture, *, expect_lerobot: bool, label: str) -> None:
    for mixture_name, datasets, _rate in raw_mixture:
        for dataset_name in datasets:
            is_lerobot_dataset = str(dataset_name).startswith("lerobot:")
            if is_lerobot_dataset != expect_lerobot:
                expected_family = "LeRobot" if expect_lerobot else "non-LeRobot"
                raise ValueError(
                    f"{label} mixture '{mixture_name}' contains dataset '{dataset_name}', "
                    f"but separate VLM dataloader mode expects only {expected_family} datasets in that subset."
                )


def _build_robot_only_training_data_plan(
    name: str,
    builder,
    robot_style_mixture: Dict[str, float],
) -> LeRobotTrainingDataPlan:
    action_mixture, action_tag_metadata = builder()
    if not action_mixture:
        raise ValueError(
            f"{name} requires its data mixture builder to return at least one action mixture."
        )

    _validate_raw_mixture_family(action_mixture, expect_lerobot=True, label="robot")
    for tag, metadata in action_tag_metadata.items():
        TAG_METADATA_BY_TAG[tag] = metadata

    robot_mixture = _build_kwargs_mixture_from_raw_mixture(action_mixture)
    return LeRobotTrainingDataPlan(
        combined_mixture=robot_mixture,
        robot_mixture=robot_mixture,
        vlm_mixture=None,
        vlm_loader_rate=None,
        robot_style_mixture=robot_style_mixture,
    )


def get_lerobot_training_data_plan(
    name: str,
    *,
    style_robot_action: float = 1.0,
    style_robot_depth: float = 0.0,
    style_robot_depth_action: float = 0.0,
) -> LeRobotTrainingDataPlan:
    reset_tag_metadata()

    robot_style_mixture = {
        "robot_action": float(style_robot_action),
        "robot_depth": float(style_robot_depth),
        "robot_depth_action": float(style_robot_depth_action),
    }
    try:
        builder = MOLMOACT2_LEROBOT_MIXTURES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(MOLMOACT2_LEROBOT_MIXTURES))
        raise NotImplementedError(
            f"Unknown LeRobot mixture '{name}'. Supported mixtures: {supported}."
        ) from exc

    return _build_robot_only_training_data_plan(name, builder, robot_style_mixture)




def _dedupe_tokens_preserve_order(tokens: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for token in tokens:
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _normalize_registered_lerobot_tag_metadata(
    tag_metadata_by_tag: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    normalized: Dict[str, Dict[str, object]] = {}
    for raw_tag, metadata in tag_metadata_by_tag.items():
        if not is_lerobot_tag(raw_tag):
            continue
        bare_tag = strip_lerobot_tag_prefix(raw_tag)
        existing = normalized.get(bare_tag)
        if existing is not None and existing != metadata:
            raise ValueError(f"Conflicting LeRobot tag metadata registered for bare tag '{bare_tag}'.")
        _require_tag_state_keys(bare_tag, metadata)
        normalized[bare_tag] = metadata
    return normalized


def _require_tag_state_keys(tag: str, metadata: Dict[str, object]) -> List[str]:
    state_keys = metadata.get("state_keys")
    if not isinstance(state_keys, list) or not state_keys:
        raise ValueError(
            f"LeRobot tag metadata for tag '{tag}' must define non-empty state_keys."
        )
    normalized = [str(key) for key in state_keys]
    if any(not key for key in normalized):
        raise ValueError(
            f"LeRobot tag metadata for tag '{tag}' must define non-empty state_keys."
        )
    return normalized


def _normalize_style_sampling_rates(raw_rates: Optional[object]) -> Dict[str, float]:
    if not raw_rates:
        raise ValueError("robot_style_mixture must be provided and non-empty.")
    if isinstance(raw_rates, list):
        parsed: Dict[str, float] = {}
        for entry in raw_rates:
            if isinstance(entry, dict):
                parsed.update(entry)
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                parsed[str(entry[0])] = float(entry[1])
            else:
                raise ValueError(
                    "robot_style_mixture list entries must be mappings or [style, rate] pairs."
                )
        raw_rates = parsed
    if not isinstance(raw_rates, dict):
        raise ValueError("robot_style_mixture must be a dict or list of pairs.")
    cleaned: Dict[str, float] = {}
    for key, value in raw_rates.items():
        if str(key) not in ROBOT_OUTPUT_STYLES:
            raise ValueError(
                f"Unsupported LeRobot style '{key}' in robot_style_mixture. "
                f"Expected subset of {sorted(ROBOT_OUTPUT_STYLES)}."
            )
        rate = float(value)
        if rate <= 0:
            continue
        cleaned[str(key)] = rate
    if not cleaned:
        raise ValueError("robot_style_mixture must contain at least one positive sampling rate.")
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("robot_style_mixture must have a positive total weight.")
    return {key: value / total for key, value in cleaned.items()}


def get_lerobot_sft_training_data(name):
    plan = get_lerobot_training_data_plan(name)
    return (plan.robot_mixture or plan.combined_mixture), plan.robot_style_mixture
