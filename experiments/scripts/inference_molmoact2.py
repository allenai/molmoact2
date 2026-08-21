"""
MolmoAct2 standalone inference.

Takes user-specified images, text, and states, runs the MolmoAct2 checkpoint,
normalizes inputs using the checkpoint's robot processor (if available),
and prints the predicted action chunk.

Example:
python scripts/inference_molmoact2.py \
  --checkpoint /path/to/molmoact2/checkpoint \
  --images /path/to/image1.png /path/to/image2.png \
  --prompt "pick up the black bowl between the plate and the ramekin and place it on the plate" \
  --state_file /path/to/states.json \
  --norm_tag libero \
  --num_steps 10
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from dataclasses import replace
import numpy as np
import torch
from PIL import Image

from olmo.torch_util import move_to_device
from olmo.train.checkpointer import load_model_state
from olmo.models.model_config import BaseModelConfig
from olmo.util import prepare_cli_environment, resource_path

log = logging.getLogger(__name__)


def _load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def _parse_state(args) -> np.ndarray:
    if args.state_file:
        with open(args.state_file, "r") as f:
            data = json.load(f)
        arr = np.array(data, dtype=np.float32)
    else:
        if not args.state:
            raise ValueError("Provide --state values or --state_file")
        arr = np.array(args.state, dtype=np.float32)

    n_obs = args.n_obs_steps
    if arr.ndim == 1:
        if arr.size % n_obs != 0:
            raise ValueError(f"State length {arr.size} is not divisible by n_obs_steps={n_obs}")
        arr = arr.reshape(n_obs, -1)
    elif arr.ndim == 2:
        if arr.shape[0] != n_obs:
            raise ValueError(f"State has {arr.shape[0]} obs steps, expected {n_obs}")
    else:
        raise ValueError(f"Unsupported state shape {arr.shape}")
    return arr


def _load_config_and_model(checkpoint_dir: str, device: torch.device):
    config_path = resource_path(checkpoint_dir, "config.yaml")
    model_cfg = BaseModelConfig.load(config_path, key="model")
    model_cfg = _force_resize_crop_mode(model_cfg)

    log.info("Instantiating model from %s", checkpoint_dir)
    with torch.device("meta"):
        model = model_cfg.build_model()
    model.to_empty(device=device)
    load_model_state(checkpoint_dir, model)
    model.to(device)
    model.eval()
    return model_cfg, model, model_cfg


def _force_resize_crop_mode(model_cfg):
    mm_preprocessor = getattr(model_cfg, "mm_preprocessor", None)
    image_cfg = getattr(mm_preprocessor, "image", None) if mm_preprocessor is not None else None
    if image_cfg is None:
        log.warning("No image preprocessor found; cannot force crop_mode=resize.")
        return model_cfg
    if getattr(image_cfg, "crop_mode", None) == "resize":
        return model_cfg
    image_cfg = replace(image_cfg, crop_mode="resize")
    mm_preprocessor = replace(mm_preprocessor, image=image_cfg)
    log.info("Forcing crop_mode=resize for inference.")
    return replace(model_cfg, mm_preprocessor=mm_preprocessor)


def _build_processors(model_cfg, seq_len: int):
    preprocessor = model_cfg.build_preprocessor(
        for_inference=True,
        is_training=False,
        max_seq_len=seq_len,
    )
    collator = model_cfg.build_collator(
        preprocessor.get_output_shapes(),
        pad_mode=None,
        include_metadata=True,
    )
    return preprocessor, collator


def _build_example(images: List[np.ndarray], prompt: str, state: np.ndarray) -> Dict[str, Any]:
    example: Dict[str, Any] = {
        "style": "demo",
        "question": prompt,
        "state": state,
    }
    example["image"] = images if len(images) > 1 else images[0]
    return example


def _maybe_normalize_state(model_cfg, state: np.ndarray, repo_id: Optional[str]) -> np.ndarray:
    proc_cfg = getattr(model_cfg, "robot_processor", None)
    if proc_cfg is None:
        return state
    pre = proc_cfg.build_preprocessor()
    return pre.normalize_state(state, repo_id)


def _maybe_unnormalize_action(model_cfg, actions: np.ndarray, repo_id: Optional[str]) -> np.ndarray:
    proc_cfg = getattr(model_cfg, "robot_processor", None)
    if proc_cfg is None:
        return actions
    post = proc_cfg.build_postprocessor()
    return post.unnormalize_action(actions, repo_id)


def main():
    parser = argparse.ArgumentParser(description="MolmoAct2 standalone inference.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory (unsharded preferred).")
    parser.add_argument("--images", nargs="+", required=True, help="Paths to one or more images.")
    parser.add_argument("--prompt", default="", help="Text prompt / instruction.")
    parser.add_argument("--state", nargs="*", type=float, help="Flat state values (reshape using n_obs_steps).")
    parser.add_argument("--state_file", type=str, help="JSON file containing state array.")
    parser.add_argument("--norm_tag", type=str, default=None, help="Normalization tag to use for robot stats.")
    parser.add_argument("--seq_len", type=int, default=None, help="Optional max sequence length override.")
    parser.add_argument("--num_steps", type=int, default=None, help="Flow-matching integration steps.")
    parser.add_argument("--device", default="cuda", help="Device to run on.")
    parser.add_argument("--n_obs_steps", type=int, default=None, help="Override obs steps for state reshaping.")
    args = parser.parse_args()

    prepare_cli_environment()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    log.info("Loading checkpoint from %s", args.checkpoint)
    _, model, model_cfg = _load_config_and_model(args.checkpoint, device)

    # Defaults from checkpoint unless user overrides
    n_obs_steps = args.n_obs_steps or getattr(model_cfg, "n_obs_steps", 1)
    seq_len = args.seq_len or model_cfg.llm.max_sequence_length
    num_steps = args.num_steps or getattr(model_cfg, "flow_matching_num_steps", 10)

    preprocessor, collator = _build_processors(model_cfg, seq_len)

    images = [_load_image(Path(p)) for p in args.images]
    state = _parse_state(argparse.Namespace(state=args.state, state_file=args.state_file, n_obs_steps=n_obs_steps))
    state = _maybe_normalize_state(model_cfg, state, args.norm_tag)

    example = _build_example(images, args.prompt, state)
    proc = preprocessor(example)
    batch = collator([proc])
    batch = move_to_device(batch, device)

    # Collect model inputs from the collated batch.
    model_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch.get("attention_mask"),
        "position_ids": batch.get("position_ids"),
        "response_mask": batch.get("response_mask"),
        "images": batch.get("images"),
        "image_masks": batch.get("image_masks"),
        "token_pooling": batch.get("token_pooling"),
        "low_res_token_pooling": batch.get("low_res_token_pooling"),
        "states": batch.get("states"),
    }
    model_inputs = {k: v for k, v in model_inputs.items() if v is not None}

    with torch.no_grad():
        actions = model.generate_actions(
            **model_inputs,
            num_steps=num_steps,
        )

    actions_np = actions.detach().cpu().numpy()
    actions_np = _maybe_unnormalize_action(model_cfg, actions_np, args.norm_tag)

    print("Generated action chunk shape:", actions_np.shape)
    print("Action chunk (first example):", actions_np[0])


if __name__ == "__main__":
    main()
