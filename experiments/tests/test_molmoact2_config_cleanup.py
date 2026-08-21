from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf as om
from omegaconf.errors import OmegaConfBaseException

from olmo.models.model_config import BaseModelConfig
from olmo.models.molmoact2.molmoact2 import MolmoAct2Config
from olmo.nn.action_expert import ActionExpertConfig
from olmo.util import resolve_hf_checkpoint_ref, select_checkpoint


def test_state_token_value_encoding_is_rejected():
    with pytest.raises(OmegaConfBaseException):
        om.merge(
            om.structured(MolmoAct2Config),
            {
                "state_token_value_encoding": "sinusoid",
                "state_token_value_encoding_scale": 0.1,
            },
        )


def test_legacy_molmoact_hf_alias_is_not_rewritten():
    assert resolve_hf_checkpoint_ref("allenai/MolmoAct") == "allenai/MolmoAct"
    assert select_checkpoint("allenai/MolmoAct", quiet=True) == "allenai/MolmoAct"


def test_molmoact2_training_defaults_are_release_defaults():
    from launch_scripts.lerobot_utils.train_plan import DEFAULT_DISCRETE_ACTION_TOKENIZER

    cfg = MolmoAct2Config()

    assert cfg.model_name == "molmoact2"
    assert cfg.add_action_expert is True
    assert cfg.max_action_dim == 32
    assert cfg.action_horizon == 30
    assert cfg.action_expert.max_action_dim == 32
    assert cfg.action_expert.max_horizon == 30
    assert cfg.action_format == "continuous"
    assert cfg.state_format == "discrete"
    assert cfg.action_expert_detach_vlm is False
    assert not hasattr(cfg, "action_expert_layer_mode")
    assert not hasattr(cfg, "action_expert_condition_source")
    assert DEFAULT_DISCRETE_ACTION_TOKENIZER == "allenai/MolmoAct2-FAST-Tokenizer"


@pytest.mark.parametrize("action_format", ["both", "discrete"])
def test_molmoact2_training_allows_discrete_action_formats_up_to_32d(action_format):
    from launch_scripts.lerobot_utils.train_plan import _validate_action_training_args

    _validate_action_training_args(
        action_format,
        max_action_dim=32,
        discrete_action_tokenizer="allenai/MolmoAct2-FAST-Tokenizer",
    )


@pytest.mark.parametrize("action_format", ["both", "discrete"])
def test_molmoact2_training_rejects_discrete_action_formats_above_32d(action_format):
    from launch_scripts.lerobot_utils.train_plan import _validate_action_training_args

    with pytest.raises(ValueError, match="action tokenizer is only trained with action dim max to 32"):
        _validate_action_training_args(
            action_format,
            max_action_dim=33,
            discrete_action_tokenizer="allenai/MolmoAct2-FAST-Tokenizer",
        )


@pytest.mark.parametrize("action_format", ["both", "discrete"])
def test_molmoact2_training_requires_tokenizer_for_discrete_action_formats(action_format):
    from launch_scripts.lerobot_utils.train_plan import _validate_action_training_args

    with pytest.raises(ValueError, match="requires --discrete_action_tokenizer"):
        _validate_action_training_args(
            action_format,
            max_action_dim=14,
            discrete_action_tokenizer=None,
        )


@pytest.mark.parametrize("max_action_dim", [14, 33, 64])
def test_molmoact2_training_allows_continuous_action_format(max_action_dim):
    from launch_scripts.lerobot_utils.train_plan import _validate_action_training_args

    _validate_action_training_args(
        "continuous",
        max_action_dim=max_action_dim,
        discrete_action_tokenizer=None,
    )


def test_action_expert_checkpoint_weights_resize_when_action_dim_exceeds_32():
    from olmo.train.checkpoint_loading import resize_or_init_action_expert_dim_for_checkpoint

    state_dict = {
        "action_expert.action_embed.weight": torch.ones(8, 32),
        "action_expert.final_layer.linear.weight": torch.ones(32, 8),
        "action_expert.final_layer.linear.bias": torch.ones(32),
    }
    model = SimpleNamespace(
        config=SimpleNamespace(
            max_action_dim=64,
            action_expert=SimpleNamespace(max_action_dim=64),
        ),
        action_expert=object(),
    )

    resize_or_init_action_expert_dim_for_checkpoint(state_dict, model)

    assert state_dict["action_expert.action_embed.weight"].shape == (8, 64)
    assert state_dict["action_expert.final_layer.linear.weight"].shape == (64, 8)
    assert state_dict["action_expert.final_layer.linear.bias"].shape == (64,)


def test_legacy_molmoact_model_name_is_rejected(tmp_path):
    cfg = om.structured(MolmoAct2Config)
    cfg.model_name = "molmoact"
    path = tmp_path / "config.yaml"
    path.write_text(om.to_yaml({"model": cfg}))

    with pytest.raises(ValueError, match="Unknown model type molmoact"):
        BaseModelConfig.load(path, key="model", validate_paths=False)


def test_molmoact2_eval_config_hides_discrete_generation_cap():
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    cfg = MolmoAct2Config()

    assert not hasattr(cfg, "discrete_generation_max_steps")
    with pytest.raises(TypeError, match="discrete_generation_max_steps"):
        MolmoAct2Config(discrete_generation_max_steps=256)


def test_molmoact2_eval_config_hides_style_override():
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    cfg = MolmoAct2Config()

    assert not hasattr(cfg, "style")
    with pytest.raises(TypeError, match="style"):
        MolmoAct2Config(style="robot_depth_action")


def test_molmoact2_eval_config_hides_text_input_mode():
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    cfg = MolmoAct2Config()

    assert not hasattr(cfg, "text_input_mode")
    with pytest.raises(TypeError, match="text_input_mode"):
        MolmoAct2Config(text_input_mode="question_answer")


def test_molmoact2_eval_config_hides_cuda_graph_alias():
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    cfg = MolmoAct2Config()

    assert cfg.enable_inference_cuda_graph is True
    assert not hasattr(cfg, "enable_cuda_graph")
    with pytest.raises(TypeError, match="enable_cuda_graph"):
        MolmoAct2Config(enable_cuda_graph=False)


def test_molmoact2_eval_config_hides_trust_remote_code():
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    cfg = MolmoAct2Config()

    assert not hasattr(cfg, "trust_remote_code")
    with pytest.raises(TypeError, match="trust_remote_code"):
        MolmoAct2Config(trust_remote_code=False)


def test_removed_action_expert_mode_flags_are_rejected():
    from launch_scripts.train_lerobot import _reject_removed_action_training_flags

    with pytest.raises(ValueError, match="action_expert_condition_source"):
        _reject_removed_action_training_flags(["--action_expert_condition_source", "hidden_states"])

    with pytest.raises(ValueError, match="action_expert_layer_mode"):
        _reject_removed_action_training_flags(["--action_expert_layer_mode=last"])


def test_hf_get_model_applies_lerobot_multi_image_defaults(monkeypatch):
    import launch_scripts.train_lerobot as train_lerobot
    from olmo.models.molmo2.molmo2_preprocessor import Molmo2PreprocessorConfig
    from olmo.preprocessing.multicrop_preprocessor import MultiCropConfig

    cfg = MolmoAct2Config(
        mm_preprocessor=Molmo2PreprocessorConfig(
            image=MultiCropConfig(max_images=None, max_multi_image_crops=1)
        )
    )
    monkeypatch.setattr(train_lerobot, "get_hf_model_config", lambda *args, **kwargs: cfg)

    model_cfg = train_lerobot.get_model("allenai/MolmoAct2-BimanualYAM", "pre_post_train")

    assert model_cfg is cfg
    assert model_cfg.mm_preprocessor.image.max_images == 5
    assert model_cfg.mm_preprocessor.image.max_multi_image_crops == 8
    assert model_cfg.mm_preprocessor.loss_token_weighting == "root_subsegments_root_tokens"


def test_action_expert_build_uses_single_per_layer_kv_cache_path():
    cfg = ActionExpertConfig(
        max_horizon=2,
        max_action_dim=3,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        timestep_embed_dim=4,
        ffn_multiple_of=8,
    )

    expert = cfg.build(llm_dim=8, llm_kv_dim=4)

    assert expert.use_kv_condition is True
    assert expert.use_kv_flat_condition is True
    assert not hasattr(expert, "layer_mode")
    assert not hasattr(expert, "condition_source")
