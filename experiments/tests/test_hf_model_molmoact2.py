import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from olmo.hf_model.configuration_molmoact2 import (
    MolmoAct2ActionExpertConfig,
    MolmoAct2AdapterConfig,
    MolmoAct2Config,
    MolmoAct2TextConfig,
    MolmoAct2VitConfig,
)
from olmo.hf_model.convert_molmoact2_to_hf import (
    _save_norm_stats,
    _validate_release_config,
    convert_molmoact2,
)
from olmo.hf_model.modeling_molmoact2 import (
    ActionExpert,
    MolmoAct2ForConditionalGeneration,
    _RobotStats,
    _build_discrete_state_string,
    _build_robot_text,
    _normalize_question_text,
    _select_constrained_discrete_action_token,
)


def _tiny_config(
    enable_depth_reasoning: bool = False,
    add_action_expert: bool = True,
) -> MolmoAct2Config:
    return MolmoAct2Config(
        vit_config=MolmoAct2VitConfig(
            hidden_size=8,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            image_default_input_size=(14, 14),
            image_patch_size=14,
            image_num_pos=2,
        ),
        adapter_config=MolmoAct2AdapterConfig(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            intermediate_size=8,
            text_hidden_size=16,
        ),
        text_config=MolmoAct2TextConfig(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=128,
            additional_vocab_size=None,
            num_hidden_layers=1,
            intermediate_size=16,
        ),
        action_expert_config=MolmoAct2ActionExpertConfig(
            max_action_horizon=3,
            max_action_dim=4,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            ffn_multiple_of=8,
            timestep_embed_dim=8,
        ),
        add_action_expert=add_action_expert,
        max_action_dim=4,
        max_action_horizon=3,
        n_obs_steps=1,
        action_mode="both",
        state_format="discrete",
        state_token_start_id=10,
        num_state_tokens=8,
        action_start_token_id=20,
        action_end_token_id=21,
        action_token_start_id=22,
        num_action_tokens=16,
        enable_depth_reasoning=enable_depth_reasoning,
    )


def _stats_payload():
    return {
        "format": "molmoact2_norm_stats.v1",
        "norm_mode": "q01_q99",
        "metadata_by_tag": {
            "toy": {
                "setup_type": "bench",
                "control_mode": "ee",
                "action_horizon": 3,
                "n_action_steps": 2,
                "state_stats": {
                    "q01": [-1.0, 0.0],
                    "q99": [1.0, 2.0],
                    "min": [-1.0, 0.0],
                    "max": [1.0, 2.0],
                    "mask": [True, True],
                },
                "action_stats": {
                    "q01": [-2.0, 10.0],
                    "q99": [2.0, 20.0],
                    "min": [-2.0, 10.0],
                    "max": [2.0, 20.0],
                    "mask": [True, True],
                },
            }
        },
    }


def test_config_rejects_unsupported_release_modes():
    with pytest.raises(ValueError, match="state_format='discrete'"):
        MolmoAct2Config(state_format="continuous")


def test_hf_config_defaults_match_release_action_token_layout():
    cfg = MolmoAct2Config()

    assert cfg.max_action_horizon == 30
    assert cfg.max_action_dim == 32
    assert cfg.action_expert_config.max_action_horizon == 30
    assert cfg.action_expert_config.max_action_dim == 32
    assert cfg.action_mode == "both"
    assert cfg.state_format == "discrete"
    assert cfg.num_action_tokens == 2048
    assert cfg.num_state_tokens == 256
    assert cfg.add_setup_tokens is True
    assert cfg.add_control_tokens is True


def test_hf_model_has_no_state_token_value_bias_surface():
    cfg = _tiny_config()
    model = MolmoAct2ForConditionalGeneration(cfg)

    assert not hasattr(cfg, "state_token_value_encoding")
    assert not hasattr(cfg, "state_token_value_encoding_scale")
    assert not any(name.endswith("state_token_value_bias") for name, _ in model.named_buffers())
    assert "model.state_token_value_bias" not in model.state_dict()


def test_hf_checkpoint_loader_converts_state_keys_and_layout():
    from olmo.train.checkpoint_loading import (
        _convert_hf_state_dict_to_olmo,
        _hf_source_layout_from_config_dict,
    )

    converted = _convert_hf_state_dict_to_olmo(
        {
            "lm_head.weight": torch.ones(8, 4),
            "model.transformer.blocks.0.self_attn.att_proj.weight": torch.ones(4, 4),
            "model.transformer.blocks.0.mlp.ff_proj.weight": torch.ones(8, 4),
            "model.transformer.rotary_emb.inv_freq": torch.ones(4),
            "model.action_expert.blocks.0.mlp.down_proj.weight": torch.ones(4, 8),
            "model.transformer.wte.embedding": torch.ones(120, 4),
        }
    )

    assert "transformer.ff_out.weight" in converted
    assert "transformer.blocks.0.att_proj.weight" in converted
    assert "transformer.blocks.0.ff_proj.weight" in converted
    assert "transformer.rotary_emb.inv_freq" not in converted
    assert "action_expert.blocks.0.mlp.down_proj.weight" in converted

    layout = _hf_source_layout_from_config_dict(
        {
            "add_setup_tokens": True,
            "add_control_tokens": True,
            "num_state_tokens": 2,
            "state_start_token_id": 104,
            "num_action_tokens": 3,
            "action_output_token_id": 108,
        },
        converted,
    )

    assert layout.base_tokens == 100
    assert layout.added_tokens == 14
    assert layout.padding_tokens == 6
    assert layout.total_tokens == 120


def test_hf_checkpoint_loader_resizes_untied_output_head_rows():
    from olmo.token_layout import TokenLayout
    from olmo.train.checkpoint_loading import _resize_token_matrix_rows_for_layout

    source_layout = TokenLayout(base_tokens=10, added_tokens=2, padding_tokens=2, total_tokens=14)
    target_layout = TokenLayout(base_tokens=10, added_tokens=4, padding_tokens=6, total_tokens=20)
    source = torch.arange(14 * 3, dtype=torch.float32).reshape(14, 3)

    resized = _resize_token_matrix_rows_for_layout(
        source,
        key="transformer.ff_out.weight",
        target_layout=target_layout,
        source_layout=source_layout,
        default_std=0.02,
    )

    assert resized.shape == (20, 3)
    assert torch.equal(resized[:10], source[:10])
    assert torch.equal(resized[10:12], source[10:12])
    assert torch.equal(resized[14:16], source[12:14])


def test_action_expert_checkpoint_loader_synthesizes_cleaned_kv_cache_keys():
    from olmo.nn.action_expert import ActionExpertConfig

    expert = ActionExpertConfig(
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        mlp_ratio=2.0,
        ffn_multiple_of=8,
        timestep_embed_dim=4,
    ).build(llm_dim=8, llm_kv_dim=8, llm_num_layers=2)
    state = {
        "action_expert.action_embed.weight": torch.ones(8, 8),
        "action_expert.final_layer.linear.weight": torch.ones(8, 8),
        "action_expert.context_k_proj.weight": torch.ones(8, 8),
        "action_expert.blocks.0.cross_attn.q_proj.weight": torch.ones(8, 8),
        "action_expert.blocks.1.cross_attn.q_proj.weight": torch.ones(8, 8),
    }

    added = expert.prepare_state_dict_for_loading(state, prefix="action_expert.")

    assert added == 6
    assert torch.equal(state["action_expert.state_encoder.weight"], torch.eye(8))
    assert torch.equal(state["action_expert.state_encoder.bias"], torch.zeros(8))
    for layer_idx in range(2):
        assert state[f"action_expert.blocks.{layer_idx}.cross_attn.kv_proj.weight"].shape == (16, 8)
        assert state[f"action_expert.blocks.{layer_idx}.cross_attn.kv_proj.bias"].shape == (16,)
        assert not state[f"action_expert.blocks.{layer_idx}.cross_attn.kv_proj.weight"].any()
        assert not state[f"action_expert.blocks.{layer_idx}.cross_attn.kv_proj.bias"].any()


def test_hf_predict_action_drops_trivial_attention_mask():
    cfg = _tiny_config()
    model = MolmoAct2ForConditionalGeneration(cfg)
    inputs = {
        "input_ids": torch.ones(1, 3, dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }

    out = model._drop_trivial_attention_mask(inputs)

    assert "attention_mask" not in out
    assert "attention_mask" in inputs


def test_constrained_fast_generation_enforces_exact_decoded_shape():
    class BpeTokenizer:
        expansions = {0: "a", 1: "bb", 2: "c"}

        def decode(self, token_ids):
            return "".join(self.expansions[int(token_id)] for token_id in token_ids)

    action_tokenizer = SimpleNamespace(bpe_tokenizer=BpeTokenizer())
    logits = torch.full((1, 32), -10.0)
    logits[0, 21] = 20.0  # The model prefers ending one coefficient early.
    logits[0, 24] = 10.0
    token_map = {22: 0, 23: 1, 24: 2}

    selected = _select_constrained_discrete_action_token(
        logits,
        [20, 22, 23],
        action_tokenizer=action_tokenizer,
        action_start_token_id=20,
        action_end_token_id=21,
        eos_token_id=2,
        action_token_id_to_bin=token_map,
        target_coefficients=4,
    )
    assert selected.tolist() == [24]

    selected = _select_constrained_discrete_action_token(
        logits,
        [20, 22, 23, 24],
        action_tokenizer=action_tokenizer,
        action_start_token_id=20,
        action_end_token_id=21,
        eos_token_id=2,
        action_token_id_to_bin=token_map,
        target_coefficients=4,
    )
    assert selected.tolist() == [21]

    selected = _select_constrained_discrete_action_token(
        logits,
        [20, 22, 23, 24, 21],
        action_tokenizer=action_tokenizer,
        action_start_token_id=20,
        action_end_token_id=21,
        eos_token_id=2,
        action_token_id_to_bin=token_map,
        target_coefficients=4,
    )
    assert selected.tolist() == [2]


def test_converter_release_validation_rejects_unsupported_modes():
    class Config:
        model_name = "molmoact2"
        add_action_expert = True
        action_expert = None
        state_format = "discrete"

    with pytest.raises(ValueError, match="action_expert config"):
        _validate_release_config(Config())

    _validate_release_config(Config(), add_action_expert=False)


def test_vlm_only_config_omits_action_expert():
    cfg = _tiny_config(add_action_expert=False)
    model = MolmoAct2ForConditionalGeneration(cfg)

    assert cfg.add_action_expert is False
    assert cfg.action_expert_config is None
    assert model.model.action_expert is None
    assert not any(key.startswith("model.action_expert.") for key in model.state_dict())

    with pytest.raises(RuntimeError, match="add_action_expert=False"):
        model.predict_action(
            processor=object(),
            images=[],
            task="task",
            state=[0.0, 0.0],
            norm_tag="toy",
            inference_action_mode="continuous",
        )
    model._molmoact2_robot_stats = _RobotStats(_stats_payload())
    with pytest.raises(TypeError, match="not callable"):
        model.predict_action(
            processor=object(),
            images=[],
            task="task",
            state=[0.0, 0.0],
            norm_tag="toy",
            inference_action_mode="discrete",
            action_tokenizer=object(),
        )


def test_vlm_only_predict_action_supports_discrete_postprocess(monkeypatch):
    cfg = _tiny_config(add_action_expert=False)
    cfg.eos_token_id = cfg.action_end_token_id
    model = MolmoAct2ForConditionalGeneration(cfg)
    model._molmoact2_robot_stats = _RobotStats(_stats_payload())

    class Processor:
        def __call__(self, **kwargs):
            return {
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            }

    class ActionTokenizer:
        def decode(self, token_bins, *, time_horizon, action_dim):
            assert time_horizon == 3
            assert action_dim == 2
            return np.zeros((3, 2), dtype=np.float32)

    def fake_forward(**kwargs):
        return SimpleNamespace(
            logits=torch.zeros(1, 1, cfg.vocab_size),
            past_key_values=None,
        )

    def fake_continue(*args, **kwargs):
        return torch.tensor(
            [
                [
                    cfg.action_start_token_id,
                    cfg.action_token_start_id,
                    cfg.action_end_token_id,
                ]
            ],
            dtype=torch.long,
        )

    model.forward = fake_forward
    monkeypatch.setattr(model, "_continue_discrete_generation_from_output", fake_continue)

    out = model.predict_action(
        processor=Processor(),
        images=[],
        task="task",
        state=[0.0, 0.0],
        norm_tag="toy",
        inference_action_mode="discrete",
        action_tokenizer=ActionTokenizer(),
        n_action_steps=2,
        enable_cuda_graph=False,
    )

    assert out.actions.shape == (1, 2, 2)
    assert out.actions.dtype == torch.float32


def test_vlm_only_conversion_drops_action_expert_weights():
    cfg = _tiny_config(add_action_expert=False)
    state_dict = {
        "transformer.ff_out.weight": torch.empty(1),
        "transformer.wte.embedding": torch.empty(1),
        "transformer.blocks.0.att_proj.weight": torch.empty(1),
        "transformer.blocks.0.att_proj.bias": torch.empty(1),
        "transformer.blocks.0.attn_out.weight": torch.empty(1),
        "transformer.blocks.0.q_norm.weight": torch.empty(1),
        "transformer.blocks.0.k_norm.weight": torch.empty(1),
        "transformer.blocks.0.ff_proj.weight": torch.empty(1),
        "transformer.blocks.0.ff_out.weight": torch.empty(1),
        "action_expert.some_weight": torch.empty(1),
        "action_expert_depth_gate.weight": torch.empty(1),
    }

    converted = convert_molmoact2(state_dict, cfg, weight_tying=False)

    assert "model.action_expert.some_weight" not in converted
    assert "model.action_expert_depth_gate.weight" not in converted
    assert "model.transformer.blocks.0.self_attn.att_proj.weight" in converted


def test_robot_stats_match_robot_processor():
    from olmo.data.robot_processing import RobotProcessorConfig

    payload = _stats_payload()
    hf_stats = _RobotStats(payload)
    olmo_processor = RobotProcessorConfig(
        metadata_by_tag=payload["metadata_by_tag"],
        norm_mode=payload["norm_mode"],
    ).build_processor()

    state = np.asarray([[0.0, 2.0]], dtype=np.float32)
    action = torch.asarray([[[0.0, 0.5]]], dtype=torch.float32)
    np.testing.assert_allclose(
        hf_stats.normalize_state(state, "toy"),
        olmo_processor.normalize_state(state, "toy"),
    )
    torch.testing.assert_close(
        hf_stats.unnormalize_action(action, "toy"),
        olmo_processor.unnormalize_action(action, "toy"),
    )
    torch.testing.assert_close(
        hf_stats.unnormalize_action(action.to(torch.bfloat16), "toy").float(),
        olmo_processor.unnormalize_action(action, "toy"),
    )
    assert hf_stats.get_action_dim("toy") == 2
    assert hf_stats.get_n_action_steps("toy") == 2


def test_robot_stats_downloads_for_repo_id_name_or_path(monkeypatch, tmp_path):
    stats_path = tmp_path / "norm_stats.json"
    stats_path.write_text(json.dumps(_stats_payload()), encoding="utf-8")
    cfg = _tiny_config()
    cfg._name_or_path = "allenai/fake-molmoact2"
    model = MolmoAct2ForConditionalGeneration(cfg)

    def fake_hf_hub_download(repo_id, filename, *, repo_type):
        assert repo_id == "allenai/fake-molmoact2"
        assert filename == "norm_stats.json"
        assert repo_type == "model"
        return str(stats_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    stats = model._get_robot_stats()

    assert stats.get_action_dim("toy") == 2


def test_save_norm_stats_strips_repo_ids(tmp_path):
    model_config = SimpleNamespace(
        robot_processor=SimpleNamespace(
            norm_mode="q01_q99",
            metadata_by_tag={
                "toy": {
                    "repo_ids": ["repo/a"],
                    "setup_type": "bench",
                    "state_stats": {"q01": [0.0], "q99": [1.0], "mask": [True]},
                }
            },
        )
    )

    _save_norm_stats(model_config, str(tmp_path))

    payload = json.loads((tmp_path / "norm_stats.json").read_text())
    assert payload["format"] == "molmoact2_norm_stats.v1"
    assert "repo_to_tag" not in payload
    assert "repo_ids" not in payload["metadata_by_tag"]["toy"]


def test_prompt_text_matches_lerobot_policy_helper():
    sys.path.insert(0, "lerobot/src")
    from lerobot.policies.molmoact2.prompt_utils import build_final_text_input

    state = np.asarray([0.0, 1.0, -1.0], dtype=np.float32)
    discrete_state = _build_discrete_state_string(state, 5)
    ours = _build_robot_text(
        task="pick cup",
        style="robot_depth_action",
        discrete_state_string=discrete_state,
        setup_type="yam",
        control_mode="ee",
        add_setup_tokens=True,
        add_control_tokens=True,
        num_images=1,
    )
    theirs = "<|image|>" + build_final_text_input(
        task="pick cup",
        normalized_states=state,
        style="robot_depth_action",
        setup_type="yam",
        control_mode="ee",
        num_state_tokens=5,
        add_setup_tokens=True,
        add_control_tokens=True,
    )
    assert ours == theirs


def test_hf_language_normalization_matches_lerobot_training():
    assert (
        _normalize_question_text("Instruction: Put the black objects into the drawer and close the drawer.")
        == "put the black objects into the drawer and close the drawer"
    )
    assert _normalize_question_text("Open drawer. Put item inside.") == "open drawer; put item inside"


def test_predict_action_normalizes_language_by_default():
    class RecordingProcessor:
        def __init__(self):
            self.text = None

        def __call__(self, *, text, images, return_tensors):
            self.text = text
            return {
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            }

    model = MolmoAct2ForConditionalGeneration(_tiny_config(enable_depth_reasoning=False))
    model._molmoact2_robot_stats = _RobotStats(_stats_payload())

    def fake_generate_actions_from_inputs(**kwargs):
        return torch.zeros(1, 3, 4)

    model.model.generate_actions_from_inputs = fake_generate_actions_from_inputs
    processor = RecordingProcessor()

    model.predict_action(
        processor=processor,
        images=[],
        task="Instruction: Put the BLACK objects into the drawer and close the drawer.",
        state=[0.0, 0.0],
        norm_tag="toy",
        inference_action_mode="continuous",
    )

    assert "The task is to put the black objects into the drawer and close the drawer." in processor.text


def test_predict_action_can_leave_language_untouched():
    class RecordingProcessor:
        def __init__(self):
            self.text = None

        def __call__(self, *, text, images, return_tensors):
            self.text = text
            return {
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            }

    model = MolmoAct2ForConditionalGeneration(_tiny_config(enable_depth_reasoning=False))
    model._molmoact2_robot_stats = _RobotStats(_stats_payload())

    def fake_generate_actions_from_inputs(**kwargs):
        return torch.zeros(1, 3, 4)

    model.model.generate_actions_from_inputs = fake_generate_actions_from_inputs
    processor = RecordingProcessor()

    model.predict_action(
        processor=processor,
        images=[],
        task="Put the BLACK objects into the drawer and close the drawer.",
        state=[0.0, 0.0],
        norm_tag="toy",
        inference_action_mode="continuous",
        normalize_language=False,
    )

    assert "The task is to Put the BLACK objects into the drawer and close the drawer.." in processor.text


def test_predict_action_uses_tag_horizon_for_generation():
    class RecordingProcessor:
        def __call__(self, *, text, images, return_tensors):
            return {
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            }

    cfg = _tiny_config(enable_depth_reasoning=False)
    model = MolmoAct2ForConditionalGeneration(cfg)
    payload = json.loads(json.dumps(_stats_payload()))
    payload["metadata_by_tag"]["toy"]["action_horizon"] = 2
    payload["metadata_by_tag"]["toy"]["n_action_steps"] = 2
    model._molmoact2_robot_stats = _RobotStats(payload)

    captured = {}

    def fake_generate_actions_from_inputs(**kwargs):
        captured["action_horizon"] = kwargs.get("action_horizon")
        return torch.zeros(1, 2, cfg.max_action_dim)

    model.model.generate_actions_from_inputs = fake_generate_actions_from_inputs

    output = model.predict_action(
        processor=RecordingProcessor(),
        images=[],
        task="task",
        state=[0.0, 0.0],
        norm_tag="toy",
        inference_action_mode="continuous",
    )

    assert captured["action_horizon"] == 2
    assert output.actions.shape == (1, 2, 2)


def test_embedded_action_expert_matches_source_modern_impl():
    from olmo.nn.action_expert import ActionExpert as SourceActionExpert
    from olmo.nn.action_expert import ActionExpertConfig as SourceActionExpertConfig

    torch.manual_seed(0)
    source_cfg = SourceActionExpertConfig(
        max_horizon=3,
        max_action_dim=5,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        ffn_multiple_of=8,
        timestep_embed_dim=8,
        dropout=0.0,
        attn_dropout=0.0,
        context_layer_norm=True,
        qk_norm=True,
        qk_norm_eps=1e-6,
        rope=True,
        causal_attn=False,
    )
    hf_cfg = MolmoAct2ActionExpertConfig(
        max_action_horizon=3,
        max_action_dim=5,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        ffn_multiple_of=8,
        timestep_embed_dim=8,
        dropout=0.0,
        attn_dropout=0.0,
        context_layer_norm=True,
        qk_norm=True,
        qk_norm_eps=1e-6,
        rope=True,
        causal_attn=False,
    )
    source = SourceActionExpert(
        source_cfg,
        llm_dim=32,
        llm_kv_dim=12,
        llm_num_kv_heads=3,
        llm_num_layers=2,
    )
    embedded = ActionExpert(hf_cfg, llm_dim=32, llm_kv_dim=12, llm_num_layers=2)
    embedded_keys = set(embedded.state_dict())
    embedded.load_state_dict(
        {key: value for key, value in source.state_dict().items() if key in embedded_keys},
        strict=True,
    )
    source.eval()
    embedded.eval()

    actions = torch.randn(2, 3, 5)
    timesteps = torch.tensor([0.1, 0.5])
    kv_states = [(torch.randn(2, 4, 12), torch.randn(2, 4, 12)) for _ in range(2)]
    mask = torch.ones(2, 4, dtype=torch.bool)
    with torch.no_grad():
        expected = source(
            actions,
            timesteps,
            encoder_kv_states=kv_states,
            encoder_attention_mask=mask,
        )
        actual = embedded(
            actions,
            timesteps,
            encoder_kv_states=kv_states,
            encoder_attention_mask=mask,
        )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_generate_actions_from_inputs_supports_bf16_action_expert():
    cfg = _tiny_config(enable_depth_reasoning=False)
    model = MolmoAct2ForConditionalGeneration(cfg).model.to(dtype=torch.bfloat16)
    batch_size = 1
    seq_len = 3
    kv_dim = cfg.text_config.num_key_value_heads * cfg.text_config.head_dim
    encoder_kv_states = [
        (
            torch.randn(batch_size, seq_len, kv_dim, dtype=torch.bfloat16),
            torch.randn(batch_size, seq_len, kv_dim, dtype=torch.bfloat16),
        )
    ]

    actions = model.generate_actions_from_inputs(
        input_ids=torch.ones(batch_size, seq_len, dtype=torch.long),
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
        encoder_kv_states=encoder_kv_states,
        encoder_attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
        num_steps=2,
    )

    assert actions.dtype == torch.bfloat16
    assert actions.shape == (batch_size, cfg.max_action_horizon, cfg.max_action_dim)


def test_predict_action_public_validation_errors():
    model = MolmoAct2ForConditionalGeneration(_tiny_config(enable_depth_reasoning=False))
    model._molmoact2_robot_stats = _RobotStats(_stats_payload())

    with pytest.raises(ValueError, match="inference_action_mode.*explicitly"):
        model.predict_action(
            processor=object(),
            images=[],
            task="task",
            state=[0.0, 0.0],
            norm_tag="toy",
        )
    with pytest.raises(TypeError, match="unexpected keyword argument 'action_mode'"):
        model.predict_action(
            processor=object(),
            images=[],
            task="task",
            state=[0.0, 0.0],
            norm_tag="toy",
            action_mode="continuous",
        )
    with pytest.raises(ValueError, match="action_tokenizer"):
        model.predict_action(
            processor=object(),
            images=[],
            task="task",
            state=[0.0, 0.0],
            norm_tag="toy",
            inference_action_mode="discrete",
        )
    with pytest.raises(ValueError, match="--enable_depth_reasoning"):
        model.predict_action(
            processor=object(),
            images=[],
            task="task",
            state=[0.0, 0.0],
            norm_tag="toy",
            inference_action_mode="continuous",
            enable_depth_reasoning=True,
        )
    with pytest.raises(ValueError, match="requires `norm_tag`"):
        model.predict_action(
            processor=object(),
            images=[],
            task="task",
            state=[0.0, 0.0],
            norm_tag="",
            inference_action_mode="continuous",
        )
    with pytest.raises(ValueError, match="Unknown MolmoAct2 normalization tag"):
        model.predict_action(
            processor=object(),
            images=[],
            task="task",
            state=[0.0, 0.0],
            norm_tag="missing",
            inference_action_mode="continuous",
        )
