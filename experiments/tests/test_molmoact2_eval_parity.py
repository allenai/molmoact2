from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.molmoact2.modeling_molmoact2 import (
    MolmoAct2InferenceResult,
    MolmoAct2Policy,
    _disable_inference_token_bias,
)
from lerobot.policies.molmoact2 import hf_backend
from lerobot.policies.molmoact2.hf_backend import (
    MolmoAct2HFBackend,
    _load_discrete_action_processor_from_path,
)


def test_hf_policy_preserves_native_observation_image_order():
    first = np.full((2, 2, 3), 1, dtype=np.uint8)
    second = np.full((2, 2, 3), 2, dtype=np.uint8)
    obs = {
        "observation.images.image2": first,
        "observation.images.image": second,
    }

    images = MolmoAct2HFBackend._extract_images(obs)

    assert [int(image[0, 0, 0]) for image in images] == [1, 2]


def test_disable_inference_token_bias_removes_legacy_nonpersistent_bias():
    token_bias = object()
    model = SimpleNamespace(transformer=SimpleNamespace(token_bias=token_bias))

    assert _disable_inference_token_bias(model)
    assert model.transformer.token_bias is None
    assert not _disable_inference_token_bias(model)


def test_native_chunk_api_returns_environment_space_actions_once():
    class RobotProcessor:
        def __init__(self):
            self.calls = 0

        def unnormalize_action(self, actions, repo_id):
            assert repo_id == "libero"
            self.calls += 1
            return actions + 10.0

    robot_processor = RobotProcessor()
    handles = SimpleNamespace(
        n_obs_steps=1,
        norm_tag="libero",
        robot_processor=robot_processor,
    )
    normalized = torch.tensor([[[0.0, 1.0, 99.0], [2.0, 3.0, 99.0]]])
    policy = SimpleNamespace(
        _hf_backend=None,
        _handles=handles,
        _obs_to_example=lambda observation: observation,
        _combine_history_examples=lambda examples, handles, norm_tag: {},
        _resolve_style_for_inference=lambda handles: "robot_action",
        _collate_example=lambda example, handles: {},
        _call_generate_inference_result=lambda *args, **kwargs: (
            MolmoAct2InferenceResult(style="robot_action", actions=normalized),
            0.01,
        ),
        _maybe_log_generated_text=lambda *args, **kwargs: None,
        _resolve_n_action_steps_for_tag=lambda *args, **kwargs: 2,
        _resolve_action_dim_for_tag=lambda *args, **kwargs: 2,
    )

    result = MolmoAct2Policy.generate_inference_result_from_observations(
        policy,
        [{}],
        norm_tag="libero",
    )

    torch.testing.assert_close(
        result.actions,
        torch.tensor([[[10.0, 11.0], [12.0, 13.0]]]),
    )
    assert robot_processor.calls == 1


def test_hf_backend_loads_custom_action_processor_without_model_config(tmp_path, monkeypatch):
    (tmp_path / "processor_config.json").write_text(
        '{"auto_map":{"AutoProcessor":"processing_action_tokenizer.UniversalActionProcessor"},'
        '"processor_class":"UniversalActionProcessor","scale":10,"vocab_size":2048}',
        encoding="utf-8",
    )
    (tmp_path / "processing_action_tokenizer.py").write_text(
        "class UniversalActionProcessor:\n"
        "    def __init__(self, tokenizer, **kwargs):\n"
        "        self.tokenizer = tokenizer\n"
        "        self.bpe_tokenizer = tokenizer\n"
        "        self.scale = kwargs['scale']\n"
        "        self.vocab_size = kwargs['vocab_size']\n"
        "        self.kwargs = kwargs\n",
        encoding="utf-8",
    )
    tokenizer = object()
    calls = []

    def fake_from_pretrained(path, **kwargs):
        calls.append((path, kwargs))
        return tokenizer

    monkeypatch.setattr(
        hf_backend.PreTrainedTokenizerFast,
        "from_pretrained",
        fake_from_pretrained,
    )

    processor = _load_discrete_action_processor_from_path(tmp_path)

    assert processor.tokenizer is tokenizer
    assert processor.kwargs == {"scale": 10, "vocab_size": 2048}
    assert calls == [(str(tmp_path), {"local_files_only": True, "token": None})]


def test_inference_normalization_errors_fail_closed():
    from scripts.inference_molmoact2 import (
        _maybe_normalize_state,
        _maybe_unnormalize_action,
    )

    class FailingProcessor:
        @staticmethod
        def normalize_state(*args, **kwargs):
            raise ValueError("missing state statistics")

        @staticmethod
        def unnormalize_action(*args, **kwargs):
            raise ValueError("missing action statistics")

    processor = FailingProcessor()
    config = SimpleNamespace(
        robot_processor=SimpleNamespace(
            build_preprocessor=lambda: processor,
            build_postprocessor=lambda: processor,
        )
    )

    with pytest.raises(ValueError, match="missing state statistics"):
        _maybe_normalize_state(config, np.zeros(2, dtype=np.float32), "missing")
    with pytest.raises(ValueError, match="missing action statistics"):
        _maybe_unnormalize_action(config, np.zeros((1, 2), dtype=np.float32), "missing")
