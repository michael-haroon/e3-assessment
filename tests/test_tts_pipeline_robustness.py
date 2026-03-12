import pytest
import torch
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tts.qwen3_tts_pipeline import _normalize_predictor_hidden, _sanitize_prefill_kwargs


def test_sanitize_prefill_kwargs_removes_conflicting_keys():
    kwargs = {
        "max_new_tokens": 123,
        "min_new_tokens": 7,
        "eos_token_id": 9,
        "return_dict_in_generate": False,
        "use_cache": False,
        "temperature": 0.8,
    }

    clean, removed = _sanitize_prefill_kwargs(kwargs)

    assert "temperature" in clean
    assert clean["temperature"] == 0.8
    for key in ("max_new_tokens", "min_new_tokens", "eos_token_id", "return_dict_in_generate", "use_cache"):
        assert key not in clean
        assert key in removed


def test_normalize_predictor_hidden_from_rank1():
    hidden = torch.randn(1024)
    norm = _normalize_predictor_hidden(hidden)
    assert tuple(norm.shape) == (1, 1, 1024)


def test_normalize_predictor_hidden_from_rank2():
    hidden = torch.randn(1, 1024)
    norm = _normalize_predictor_hidden(hidden)
    assert tuple(norm.shape) == (1, 1, 1024)


def test_normalize_predictor_hidden_rank3_passthrough():
    hidden = torch.randn(1, 1, 1024)
    norm = _normalize_predictor_hidden(hidden)
    assert tuple(norm.shape) == (1, 1, 1024)


def test_normalize_predictor_hidden_invalid_rank2_batch():
    with pytest.raises(ValueError):
        _normalize_predictor_hidden(torch.randn(2, 1024))


def test_normalize_predictor_hidden_invalid_rank4():
    with pytest.raises(ValueError):
        _normalize_predictor_hidden(torch.randn(1, 1, 1, 1024))
