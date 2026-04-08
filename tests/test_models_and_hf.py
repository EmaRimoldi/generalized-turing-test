from __future__ import annotations

import pytest

import turing_eval.models as models
from turing_eval.backends.base import GenerationConfig
from turing_eval.backends.hf_backend import HFChatBackend


def test_load_backend_unknown_model_raises_keyerror():
    with pytest.raises(KeyError):
        models.load_backend("does-not-exist", GenerationConfig())


def test_load_backend_uses_primary_api(monkeypatch):
    captured = {}

    class FakeAPIBackend:
        def __init__(self, provider, model_id, display_name, generation):
            captured.update(
                provider=provider,
                model_id=model_id,
                display_name=display_name,
                generation=generation,
            )

    monkeypatch.setattr(models, "APIChatBackend", FakeAPIBackend)
    generation = GenerationConfig()
    models.load_backend("claude-opus-4.6", generation)
    assert captured["provider"] == "openrouter"
    assert captured["model_id"] == "anthropic/claude-opus-4.6"
    assert captured["display_name"] == "anthropic/claude-opus-4.6"
    assert captured["generation"] is generation


def test_load_backend_uses_api_fallback(monkeypatch):
    captured = {}

    class FakeAPIBackend:
        def __init__(self, provider, model_id, display_name, generation):
            captured.update(provider=provider, model_id=model_id, display_name=display_name)

    monkeypatch.setattr(models, "APIChatBackend", FakeAPIBackend)
    models.load_backend("deepseek-v3.2", GenerationConfig(), force_backend="api")
    assert captured["provider"] == "openrouter"
    assert captured["model_id"] == "deepseek/deepseek-v3.2"


def test_load_backend_uses_hf_backend(monkeypatch):
    captured = {}

    class FakeHFBackend:
        def __init__(self, repo_id, display_name, generation, quantization=None, dtype="bfloat16"):
            captured.update(
                repo_id=repo_id,
                display_name=display_name,
                generation=generation,
                quantization=quantization,
                dtype=dtype,
            )

    monkeypatch.setattr("turing_eval.backends.hf_backend.HFChatBackend", FakeHFBackend)
    generation = GenerationConfig()
    models.load_backend("llama-3.3-70b", generation, force_backend="hf")
    assert captured["repo_id"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert captured["display_name"] == "meta-llama/llama-3.3-70b-instruct"
    assert captured["generation"] is generation
    assert captured["quantization"] == "4bit"


def test_load_backend_missing_hf_route_raises():
    with pytest.raises(ValueError):
        models.load_backend("claude-opus-4.6", GenerationConfig(), force_backend="hf")


def test_hf_backend_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        HFChatBackend("repo/id", "display", GenerationConfig())
