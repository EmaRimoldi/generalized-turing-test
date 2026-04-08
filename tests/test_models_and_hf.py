from __future__ import annotations

from types import SimpleNamespace

import pytest

import turing_eval.models as models
from turing_eval.backends.base import GenerationConfig
from turing_eval.backends.hf_backend import HFChatBackend, _resolve_dtype


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


def test_hf_backend_requires_hf_image_when_deps_missing(monkeypatch):
    monkeypatch.setattr(
        "turing_eval.backends.hf_backend._import_hf_dependencies",
        lambda: (_ for _ in ()).throw(RuntimeError("HF dependencies are not installed in the default image.")),
    )
    with pytest.raises(RuntimeError, match="HF dependencies"):
        HFChatBackend("repo/id", "display", GenerationConfig())


def test_resolve_dtype_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported HF dtype"):
        _resolve_dtype(SimpleNamespace(float16="float16"), "nope")


def test_hf_backend_generate_with_mocked_hf_stack(monkeypatch):
    class FakeBatch(dict):
        def to(self, device):
            self["device"] = device
            return self

    class FakeTokenizer:
        pad_token = None
        eos_token = "</s>"
        pad_token_id = 0

        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return "prompt-text"

        def __call__(self, prompt, return_tensors):
            assert prompt == "prompt-text"
            assert return_tensors == "pt"
            return FakeBatch({"input_ids": FakeTensor((1, 3))})

        def batch_decode(self, output_ids, skip_special_tokens):
            self.calls.append((output_ids, skip_special_tokens))
            if output_ids == "generated":
                return ["decoded reply"]
            return ["full decoded reply"]

    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

    class FakeOutput:
        def __getitem__(self, key):
            if isinstance(key, tuple) and key[1].start == 3:
                return "generated"
            raise AssertionError(f"unexpected key {key!r}")

    class FakeModel:
        device = "cuda:0"
        hf_device_map = {"model": "cuda:0"}

        def generate(self, **kwargs):
            assert kwargs["pad_token_id"] == 0
            assert kwargs["max_new_tokens"] == 12
            assert kwargs["do_sample"] is True
            assert kwargs["input_ids"].shape == (1, 3)
            return FakeOutput()

    tokenizer = FakeTokenizer()
    model = FakeModel()

    fake_torch = SimpleNamespace(
        bfloat16="bfloat16",
        float16="float16",
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    fake_auto_tokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
    fake_auto_model = SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)

    def fake_bitsandbytes(**kwargs):
        return kwargs

    monkeypatch.setattr(
        "turing_eval.backends.hf_backend._import_hf_dependencies",
        lambda: (fake_torch, fake_auto_model, fake_auto_tokenizer, fake_bitsandbytes),
    )

    backend = HFChatBackend(
        "Qwen/Qwen2.5-7B-Instruct",
        "qwen/qwen2.5-7b",
        GenerationConfig(max_new_tokens=12),
    )
    reply = backend.generate([{"role": "user", "content": "hello"}])

    assert reply == "decoded reply"
    assert backend.device_summary == "model=cuda:0"
    assert tokenizer.pad_token == tokenizer.eos_token


def test_hf_backend_4bit_requires_cuda(monkeypatch):
    fake_torch = SimpleNamespace(
        bfloat16="bfloat16",
        float16="float16",
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    fake_auto_tokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace(
        pad_token="</s>",
        eos_token="</s>",
        pad_token_id=0,
    ))
    fake_auto_model = SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())

    monkeypatch.setattr(
        "turing_eval.backends.hf_backend._import_hf_dependencies",
        lambda: (fake_torch, fake_auto_model, fake_auto_tokenizer, lambda **kwargs: kwargs),
    )

    with pytest.raises(RuntimeError, match="4bit HF quantization requires CUDA"):
        HFChatBackend(
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/llama-3.3-70b-instruct",
            GenerationConfig(),
            quantization="4bit",
        )
