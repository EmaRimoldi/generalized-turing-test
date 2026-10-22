"""Optional real-stack smoke test using tiny, locally generated safe weights."""
import os

import pytest

if os.environ.get("GTT_TEST_REAL_HF") != "1":
    pytest.skip("Set GTT_TEST_REAL_HF=1 in the optional HF environment", allow_module_level=True)

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from turing_eval.backends.base import GenerationConfig  # noqa: E402
from turing_eval.backends.hf_backend import HFChatBackend  # noqa: E402


def test_hf_backend_loads_local_safe_weights_and_generates(tmp_path, monkeypatch):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2, "hello": 3, "world": 4}
    tokenizer_core = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer_core.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_core, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>",
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=len(vocab), n_positions=32, n_embd=16, n_layer=1, n_head=2,
        bos_token_id=2, eos_token_id=2, pad_token_id=0,
    ))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.generation_config.suppress_tokens = [0, 1, 2]
    local_model = tmp_path / "tiny-model"
    model.save_pretrained(local_model, safe_serialization=True)
    tokenizer.save_pretrained(local_model)
    backend = HFChatBackend(str(local_model), "local/tiny", GenerationConfig(
        max_new_tokens=2, temperature=0), dtype="float32")
    assert backend.generate([{"role": "user", "content": "hello"}]) == "hello hello"
