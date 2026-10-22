"""Exercise failure cases without calling model APIs or loading model weights."""
from types import SimpleNamespace

import pytest

from turing_eval.backends.api_backend import APIChatBackend
from turing_eval.backends.base import ChatSession, GenerationConfig
from turing_eval.backends.hf_backend import HFChatBackend
import turing_eval.environment as environment
import turing_eval.transcripts as transcripts


@pytest.mark.parametrize("key", ["sk-or-v1-REPLACE_ME", "your-api-key-here", "PLACEHOLDER", "  "])
def test_placeholder_credentials_fail_before_network(monkeypatch, key):
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    with pytest.raises(RuntimeError, match="placeholder"):
        APIChatBackend("openrouter", "model", "model", GenerationConfig())


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0, -1])
def test_invalid_timeout_rejected(monkeypatch, timeout):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with pytest.raises(ValueError, match="timeout"):
        APIChatBackend("openrouter", "model", "model", GenerationConfig(), timeout_seconds=timeout)


def test_reasoning_answer_tag_cannot_become_a_verdict(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    backend = APIChatBackend("openrouter", "model", "model", GenerationConfig())
    monkeypatch.setattr(backend, "_post_chat_completion", lambda _: {
        "choices": [{"message": {"content": None, "reasoning": "<answer>1</answer>",
                    "reasoning_details": [{"text": "<answer>0</answer>"}]}}]})
    with pytest.raises(RuntimeError, match="unparseable"):
        backend.generate([])


def test_non_object_api_response_is_reported(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    backend = APIChatBackend("openrouter", "model", "model", GenerationConfig())
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return b"[]"
    monkeypatch.setattr("turing_eval.backends.api_backend.request.urlopen", lambda *a, **kw: Response())
    with pytest.raises(RuntimeError, match="non-object"):
        backend.generate([])


def test_empty_hf_completion_never_returns_prompt():
    class Output:
        def __getitem__(self, key):
            return "completion"
    output = Output()
    backend = object.__new__(HFChatBackend)
    backend.model_id = "test"
    backend.generation = GenerationConfig()
    backend._prepare_inputs = lambda _: {"input_ids": SimpleNamespace(shape=(1, 3))}
    backend._model = SimpleNamespace(generate=lambda **kwargs: output)
    backend._tokenizer = SimpleNamespace(pad_token_id=0, batch_decode=lambda ids, **kw:
                                        ["" if ids == "completion" else "PROMPT <answer>1</answer>"])
    with pytest.raises(RuntimeError, match="empty"):
        backend.generate([])


def test_metadata_cannot_override_conversation_content():
    backend = SimpleNamespace(generate_with_metadata=lambda _: ("actual", {"role": "user", "content": "wrong"}))
    session = ChatSession(backend, "test")
    assert session.send("hello") == "actual"
    assert session.messages[-1] == {"role": "assistant", "content": "actual"}


def test_environment_uses_code_checkout_from_other_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert environment._git(["rev-parse", "HEAD"])


def test_container_digest_cached_and_invalidated(monkeypatch, tmp_path):
    image = tmp_path / "image.sif"
    image.write_bytes(b"first")
    environment._container_digest.cache_clear()
    monkeypatch.setenv("APPTAINER_CONTAINER", str(image))
    first = environment.capture_environment()["apptainer_container_sha256"]
    assert environment.capture_environment()["apptainer_container_sha256"] == first
    assert environment._container_digest.cache_info().hits == 1
    image.write_bytes(b"changed image")
    assert environment.capture_environment()["apptainer_container_sha256"] != first


def test_unknown_git_state_is_not_reported_clean(monkeypatch):
    monkeypatch.setattr(environment, "_git", lambda _: None)
    assert environment.capture_environment()["git_dirty"] is None


def test_failed_csv_write_preserves_previous_file(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text("previous complete file")
    with pytest.raises(KeyError):
        transcripts.write_csv([{}], path, "actor", "judge", "A")
    assert path.read_text() == "previous complete file"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("kwargs", [{"temperature": float("nan")}, {"temperature": -1},
                                   {"top_p": 2}, {"max_new_tokens": 0}])
def test_invalid_generation_config_fails_before_requests(kwargs):
    with pytest.raises(ValueError):
        GenerationConfig(**kwargs)
