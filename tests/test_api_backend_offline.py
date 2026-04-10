"""Unit tests that do not touch the network."""

from turing_eval.backends.api_backend import APIChatBackend, PROVIDERS


def test_providers_registered():
    for name in ("openai", "openrouter", "deepseek", "moonshot", "together", "groq", "mistral"):
        assert name in PROVIDERS


def test_extract_text_string_content():
    msg = {"content": "hello"}
    assert APIChatBackend._extract_text_from_message(msg) == "hello"


def test_extract_text_list_content():
    msg = {"content": [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]}
    assert APIChatBackend._extract_text_from_message(msg) == "hello world"


def test_extract_text_reasoning_fallback():
    msg = {"content": "", "reasoning": "thinking..."}
    assert APIChatBackend._extract_text_from_message(msg) == "thinking..."


def test_extract_text_empty_returns_none():
    msg = {"content": ""}
    assert APIChatBackend._extract_text_from_message(msg) is None


def test_sanitize_messages_replaces_lone_surrogates():
    msg = [{"role": "user", "content": "bad \ud800 text", "finish_reason": "length"}]
    sanitized = APIChatBackend._sanitize_messages(msg)
    assert sanitized[0]["content"] == "bad \ufffd text"
    assert "finish_reason" not in sanitized[0]


def test_generate_with_metadata_surfaces_finish_reason():
    backend = object.__new__(APIChatBackend)
    backend.provider_config = PROVIDERS["openrouter"]
    backend.model_id = "openai/test-model"
    backend._post_chat_completion = lambda messages: {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": "trimmed reply"},
            }
        ]
    }
    reply, metadata = APIChatBackend.generate_with_metadata(backend, [])
    assert reply == "trimmed reply"
    assert metadata == {"finish_reason": "length", "truncated": True}
