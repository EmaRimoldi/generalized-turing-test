"""Unit tests that do not touch the network."""

from turing_eval.backends.api_backend import APIChatBackend, PROVIDERS


def test_providers_registered():
    for name in ("openrouter", "deepseek", "moonshot", "together", "groq", "mistral"):
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
    msg = [{"role": "user", "content": "bad \ud800 text"}]
    sanitized = APIChatBackend._sanitize_messages(msg)
    assert sanitized[0]["content"] == "bad \ufffd text"
