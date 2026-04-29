from turing_eval.backends.base import ChatBackend, ChatSession, GenerationConfig


class EchoBackend(ChatBackend):
    def __init__(self, prefix: str = "reply") -> None:
        self.display_name = "echo"
        self.model_id = "echo-model"
        self.prefix = prefix
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        return f"{self.prefix}-{self.calls}-{len(messages)}"


class MetadataBackend(EchoBackend):
    def generate_with_metadata(self, messages):
        reply = self.generate(messages)
        return reply, {"finish_reason": "length", "truncated": True}


def test_generation_config_defaults_to_unlimited():
    cfg = GenerationConfig()
    assert cfg.max_new_tokens is None


def test_generation_config_to_openai_payload_omits_max_tokens_when_unlimited():
    cfg = GenerationConfig()
    assert cfg.to_openai_payload() == {}


def test_generation_config_to_openai_payload_sampling():
    cfg = GenerationConfig(max_new_tokens=123, temperature=0.3, top_p=0.8, top_k=20)
    assert cfg.to_openai_payload() == {
        "max_tokens": 123,
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 20,
    }


def test_generation_config_to_openai_payload_non_sampling_sets_zero_temp():
    cfg = GenerationConfig(max_new_tokens=55, do_sample=False)
    assert cfg.to_openai_payload() == {"max_tokens": 55, "temperature": 0.0}


def test_generation_config_to_hf_kwargs():
    cfg = GenerationConfig(max_new_tokens=77, temperature=0.4, top_p=0.7, top_k=10, do_sample=False)
    assert cfg.to_hf_kwargs() == {
        "max_new_tokens": 77,
        "do_sample": False,
        "return_full_text": True,
        "temperature": 0.4,
        "top_p": 0.7,
        "top_k": 10,
    }


def test_generation_config_to_hf_kwargs_omits_max_new_tokens_when_unlimited():
    cfg = GenerationConfig()
    assert cfg.to_hf_kwargs() == {
        "do_sample": True,
        "return_full_text": True,
    }


def test_chat_session_send_appends_history():
    backend = EchoBackend()
    session = ChatSession(backend, "test", initial_prompt="prompt")
    reply = session.send("hello")
    assert reply == "reply-1-2"
    assert session.messages == [
        {"role": "user", "content": "prompt"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "reply-1-2"},
    ]


def test_chat_session_generate_opening_uses_current_messages():
    backend = EchoBackend("open")
    session = ChatSession(backend, "test", initial_prompt="prompt")
    reply = session.generate_opening()
    assert reply == "open-1-1"
    assert session.messages == [
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "content": "open-1-1"},
    ]


def test_chat_session_preserves_generation_metadata_on_assistant_messages():
    backend = MetadataBackend("meta")
    session = ChatSession(backend, "test", initial_prompt="prompt")
    reply = session.send("hello")
    assert reply == "meta-1-2"
    assert session.messages[-1] == {
        "role": "assistant",
        "content": "meta-1-2",
        "finish_reason": "length",
        "truncated": True,
    }


def test_chat_session_can_append_user_message_without_generating():
    backend = EchoBackend()
    session = ChatSession(backend, "test", initial_prompt="prompt")
    session.add_user_message("context-only")
    assert backend.calls == 0
    assert session.messages == [
        {"role": "user", "content": "prompt"},
        {"role": "user", "content": "context-only"},
    ]


def test_chat_session_legacy_system_prompt_arg_is_user_framed():
    backend = EchoBackend()
    session = ChatSession(backend, "test", system_prompt="legacy")
    assert session.messages == [{"role": "user", "content": "legacy"}]
