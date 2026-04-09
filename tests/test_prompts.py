from turing_eval.prompts import (
    build_actor_prompt_no_query,
    build_actor_prompt_yes_query,
    build_distinguisher_prompt,
)


def test_actor_no_query_fills_placeholders():
    prompt = build_actor_prompt_no_query("anthropic/claude-opus-4.6", "Hello there")
    assert "anthropic/claude-opus-4.6" in prompt
    assert "Hello there" in prompt
    assert "{distinguisher_model}" not in prompt
    assert "{first_message}" not in prompt


def test_actor_no_query_preserves_quotes_verbatim():
    prompt = build_actor_prompt_no_query("model", 'say "hi"')
    assert '"hi"' in prompt
    assert '\\"hi\\"' not in prompt


def test_actor_yes_query_has_specimen_language():
    prompt = build_actor_prompt_yes_query("openai/gpt-5.4")
    assert "specimen" in prompt
    assert "STOP" in prompt
    assert "openai/gpt-5.4" in prompt


def test_distinguisher_with_reasoning_has_reasoning_tag():
    prompt = build_distinguisher_prompt(with_reasoning=True)
    assert "<reasoning>" in prompt


def test_distinguisher_without_reasoning_has_no_reasoning_tag():
    prompt = build_distinguisher_prompt(with_reasoning=False)
    assert "<reasoning>" not in prompt
