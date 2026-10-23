from turing_eval.prompts import (
    build_distinguisher_prompt_50_50_trusted_distinguisher,
    build_distinguisher_prompt_yes_query_trusted_distinguisher_stage1,
    build_distinguisher_prompt_yes_query_trusted_distinguisher_stage2,
    build_actor_prompt_no_query,
    build_actor_prompt_no_query_controlled_turns,
    build_actor_prompt_no_query_trusted_distinguisher,
    build_actor_prompt_yes_query,
    build_actor_prompt_yes_query_controlled_queries_final,
    build_actor_prompt_yes_query_controlled_queries_initial,
    build_actor_prompt_yes_query_controlled_queries_intermediate,
    build_distinguisher_prompt,
    build_distinguisher_prompt_controlled_turns_final,
    build_distinguisher_prompt_controlled_turns_initial,
    build_distinguisher_prompt_controlled_turns_intermediate,
    build_distinguisher_prompt_no_query_trusted_distinguisher,
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
    prompt = build_actor_prompt_yes_query("openai/gpt-5.4-pro")
    assert "specimen" in prompt
    assert "STOP" in prompt
    assert "openai/gpt-5.4-pro" in prompt
    assert "conversational style" in prompt
    assert "reasoning patterns" in prompt


def test_distinguisher_with_reasoning_has_reasoning_tag():
    prompt = build_distinguisher_prompt(with_reasoning=True)
    assert "<reasoning>" in prompt
    assert "Do NOT output an answer in your first message." in prompt


def test_distinguisher_without_reasoning_has_no_reasoning_tag():
    prompt = build_distinguisher_prompt(with_reasoning=False)
    assert "<reasoning>" not in prompt
    assert "Do NOT output an answer in your first message." in prompt


def test_actor_no_query_trusted_distinguisher_fills_placeholders():
    prompt = build_actor_prompt_no_query_trusted_distinguisher("anthropic/claude-opus-4.6", "Hello there")
    assert "anthropic/claude-opus-4.6" in prompt
    assert "Hello there" in prompt
    assert "{target_model}" not in prompt
    assert "{first_message}" not in prompt


def test_distinguisher_no_query_trusted_distinguisher_mentions_target_model():
    prompt = build_distinguisher_prompt_no_query_trusted_distinguisher("openai/gpt-5.4-pro")
    assert "openai/gpt-5.4-pro" in prompt
    assert "same model as yourself" not in prompt
    assert "Do NOT output an answer in your first message." in prompt


def test_actor_yes_query_controlled_queries_prompts_fill_placeholders():
    initial = build_actor_prompt_yes_query_controlled_queries_initial("openai/gpt-5.4-pro", 3)
    intermediate = build_actor_prompt_yes_query_controlled_queries_intermediate("specimen reply", 2, 3)
    final = build_actor_prompt_yes_query_controlled_queries_final("specimen reply", 3)
    assert "openai/gpt-5.4-pro" in initial
    assert "exactly 3 queries" in initial
    assert "query 2 of 3" in intermediate
    assert "specimen reply" in intermediate
    assert "used all 3 queries" in final
    assert "The specimen stage is now over." in final


def test_distinguisher_controlled_turn_prompts_fill_placeholders():
    actor = build_actor_prompt_no_query_controlled_turns("anthropic/claude-opus-4.6", "hello")
    initial = build_distinguisher_prompt_controlled_turns_initial(4)
    intermediate = build_distinguisher_prompt_controlled_turns_intermediate("agent reply", 3, 4)
    final = build_distinguisher_prompt_controlled_turns_final("agent reply")
    assert "anthropic/claude-opus-4.6" in actor
    assert 'The first message from the distinguisher is: "hello."' in actor
    assert "exactly 4 rounds of interaction" in initial
    assert "round 3 of 4" in intermediate
    assert "Do not provide your final judgment yet." in intermediate
    assert "This was the last round of interaction" in final
    assert "<answer>" in final


def test_distinguisher_50_50_trusted_distinguisher_mentions_prior_and_target_model():
    prompt = build_distinguisher_prompt_50_50_trusted_distinguisher("openai/gpt-5.4-pro")
    assert "openai/gpt-5.4-pro" in prompt
    assert "With 50% probability" in prompt
    assert "different model instructed to imitate" in prompt
    assert "Do NOT output an answer in your first message." in prompt



def test_distinguisher_yes_query_trusted_distinguisher_stage1_mentions_specimen_and_stop():
    prompt = build_distinguisher_prompt_yes_query_trusted_distinguisher_stage1("openai/gpt-5.4-pro")
    assert "openai/gpt-5.4-pro" in prompt
    assert "specimen" in prompt
    assert "reasoning patterns" in prompt
    assert 'return "STOP"' in prompt



def test_distinguisher_yes_query_trusted_distinguisher_stage2_mentions_target_model_and_answer_tag():
    prompt = build_distinguisher_prompt_yes_query_trusted_distinguisher_stage2("openai/gpt-5.4-pro")
    assert "openai/gpt-5.4-pro" in prompt
    assert "<answer></answer>" in prompt
    assert "Please send your first message to the agent" in prompt
