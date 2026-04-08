from turing_eval.conditions import extract_answer, extract_reasoning


def test_extract_answer_valid_1():
    assert extract_answer("some text <answer>1</answer> more") == 1


def test_extract_answer_valid_0():
    assert extract_answer("<answer>0</answer>") == 0


def test_extract_answer_invalid_value():
    assert extract_answer("<answer>yes</answer>") is None


def test_extract_answer_missing():
    assert extract_answer("no tags here") is None


def test_extract_answer_case_insensitive():
    assert extract_answer("<ANSWER>1</ANSWER>") == 1


def test_extract_reasoning_multiline():
    text = "<reasoning>line1\nline2</reasoning>"
    assert extract_reasoning(text) == "line1\nline2"


def test_extract_reasoning_missing():
    assert extract_reasoning("no tag") is None
