"""Check the released launch schedules against Appendix B.1 of the paper."""
from pathlib import Path

import pytest

from run_experiment_batch import load_experiments


PAPER_MODELS = {
    "claude-opus-4.6", "claude-sonnet-4.6", "gpt-5.4", "gemini-3-pro",
    "deepseek-v3.2", "mistral-large-2512", "ministral-8b-2512", "qwen3-32b",
    "grok-4.20",
}
FRONTIER = {"claude-opus-4.6", "gpt-5.4", "gemini-3-pro"}
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "schedule,condition,frontier_trials,other_trials,total,specimen_cap",
    [
        ("gtt", "no_query_no_explanation", 40, 25, 2160, None),
        ("gttq", "yes_query_no_explanation", 25, 10, 945, 20),
    ],
)
def test_paper_schedule_is_loadable_and_covers_the_reported_grid(
    schedule, condition, frontier_trials, other_trials, total, specimen_cap
):
    runs = load_experiments(ROOT / "configs" / f"paper_{schedule}.csv")
    assert len(runs) == 81
    assert len({run.run_name for run in runs}) == 81
    assert {(r.actor, r.distinguisher) for r in runs} == {
        (actor, judge) for actor in PAPER_MODELS for judge in PAPER_MODELS
    }
    assert sum(run.num_trials for run in runs) == total
    for run in runs:
        expected = frontier_trials if {run.actor, run.distinguisher} <= FRONTIER else other_trials
        assert run.num_trials == expected
        assert run.condition == condition
        assert run.turn_limit == 40
        assert run.specimen_turn_limit == specimen_cap
