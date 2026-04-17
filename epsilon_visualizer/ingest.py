from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .stats import wilson_interval, wilson_lower, wilson_upper
from .thresholding import classify_pair
from .types import Branch, BranchEvidence, PairEvidence, TrialRecord


@dataclass(frozen=True)
class IngestConfig:
    results_root: Path
    epsilon: float
    min_valid_per_branch: int = 3
    z: float = 1.645  # ~95% one-sided
    exclude_answered_in_opening: bool = True
    exclude_any_error: bool = True


def _parse_int(x: str | None) -> int | None:
    if x is None:
        return None
    s = x.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _parse_answer(x: str | None) -> int | None:
    if x is None:
        return None
    s = x.strip()
    if s in {"0", "1"}:
        return int(s)
    return None


def load_trials_from_results_csv(results_csv: Path) -> list[TrialRecord]:
    with results_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    trials: list[TrialRecord] = []
    for row in rows:
        actor_model = (row.get("actor_model") or "").strip()
        distinguisher_model = (row.get("distinguisher_model") or "").strip()
        condition = (row.get("condition") or "").strip()
        error = (row.get("error") or "").strip() or None
        answer = _parse_answer(row.get("answer"))
        success = (row.get("success") or "").strip().lower() == "true"
        turns = _parse_int(row.get("turns"))
        specimen_turns = _parse_int(row.get("specimen_turns"))
        answered_in_opening = error == "distinguisher_answered_in_opening"
        valid = bool(success and answer in (0, 1))
        raw_path = (row.get("transcript_path") or "").strip() or None

        # Branch inference for strict mode:
        # - self-pair B vs B => genuine_target
        # - A vs B (A != B)  => imitation
        branch: Branch | None
        if actor_model and distinguisher_model and actor_model == distinguisher_model:
            branch = "genuine_target"
        elif actor_model and distinguisher_model:
            branch = "imitation"
        else:
            branch = None

        trials.append(
            TrialRecord(
                actor_model=actor_model,
                distinguisher_model=distinguisher_model,
                condition=condition,
                branch=branch,
                answer=answer,
                valid=valid,
                error=error,
                answered_in_opening=answered_in_opening,
                turn_count=turns,
                specimen_turn_count=specimen_turns,
                raw_path=raw_path,
            )
        )

    return trials


def discover_results_csvs(results_root: Path) -> list[Path]:
    return sorted(p for p in results_root.glob("*/results.csv") if p.is_file())


def _should_exclude(trial: TrialRecord, cfg: IngestConfig) -> tuple[bool, str | None]:
    if trial.answer is None:
        return True, "missing_answer"
    if trial.answer not in (0, 1):
        return True, "invalid_answer"
    if cfg.exclude_any_error and trial.error:
        return True, f"error:{trial.error}"
    if cfg.exclude_answered_in_opening and trial.answered_in_opening:
        return True, "answered_in_opening"
    if not trial.valid:
        return True, "not_success"
    return False, None


def _binom_from_trials(trials: list[TrialRecord], *, success_value: int) -> tuple[int, int, dict[str, int], list[str]]:
    """Return (k, n, exclusion_counts, raw_paths) where k counts answer==success_value."""
    k = 0
    n = 0
    exclusions: dict[str, int] = {}
    raw_paths: list[str] = []
    for t in trials:
        if t.raw_path:
            raw_paths.append(t.raw_path)
        if t.answer is None or t.answer not in (0, 1):
            exclusions["missing_answer"] = exclusions.get("missing_answer", 0) + 1
            continue
        if t.error:
            exclusions[f"error:{t.error}"] = exclusions.get(f"error:{t.error}", 0) + 1
            continue
        if not t.valid:
            exclusions["not_success"] = exclusions.get("not_success", 0) + 1
            continue
        n += 1
        if t.answer == success_value:
            k += 1
    return k, n, exclusions, raw_paths


def _branch_evidence(trials: list[TrialRecord], *, success_value: int, cfg: IngestConfig) -> tuple[BranchEvidence, dict[str, int], list[str]]:
    kept: list[TrialRecord] = []
    exclusions: dict[str, int] = {}
    raw_paths: list[str] = []
    for t in trials:
        if t.raw_path:
            raw_paths.append(t.raw_path)
        exclude, reason = _should_exclude(t, cfg)
        if exclude:
            assert reason is not None
            exclusions[reason] = exclusions.get(reason, 0) + 1
            continue
        kept.append(t)
    n = len(kept)
    k = sum(1 for t in kept if t.answer == success_value)
    if n == 0:
        return (
            BranchEvidence(n_valid=0, n_excluded=sum(exclusions.values()), p_hat=None, lower=None, upper=None),
            exclusions,
            raw_paths,
        )
    lo, hi = wilson_interval(k, n, z=cfg.z)
    return (
        BranchEvidence(n_valid=n, n_excluded=sum(exclusions.values()), p_hat=k / n, lower=lo, upper=hi),
        exclusions,
        raw_paths,
    )


def compute_pair_evidence_strict(
    *,
    trials: list[TrialRecord],
    cfg: IngestConfig,
) -> list[PairEvidence]:
    """Compute strict evidence for all ordered pairs A->B using self-pairs as genuine-B branch.

    For target B under each condition:
      - genuine branch uses trials from self-pair (actor=B, distinguisher=B), with success_value=1
      - imitation branch uses trials from pair (actor=A, distinguisher=B), with success_value=0
    """
    # Group trials by (condition, actor_model, distinguisher_model)
    by_run: dict[tuple[str, str, str], list[TrialRecord]] = {}
    models: set[str] = set()
    conditions: set[str] = set()
    for t in trials:
        if not t.actor_model or not t.distinguisher_model or not t.condition:
            continue
        models.add(t.actor_model)
        models.add(t.distinguisher_model)
        conditions.add(t.condition)
        by_run.setdefault((t.condition, t.actor_model, t.distinguisher_model), []).append(t)

    pair_evidence: list[PairEvidence] = []

    for cond in sorted(conditions):
        # selfpairs define genuine branch per target
        genuine_by_target: dict[str, tuple[BranchEvidence, dict[str, int], list[str], float | None, float | None]] = {}
        for m in sorted(models):
            key = (cond, m, m)
            if key not in by_run:
                continue
            ev, exc, raw_paths = _branch_evidence(by_run[key], success_value=1, cfg=cfg)
            genuine_by_target[m] = (ev, exc, raw_paths, ev.lower, ev.upper)

        for a in sorted(models):
            for b in sorted(models):
                if a == "" or b == "":
                    continue
                if a == b:
                    # Self-pairs are calibration diagnostics, not ordering edges.
                    tr = classify_pair(
                        epsilon=cfg.epsilon,
                        estimate=None,
                        upper=None,
                        n_valid=0,
                        min_valid=cfg.min_valid_per_branch,
                    )
                    pair_evidence.append(
                        PairEvidence(
                            source=a,
                            target=b,
                            condition=cond,
                            mode="strict",
                            n_valid=0,
                            n_excluded=0,
                            estimate=None,
                            lower_bound=None,
                            upper_bound=None,
                            threshold_at_point=tr.threshold_at_point,
                            threshold_at_upper=tr.threshold_at_upper,
                            status_at_epsilon=tr.status,
                            p_same_real_b=None,
                            p_diff_impostor=None,
                            p_correct=None,
                            avg_turn_count=None,
                            avg_specimen_turn_count=None,
                            exclusions={"self_pair_calibration_only": 1},
                            raw_paths=[],
                        )
                    )
                    continue
                # imitation branch comes from (cond, a, b)
                key = (cond, a, b)
                if key not in by_run:
                    # missing pair
                    tr = classify_pair(epsilon=cfg.epsilon, estimate=None, upper=None, n_valid=0, min_valid=cfg.min_valid_per_branch)
                    pair_evidence.append(
                        PairEvidence(
                            source=a,
                            target=b,
                            condition=cond,
                            mode="strict",
                            n_valid=0,
                            n_excluded=0,
                            estimate=None,
                            lower_bound=None,
                            upper_bound=None,
                            threshold_at_point=tr.threshold_at_point,
                            threshold_at_upper=tr.threshold_at_upper,
                            status_at_epsilon=tr.status,
                            p_same_real_b=None,
                            p_diff_impostor=None,
                            p_correct=None,
                            avg_turn_count=None,
                            avg_specimen_turn_count=None,
                            exclusions={"missing_run": 1},
                            raw_paths=[],
                        )
                    )
                    continue

                # Genuine branch: from selfpair (b,b)
                genuine = genuine_by_target.get(b)
                # Imitation branch: from (a,b)
                imitation_ev, imitation_exc, imitation_raw = _branch_evidence(by_run[key], success_value=0, cfg=cfg)

                exclusion_counts: dict[str, int] = {}
                for k2, v2 in imitation_exc.items():
                    exclusion_counts[k2] = exclusion_counts.get(k2, 0) + v2
                raw_paths = list(dict.fromkeys((imitation_raw + (genuine[2] if genuine else []))))

                if genuine is None:
                    # Can't compute strict; fail closed (missing)
                    tr = classify_pair(
                        epsilon=cfg.epsilon,
                        estimate=None,
                        upper=None,
                        n_valid=0,
                        min_valid=cfg.min_valid_per_branch,
                    )
                    pair_evidence.append(
                        PairEvidence(
                            source=a,
                            target=b,
                            condition=cond,
                            mode="strict",
                            n_valid=imitation_ev.n_valid,
                            n_excluded=imitation_ev.n_excluded,
                            estimate=None,
                            lower_bound=None,
                            upper_bound=None,
                            threshold_at_point=tr.threshold_at_point,
                            threshold_at_upper=tr.threshold_at_upper,
                            status_at_epsilon=tr.status,
                            p_same_real_b=None,
                            p_diff_impostor=imitation_ev.p_hat,
                            p_correct=None,
                            avg_turn_count=_avg([t.turn_count for t in by_run[key]]),
                            avg_specimen_turn_count=_avg([t.specimen_turn_count for t in by_run[key]]),
                            exclusions={"missing_genuine_selfpair": 1, **exclusion_counts},
                            raw_paths=raw_paths,
                        )
                    )
                    continue

                genuine_ev, genuine_exc, genuine_raw, g_lo, g_hi = genuine
                for k2, v2 in genuine_exc.items():
                    exclusion_counts[k2] = exclusion_counts.get(k2, 0) + v2

                # If either branch has too few valid trials, fail closed (missing)
                n_valid = min(genuine_ev.n_valid, imitation_ev.n_valid)
                if genuine_ev.n_valid < cfg.min_valid_per_branch or imitation_ev.n_valid < cfg.min_valid_per_branch:
                    tr = classify_pair(epsilon=cfg.epsilon, estimate=None, upper=None, n_valid=n_valid, min_valid=cfg.min_valid_per_branch)
                    pair_evidence.append(
                        PairEvidence(
                            source=a,
                            target=b,
                            condition=cond,
                            mode="strict",
                            n_valid=n_valid,
                            n_excluded=genuine_ev.n_excluded + imitation_ev.n_excluded,
                            estimate=None,
                            lower_bound=None,
                            upper_bound=None,
                            threshold_at_point=tr.threshold_at_point,
                            threshold_at_upper=tr.threshold_at_upper,
                            status_at_epsilon=tr.status,
                            p_same_real_b=genuine_ev.p_hat,
                            p_diff_impostor=imitation_ev.p_hat,
                            p_correct=None,
                            avg_turn_count=_avg([t.turn_count for t in by_run[key]]),
                            avg_specimen_turn_count=_avg([t.specimen_turn_count for t in by_run[key]]),
                            exclusions={"insufficient_branch_trials": 1, **exclusion_counts},
                            raw_paths=raw_paths,
                        )
                    )
                    continue

                # p_correct and conservative bound via branch-wise Wilson bounds.
                p_correct = 0.5 * (genuine_ev.p_hat or 0.0) + 0.5 * (imitation_ev.p_hat or 0.0)
                upper = 0.5 * (genuine_ev.upper or 1.0) + 0.5 * (imitation_ev.upper or 1.0)
                lower = 0.5 * (genuine_ev.lower or 0.0) + 0.5 * (imitation_ev.lower or 0.0)

                tr = classify_pair(
                    epsilon=cfg.epsilon,
                    estimate=p_correct,
                    upper=upper,
                    n_valid=n_valid,
                    min_valid=cfg.min_valid_per_branch,
                )

                pair_evidence.append(
                    PairEvidence(
                        source=a,
                        target=b,
                        condition=cond,
                        mode="strict",
                        n_valid=n_valid,
                        n_excluded=genuine_ev.n_excluded + imitation_ev.n_excluded,
                        estimate=p_correct,
                        lower_bound=lower,
                        upper_bound=upper,
                        threshold_at_point=tr.threshold_at_point,
                        threshold_at_upper=tr.threshold_at_upper,
                        status_at_epsilon=tr.status,
                        p_same_real_b=genuine_ev.p_hat,
                        p_diff_impostor=imitation_ev.p_hat,
                        p_correct=p_correct,
                        avg_turn_count=_avg([t.turn_count for t in by_run[key]]),
                        avg_specimen_turn_count=_avg([t.specimen_turn_count for t in by_run[key]]),
                        exclusions=exclusion_counts,
                        raw_paths=raw_paths,
                    )
                )

    return pair_evidence


def compute_pair_evidence_surrogate(
    *,
    trials: list[TrialRecord],
    cfg: IngestConfig,
) -> list[PairEvidence]:
    """Compute an explicitly labeled surrogate evidence object for each ordered pair.

    Surrogate score used here: p_diff_impostor = P[y = 0 | A -> B].
    Lower values mean the judge B more often says "same model" for the impostor,
    i.e. the impostor is harder to detect. We threshold this quantity directly,
    but the UI must label it as a surrogate (imitation-only) view.
    """
    by_run: dict[tuple[str, str, str], list[TrialRecord]] = {}
    models: set[str] = set()
    conditions: set[str] = set()
    for t in trials:
        if not t.actor_model or not t.distinguisher_model or not t.condition:
            continue
        models.add(t.actor_model)
        models.add(t.distinguisher_model)
        conditions.add(t.condition)
        by_run.setdefault((t.condition, t.actor_model, t.distinguisher_model), []).append(t)

    pair_evidence: list[PairEvidence] = []
    for cond in sorted(conditions):
        for a in sorted(models):
            for b in sorted(models):
                if not a or not b:
                    continue
                key = (cond, a, b)
                if key not in by_run:
                    tr = classify_pair(
                        epsilon=cfg.epsilon,
                        estimate=None,
                        upper=None,
                        n_valid=0,
                        min_valid=cfg.min_valid_per_branch,
                    )
                    pair_evidence.append(
                        PairEvidence(
                            source=a,
                            target=b,
                            condition=cond,
                            mode="surrogate",
                            n_valid=0,
                            n_excluded=0,
                            estimate=None,
                            lower_bound=None,
                            upper_bound=None,
                            threshold_at_point=tr.threshold_at_point,
                            threshold_at_upper=tr.threshold_at_upper,
                            status_at_epsilon=tr.status,
                            p_same_real_b=None,
                            p_diff_impostor=None,
                            p_correct=None,
                            avg_turn_count=None,
                            avg_specimen_turn_count=None,
                            exclusions={"missing_run": 1},
                            raw_paths=[],
                        )
                    )
                    continue

                imitation_ev, imitation_exc, imitation_raw = _branch_evidence(by_run[key], success_value=0, cfg=cfg)
                if imitation_ev.n_valid < cfg.min_valid_per_branch or imitation_ev.p_hat is None:
                    tr = classify_pair(
                        epsilon=cfg.epsilon,
                        estimate=None,
                        upper=None,
                        n_valid=imitation_ev.n_valid,
                        min_valid=cfg.min_valid_per_branch,
                    )
                    pair_evidence.append(
                        PairEvidence(
                            source=a,
                            target=b,
                            condition=cond,
                            mode="surrogate",
                            n_valid=imitation_ev.n_valid,
                            n_excluded=imitation_ev.n_excluded,
                            estimate=None,
                            lower_bound=None,
                            upper_bound=None,
                            threshold_at_point=tr.threshold_at_point,
                            threshold_at_upper=tr.threshold_at_upper,
                            status_at_epsilon=tr.status,
                            p_same_real_b=None,
                            p_diff_impostor=imitation_ev.p_hat,
                            p_correct=None,
                            avg_turn_count=_avg([t.turn_count for t in by_run[key]]),
                            avg_specimen_turn_count=_avg([t.specimen_turn_count for t in by_run[key]]),
                            exclusions={"insufficient_trials": 1, **imitation_exc},
                            raw_paths=imitation_raw,
                        )
                    )
                    continue

                tr = classify_pair(
                    epsilon=cfg.epsilon,
                    estimate=imitation_ev.p_hat,
                    upper=imitation_ev.upper,
                    n_valid=imitation_ev.n_valid,
                    min_valid=cfg.min_valid_per_branch,
                )

                pair_evidence.append(
                    PairEvidence(
                        source=a,
                        target=b,
                        condition=cond,
                        mode="surrogate",
                        n_valid=imitation_ev.n_valid,
                        n_excluded=imitation_ev.n_excluded,
                        estimate=imitation_ev.p_hat,
                        lower_bound=imitation_ev.lower,
                        upper_bound=imitation_ev.upper,
                        threshold_at_point=tr.threshold_at_point,
                        threshold_at_upper=tr.threshold_at_upper,
                        status_at_epsilon=tr.status,
                        p_same_real_b=None,
                        p_diff_impostor=imitation_ev.p_hat,
                        p_correct=None,
                        avg_turn_count=_avg([t.turn_count for t in by_run[key]]),
                        avg_specimen_turn_count=_avg([t.specimen_turn_count for t in by_run[key]]),
                        exclusions=imitation_exc,
                        raw_paths=imitation_raw,
                    )
                )

    return pair_evidence


def _avg(values: list[int | None]) -> float | None:
    xs = [v for v in values if isinstance(v, int)]
    if not xs:
        return None
    return sum(xs) / len(xs)


def build_visualizer_artifact(
    *,
    results_root: Path,
    epsilon: float,
    min_valid_per_branch: int = 3,
    z: float = 1.645,
    exclude_answered_in_opening: bool = True,
    exclude_any_error: bool = True,
) -> dict:
    cfg = IngestConfig(
        results_root=results_root,
        epsilon=epsilon,
        min_valid_per_branch=min_valid_per_branch,
        z=z,
        exclude_answered_in_opening=exclude_answered_in_opening,
        exclude_any_error=exclude_any_error,
    )

    csvs = discover_results_csvs(results_root)
    all_trials: list[TrialRecord] = []
    for p in csvs:
        all_trials.extend(load_trials_from_results_csv(p))

    pair_evidence = compute_pair_evidence_strict(trials=all_trials, cfg=cfg)
    surrogate_evidence = compute_pair_evidence_surrogate(trials=all_trials, cfg=cfg)
    models = sorted({t.actor_model for t in all_trials if t.actor_model} | {t.distinguisher_model for t in all_trials if t.distinguisher_model})
    conditions = sorted({t.condition for t in all_trials if t.condition})

    # Serialize as JSON-friendly dict
    return {
        "mode": "strict",
        "availableModes": ["strict", "surrogate"],
        "epsilonDefault": epsilon,
        "minValidPerBranch": min_valid_per_branch,
        "confidenceMethod": {"kind": "wilson", "z": z},
        "filtersDefault": {
            "excludeAnsweredInOpening": exclude_answered_in_opening,
            "excludeAnyError": exclude_any_error,
        },
        "models": models,
        "conditions": conditions,
        "pairs": [
            _serialize_pair_row(ev)
            for ev in pair_evidence
        ]
        + [
            _serialize_pair_row(ev)
            for ev in surrogate_evidence
        ],
        "formalismRef": "Turing_Ordering_Visualizer_Formalization_English.md",
    }


def _serialize_pair_row(ev: PairEvidence) -> dict:
    d_hat, eps_star = _d_hat_epsilon_star(ev.estimate)
    return {
        "source": ev.source,
        "target": ev.target,
        "condition": ev.condition,
        "mode": ev.mode,
        "nValid": ev.n_valid,
        "nExcluded": ev.n_excluded,
        "estimate": ev.estimate,
        "lowerBound": ev.lower_bound,
        "upperBound": ev.upper_bound,
        "dHat": d_hat,
        "epsilonStarHat": eps_star,
        "thresholdAtPoint": ev.threshold_at_point,
        "thresholdAtUpper": ev.threshold_at_upper,
        "statusAtEpsilon": ev.status_at_epsilon,
        "details": {
            "pSameRealB": ev.p_same_real_b,
            "pDiffImpostor": ev.p_diff_impostor,
            "pCorrect": ev.p_correct,
            "averageTurnCount": ev.avg_turn_count,
            "averageSpecimenTurnCount": ev.avg_specimen_turn_count,
            "exclusionCounts": ev.exclusions,
            "rawPaths": ev.raw_paths[:100],
        },
    }


def _d_hat_epsilon_star(estimate: float | None) -> tuple[float | None, float | None]:
    """Markdown: d_hat = P_hat(correct) - 1/2; epsilon_star_hat = max(0, d_hat)."""
    if estimate is None:
        return None, None
    d = float(estimate) - 0.5
    return d, max(0.0, d)


def write_visualizer_artifact_json(artifact: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

