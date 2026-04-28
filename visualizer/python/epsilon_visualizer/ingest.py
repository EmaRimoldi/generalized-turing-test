from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from .thresholding import classify_pair
from .types import Branch, BranchEvidence, PairEvidence, TrialRecord


def _normalize_model_slug(model_id: str) -> str:
    """Align matrix slugs with grouped JSON (e.g. Grok provider id vs short label)."""
    s = (model_id or "").strip()
    if s in {"x-ai/grok-2", "x-ai/grok-4.20"}:
        return "grok-4.20"
    return s


def _branch_trials_provenance(trials: list[TrialRecord]) -> str | None:
    """Single label for aggregate branch trials (all trials in a cell share provenance from grouped ingest)."""
    ps = {t.data_provenance for t in trials if getattr(t, "data_provenance", None)}
    if not ps:
        return None
    if "imputed_uniform" in ps:
        return "imputed_uniform"
    if "matrix_csv" in ps:
        return "matrix_csv"
    if "grouped_json" in ps:
        return "grouped_json"
    return next(iter(ps))


def _expand_aggregate_row_to_trials(
    actor: str,
    distinguisher: str,
    *,
    valid_trials: int,
    answer_1_count: int,
    answer_0_count: int,
    condition: str,
    turn_count: int | None = None,
    data_provenance: str | None = None,
) -> list[TrialRecord]:
    trials: list[TrialRecord] = []
    branch: Branch = "genuine_target" if actor == distinguisher else "imitation"
    for _ in range(answer_1_count):
        trials.append(
            TrialRecord(
                actor_model=actor,
                distinguisher_model=distinguisher,
                condition=condition,
                branch=branch,
                answer=1,
                valid=True,
                error=None,
                answered_in_opening=False,
                turn_count=turn_count,
                specimen_turn_count=None,
                raw_path=None,
                data_provenance=data_provenance,
            )
        )
    for _ in range(answer_0_count):
        trials.append(
            TrialRecord(
                actor_model=actor,
                distinguisher_model=distinguisher,
                condition=condition,
                branch=branch,
                answer=0,
                valid=True,
                error=None,
                answered_in_opening=False,
                turn_count=turn_count,
                specimen_turn_count=None,
                raw_path=None,
                data_provenance=data_provenance,
            )
        )
    if len(trials) != valid_trials or answer_1_count + answer_0_count != valid_trials:
        raise ValueError((actor, distinguisher, valid_trials, answer_1_count, answer_0_count))
    return trials


def pair_counts_from_matrix_a_aggregate_csv(matrix_csv: Path) -> dict[tuple[str, str], tuple[int, int, int]]:
    """(actor, dist) -> (valid_trials, answer_1_count, answer_0_count) with normalized model ids."""
    out: dict[tuple[str, str], tuple[int, int, int]] = {}
    with matrix_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            actor = _normalize_model_slug((row.get("actor") or "").strip())
            dist = _normalize_model_slug((row.get("distinguisher") or "").strip())
            vt = _parse_int(row.get("valid_trials"))
            if not actor or not dist or vt is None or vt <= 0:
                continue
            n1 = _parse_int(row.get("answer_1_count"))
            n0 = _parse_int(row.get("answer_0_count"))
            if n1 is None or n0 is None or n1 + n0 != vt:
                continue
            out[(actor, dist)] = (vt, n1, n0)
    return out


def pair_counts_from_grouped_model_pair_json(grouped_json: Path) -> tuple[list[str], dict[tuple[str, str], tuple[int, int, int]]]:
    """Parse ``model_pair_results_grouped.json``; return canonical model order and pair aggregates."""
    raw: dict[str, Any] = json.loads(grouped_json.read_text(encoding="utf-8"))
    ordering = raw.get("ordering") or {}
    model_order = [str(x).strip() for x in (ordering.get("model_order") or []) if str(x).strip()]
    pairs: dict[tuple[str, str], tuple[int, int, int]] = {}
    for group in raw.get("records_by_actor") or []:
        actor = (group.get("actor") or "").strip()
        if not actor:
            continue
        for row in group.get("rows") or []:
            dist = (row.get("distinguisher") or "").strip()
            vt = int(row["valid_trials"]) if row.get("valid_trials") is not None else None
            n1 = int(row["answer_1_count"]) if row.get("answer_1_count") is not None else None
            n0 = int(row["answer_0_count"]) if row.get("answer_0_count") is not None else None
            if not dist or vt is None or vt <= 0 or n1 is None or n0 is None or n1 + n0 != vt:
                continue
            pairs[(actor, dist)] = (vt, n1, n0)
    return model_order, pairs


def trials_from_grouped_model_pair_json(
    grouped_json: Path,
    *,
    condition: str = "no_query_no_explanation",
    matrix_fallback_csv: Path | None = None,
    impute_uniform_if_missing: bool = True,
    impute_valid_trials: int = 10,
    impute_answer_1_count: int = 5,
    impute_answer_0_count: int = 5,
) -> tuple[list[TrialRecord], dict[str, int]]:
    """Build trial list for a full ``model_order`` grid.

    Precedence per directed pair: grouped JSON → matrix CSV (optional) → uniform split (optional).

    Returns ``(trials, stats)`` where ``stats`` counts how each pair was resolved
    (``from_json``, ``from_matrix_fallback``, ``imputed_uniform``).
    """
    model_order, json_pairs = pair_counts_from_grouped_model_pair_json(grouped_json)
    if not model_order:
        raise ValueError(f"No ordering.model_order in {grouped_json}")
    if impute_answer_1_count + impute_answer_0_count != impute_valid_trials:
        raise ValueError("impute counts must sum to impute_valid_trials")

    matrix_pairs: dict[tuple[str, str], tuple[int, int, int]] = {}
    if matrix_fallback_csv is not None and matrix_fallback_csv.is_file():
        matrix_pairs = pair_counts_from_matrix_a_aggregate_csv(matrix_fallback_csv)

    stats = {"from_json": 0, "from_matrix_fallback": 0, "imputed_uniform": 0}
    merged: dict[tuple[str, str], tuple[int, int, int]] = {}

    for key in product(model_order, model_order):
        if key in json_pairs:
            merged[key] = json_pairs[key]
            stats["from_json"] += 1
        elif key in matrix_pairs:
            merged[key] = matrix_pairs[key]
            stats["from_matrix_fallback"] += 1
        elif impute_uniform_if_missing:
            merged[key] = (impute_valid_trials, impute_answer_1_count, impute_answer_0_count)
            stats["imputed_uniform"] += 1

    missing = [(a, b) for a in model_order for b in model_order if (a, b) not in merged]
    if missing:
        raise ValueError(
            f"Incomplete grid after JSON/matrix/impute: {len(missing)} pairs still missing "
            f"(set impute_uniform_if_missing=True). Example: {missing[0]!r}."
        )

    trials: list[TrialRecord] = []
    for (actor, dist), (vt, n1, n0) in sorted(merged.items()):
        if (actor, dist) in json_pairs:
            prov = "grouped_json"
        elif (actor, dist) in matrix_pairs:
            prov = "matrix_csv"
        else:
            prov = "imputed_uniform"
        trials.extend(
            _expand_aggregate_row_to_trials(
                actor,
                dist,
                valid_trials=vt,
                answer_1_count=n1,
                answer_0_count=n0,
                condition=condition,
                turn_count=None,
                data_provenance=prov,
            )
        )
    return trials, stats


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


def trials_from_matrix_a_aggregate_csv(
    matrix_csv: Path,
    *,
    condition: str = "no_query_no_explanation",
) -> list[TrialRecord]:
    """Expand aggregate matrix rows (valid_trials, answer_1/0 counts) into synthetic per-trial records.

    Rows with ``valid_trials`` missing or <= 0 are skipped (e.g. INVALID / incomplete runs).
    ``avg # turns`` is rounded and reused for every synthetic trial in that row (coarse but stable).
    """
    trials: list[TrialRecord] = []
    with matrix_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        avg_col = "avg # turns" if "avg # turns" in fieldnames else "avg_turns"
        for row in reader:
            actor = (row.get("actor") or "").strip()
            dist = (row.get("distinguisher") or "").strip()
            vt = _parse_int(row.get("valid_trials"))
            if not actor or not dist or vt is None or vt <= 0:
                continue
            n1 = _parse_int(row.get("answer_1_count"))
            n0 = _parse_int(row.get("answer_0_count"))
            if n1 is None or n0 is None or n1 + n0 != vt:
                continue
            avg_raw = (row.get(avg_col) or "").strip()
            turn: int | None
            try:
                turn = int(round(float(avg_raw.replace(",", ".")))) if avg_raw else None
            except ValueError:
                turn = None
            for _ in range(n1):
                trials.append(
                    TrialRecord(
                        actor_model=actor,
                        distinguisher_model=dist,
                        condition=condition,
                        branch="genuine_target" if actor == dist else "imitation",
                        answer=1,
                        valid=True,
                        error=None,
                        answered_in_opening=False,
                        turn_count=turn,
                        specimen_turn_count=None,
                        raw_path=None,
                    )
                )
            for _ in range(n0):
                trials.append(
                    TrialRecord(
                        actor_model=actor,
                        distinguisher_model=dist,
                        condition=condition,
                        branch="genuine_target" if actor == dist else "imitation",
                        answer=0,
                        valid=True,
                        error=None,
                        answered_in_opening=False,
                        turn_count=turn,
                        specimen_turn_count=None,
                        raw_path=None,
                    )
                )
    return trials


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
    return (
        BranchEvidence(n_valid=n, n_excluded=sum(exclusions.values()), p_hat=k / n, lower=None, upper=None),
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
                    genuine = genuine_by_target.get(b)
                    if genuine is None:
                        self_valid = 0
                        self_excluded = 0
                        self_psame = None
                        self_raw_paths: list[str] = []
                        self_genuine_prov = None
                    else:
                        genuine_ev, genuine_exc, genuine_raw, _g_lo, _g_hi = genuine
                        self_valid = genuine_ev.n_valid
                        self_excluded = genuine_ev.n_excluded
                        self_psame = genuine_ev.p_hat
                        self_raw_paths = genuine_raw
                        self_genuine_prov = _branch_trials_provenance(by_run[key])
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
                            n_valid=self_valid,
                            n_excluded=self_excluded,
                            estimate=None,
                            lower_bound=None,
                            upper_bound=None,
                            threshold_at_point=tr.threshold_at_point,
                            threshold_at_upper=tr.threshold_at_upper,
                            status_at_epsilon=tr.status,
                            p_same_real_b=self_psame,
                            p_diff_impostor=None,
                            p_correct=None,
                            avg_turn_count=_avg([t.turn_count for t in by_run[key]]),
                            avg_specimen_turn_count=None,
                            exclusions={"self_pair_calibration_only": 1},
                            raw_paths=self_raw_paths,
                            imitation_data_provenance=None,
                            genuine_data_provenance=self_genuine_prov,
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
                imitation_prov = _branch_trials_provenance(by_run[key])
                genuine_key = (cond, b, b)
                genuine_prov = _branch_trials_provenance(by_run[genuine_key]) if genuine_key in by_run else None

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
                            imitation_data_provenance=imitation_prov,
                            genuine_data_provenance=None,
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
                            imitation_data_provenance=imitation_prov,
                            genuine_data_provenance=genuine_prov,
                        )
                    )
                    continue

                p_correct = 0.5 * (genuine_ev.p_hat or 0.0) + 0.5 * (imitation_ev.p_hat or 0.0)

                tr = classify_pair(
                    epsilon=cfg.epsilon,
                    estimate=p_correct,
                    upper=None,
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
                        lower_bound=None,
                        upper_bound=None,
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
                        imitation_data_provenance=imitation_prov,
                        genuine_data_provenance=genuine_prov,
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
    results_root: Path | None = None,
    matrix_a_csv: Path | None = None,
    grouped_json: Path | None = None,
    grouped_matrix_fallback_csv: Path | None = None,
    grouped_impute_uniform: bool = True,
    epsilon: float,
    min_valid_per_branch: int = 3,
    z: float = 1.645,
    exclude_answered_in_opening: bool = True,
    exclude_any_error: bool = True,
) -> dict:
    n_sources = sum(x is not None for x in (results_root, matrix_a_csv, grouped_json))
    if n_sources != 1:
        raise ValueError("Provide exactly one of results_root, matrix_a_csv, or grouped_json")

    cfg = IngestConfig(
        results_root=results_root or Path("."),
        epsilon=epsilon,
        min_valid_per_branch=min_valid_per_branch,
        z=z,
        exclude_answered_in_opening=exclude_answered_in_opening,
        exclude_any_error=exclude_any_error,
    )

    pair_ingest_stats: dict[str, int] | None = None
    if grouped_json is not None:
        all_trials, pair_ingest_stats = trials_from_grouped_model_pair_json(
            grouped_json,
            matrix_fallback_csv=grouped_matrix_fallback_csv,
            impute_uniform_if_missing=grouped_impute_uniform,
        )
    elif matrix_a_csv is not None:
        all_trials = trials_from_matrix_a_aggregate_csv(matrix_a_csv)
    else:
        assert results_root is not None
        csvs = discover_results_csvs(results_root)
        all_trials = []
        for p in csvs:
            all_trials.extend(load_trials_from_results_csv(p))

    pair_evidence = compute_pair_evidence_strict(trials=all_trials, cfg=cfg)
    models = sorted({t.actor_model for t in all_trials if t.actor_model} | {t.distinguisher_model for t in all_trials if t.distinguisher_model})
    conditions = sorted({t.condition for t in all_trials if t.condition})
    models_by_condition = {
        cond: sorted(
            {t.actor_model for t in all_trials if t.condition == cond and t.actor_model}
            | {t.distinguisher_model for t in all_trials if t.condition == cond and t.distinguisher_model}
        )
        for cond in conditions
    }

    # Serialize as JSON-friendly dict
    return {
        "mode": "strict",
        "availableModes": ["strict"],
        "epsilonDefault": epsilon,
        "minValidPerBranch": min_valid_per_branch,
        "confidenceMethod": {"kind": "plug_in_frequency"},
        "filtersDefault": {
            "excludeAnsweredInOpening": exclude_answered_in_opening,
            "excludeAnyError": exclude_any_error,
        },
        "models": models,
        "modelsByCondition": models_by_condition,
        "conditions": conditions,
        "pairs": [
            _serialize_pair_row(ev)
            for ev in pair_evidence
        ],
        "formalismRef": "visualizer/docs/Turing_Ordering_Visualizer_Formalization_English.md",
        "dataSource": (
            "grouped_json"
            if grouped_json is not None
            else ("matrix_a_csv" if matrix_a_csv is not None else "results_csv")
        ),
        "pairIngestStats": pair_ingest_stats,
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
            "imitationDataProvenance": ev.imitation_data_provenance,
            "genuineDataProvenance": ev.genuine_data_provenance,
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
