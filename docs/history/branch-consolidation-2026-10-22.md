# Branch consolidation, 22 October 2026

All five remote branches were reviewed against `main` at
`9c2fdf942d77ab96381496df487696111f604cb1`. Counts below refer to that snapshot.
`main` remains the sole maintained branch; no published history was rewritten.

| Branch | Tip | Commits unique to branch / main | Decision |
| --- | --- | --- | --- |
| `main` | `9c2fdf9` | — | Keep as the integration and reproduction branch. |
| `cleanup/reproducible-visualizer-build` | `0480769` | 0 / 10 | Already merged via `c17d15f`; remove redundant branch. |
| `emanuele/mixed-backends-autonomous` | `9119c98` | 0 / 52 | Entire history already in main; remove redundant branch. |
| `main-pre-swap-2026-04-11` | `2c4ca79` | 3 / 79 | Superseded standalone runners; preserve with annotated tag `archive/main-pre-swap-2026-04-11`, then remove branch. |
| `susan/main-pull-local-changes` | `c8dda7d` | 2 / 16 | Merge useful protocols, schedules, tests, and hardened resuming; retain other artifacts in the merged history, then remove branch. |

## Unique work and conflict decisions

The pre-swap commits `b144f4f`, `7de9015`, and `2c4ca79` fix the original
standalone OpenRouter runner and add batch/no-actor-prompt runners and a Cartesian
CSV generator. The maintained protocol registry, batch/matrix launchers, and
unprompted self-pair condition supersede those workflows. Restoring the old
scripts would reintroduce separate routing, validation, output schemas, and
failure semantics. Their exact implementation remains in the archive tag.

Susan's `c51dffe` adds two trusted-target variants and cluster/pilot artifacts;
`c8dda7d` adds automatic trial resuming. The two protocol prompt texts are
preserved exactly and integrated with main's shared argument validation and
interrogation logic. Target identity uses the registry display name. A trusted
judge's specimen uses the nominated target rather than the judge's model.

Automatic resuming was unfinished: it silently skipped incompatible/unreadable
records, did not verify budgets or routes, and truncated the loaded trial set
when fewer trials were requested. The maintained implementation requires
`--resume`, validates a run manifest and every trial/failed-attempt record,
preserves all existing evidence, and serializes writers. Historical runs without
a manifest are intentionally refused. The baseline collection/scoring policy
and paper schedules are unchanged.

Five small historical CSV schedules are retained in `configs/historical/`.
Renamed copies of existing pilot CSVs resolve to main's documented
`data/pilot-results/` layout with identical bytes. The branch's `old_results/`
contains 41 trial JSONs, 22 failed-attempt JSONs, 30 CSVs, and two macOS metadata
files. These are an additional incomplete historical subset, not the final paper
data. They remain available in the merged parent commit. They are not silently
added to the explorer's scientific inputs or reinterpreted as audited verdicts.

The branch's `make_jobs.py` still embeds a specific cluster checkout and module
version. It and the obsolete `summarize_results.py`, stale summary CSV, and
macOS metadata remain in history rather than returning to the maintained tree.
Current reporting lives in `scripts/render_results_summary.py`; maintained
launching uses the single, batch, and matrix CLIs.

## Recover exact historical artifacts

Both sources remain reachable after branch deletion:

```bash
# Standalone OpenRouter runner and its original environment/instructions:
git show archive/main-pre-swap-2026-04-11:README.md

# Original cluster launcher, transcripts, schedules, and summary:
git ls-tree -r --name-only c8dda7ddb351669bb8e2f08c3cc95117e9d5bb29 old_results/
git show c8dda7ddb351669bb8e2f08c3cc95117e9d5bb29:make_jobs.py
git archive c8dda7ddb351669bb8e2f08c3cc95117e9d5bb29 old_results/ > historical-results.tar
```

Historical transcript commit identifiers may require the existing
[`commit-map.tsv`](commit-map.tsv). Archival records are preserved as recorded;
their routes, missing transcripts, and verdict policies need separate inspection
before analysis. A complete pre-cleanup Git bundle was also saved locally during
the consolidation.

## Verification

Before updating main: Python 3.11 dependency installation and offline tests,
Ruff, frontend dependency audit, frontend tests, TypeScript/production build,
and byte-identical regeneration of the combined 162-pair explorer snapshot.
Added regression tests cover trusted-target routing, both query stages,
interrupted/complete/extended runs, configuration and implementation mismatches,
corrupt or mixed records, preserved failed attempts, and concurrent writers.
Historical schedules are loaded with the maintained validator, and all existing
pilot/grouped data and dependency declarations are unchanged.

Paid hosted experiments, CUDA/4-bit inference, and Apptainer execution are not
part of the local offline checks. The optional real HF CPU smoke runs separately
in CI; the ordinary offline suite skips it unless explicitly enabled.
