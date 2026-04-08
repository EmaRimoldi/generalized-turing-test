# Codex Autonomous Progress

**Last updated:** 2026-04-08T20:36:59Z
**Session:** 1
**Status:** IN_PROGRESS
**Current step:** Step 2: Prompts package
**Next action:** Create the `turing_eval` package skeleton and prompt tests, then run the prompt test file inside the Apptainer container.

## Completed steps
- [x] Step 1: Apptainer environment
- [ ] Step 2: Prompts package
- [ ] Step 3: Backend abstraction
- [ ] Step 4: Model registry
- [ ] Step 5: Slug verification
- [ ] Step 6: Transcript persistence
- [ ] Step 7: Condition runners
- [ ] Step 8: Single-pair runner
- [ ] Step 9: Matrix runner
- [ ] Step 10: Reproducibility metadata
- [ ] Step 11: Lint and final test pass

## Decisions log
- 2026-04-08: Replaced the repository's prior catch-all `.gitignore` with a standard ignore list because the original file would have prevented committing the new package, scripts, docs, and tests required by the guide.
- 2026-04-08: Added `APPTAINER_OVERRIDE.md` and `CODEX_AUTONOMOUS_GUIDE.md` to `.gitignore` because they are operator-provided local guides, not repository deliverables, and leaving them untracked would keep the worktree permanently dirty.
- 2026-04-08: Used the direct Apptainer build path because `apptainer build --fakeroot` succeeded on this cluster; no fallback venv path is in use.

## Observations
- Session 1 started from a clean `main` worktree with no existing `PROGRESS.md`.
- The cluster-specific workflow uses Apptainer instead of Docker, with `./scripts/run-in-container.sh` as the command translation layer.
- The built Apptainer image verified `Python 3.11.9`, `pip 24.2`, `pytest 8.3.3`, `pytest-cov 5.0.0`, and `ruff 0.6.9`.

## Blockers
- (none)

## Artifacts
- Branch: emanuele/mixed-backends-autonomous
- Last pushed commit: c5cd993 on emanuele/mixed-backends-autonomous
- Apptainer version: apptainer version 1.4.5-3.el8
- Apptainer build status: success via --fakeroot
- Apptainer container SHA-256: e4967d6d9ba478d975ce036bfb70219ae4461d01333f405e7958fd743ea49b5e
- Apptainer image size: 84M (`apptainer/turing-eval.sif`)
