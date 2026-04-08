# Codex Autonomous Progress

**Last updated:** 2026-04-08T20:43:36Z
**Session:** 1
**Status:** IN_PROGRESS
**Current step:** Step 6: Transcript persistence
**Next action:** Add JSON transcript persistence and Susan-compatible CSV writing, then commit and push the persistence layer before implementing the condition runners.

## Completed steps
- [x] Step 1: Apptainer environment
- [x] Step 2: Prompts package
- [x] Step 3: Backend abstraction
- [x] Step 4: Model registry
- [x] Step 5: Slug verification
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
- 2026-04-08: Corrected `gemini-3-pro` from `google/gemini-3-pro-preview` to `google/gemini-3.1-pro-preview` after live OpenRouter verification.
- 2026-04-08: Corrected `glm-5-reasoning` from `z-ai/glm-5-reasoning` to `z-ai/glm-5` because the reasoning-specific slug is not routed on OpenRouter.
- 2026-04-08: Corrected `qwen3-30b-a3b` from `qwen/qwen3-30b-a3b-instruct` to `qwen/qwen3-30b-a3b` after live OpenRouter verification.
- 2026-04-08: Routed `qwen2.5-32b` to `qwen/qwen3-32b` because OpenRouter no longer exposes a plain `qwen2.5-32b` text route.
- 2026-04-08: Routed `qwen2.5-7b` to `qwen/qwen3-8b` because OpenRouter no longer exposes a plain `qwen2.5-7b` text route and the smoke tests require an OpenRouter-accessible weak Qwen model.
- 2026-04-08: Corrected `mistral-7b-v0.3` from `mistralai/mistral-7b-instruct-v0.3` to `mistralai/mistral-7b-instruct-v0.1` because `v0.3` is not currently routed on OpenRouter.

## Observations
- Session 1 started from a clean `main` worktree with no existing `PROGRESS.md`.
- The cluster-specific workflow uses Apptainer instead of Docker, with `./scripts/run-in-container.sh` as the command translation layer.
- The built Apptainer image verified `Python 3.11.9`, `pip 24.2`, `pytest 8.3.3`, `pytest-cov 5.0.0`, and `ruff 0.6.9`.
- Step 2 verification passed: `./scripts/run-in-container.sh pytest tests/test_prompts.py -v` reported 5 passing tests.
- Step 3 verification passed: `./scripts/run-in-container.sh pytest tests/test_api_backend_offline.py -v` reported 5 passing tests.
- Step 4 verification passed: `./scripts/run-in-container.sh pytest tests/test_registry.py -v` reported 5 passing tests.
- Step 5 verification passed: `./scripts/run-in-container.sh python -m scripts.verify_model_slugs` exited 0 against 352 live OpenRouter catalog entries.

## Blockers
- (none)

## Artifacts
- Branch: emanuele/mixed-backends-autonomous
- Last pushed commit: c5cd993 on emanuele/mixed-backends-autonomous
- Apptainer version: apptainer version 1.4.5-3.el8
- Apptainer build status: success via --fakeroot
- Apptainer container SHA-256: e4967d6d9ba478d975ce036bfb70219ae4461d01333f405e7958fd743ea49b5e
- Apptainer image size: 84M (`apptainer/turing-eval.sif`)
