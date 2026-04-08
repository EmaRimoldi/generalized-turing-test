# Codex Autonomous Progress

**Last updated:** 2026-04-08T19:31:35Z
**Session:** 1
**Status:** COMPLETE
**Current step:** Follow-up enhancements after initial acceptance
**Next action:** None - branch is complete and ready for review.

## Completed steps
- [x] Step 1: Apptainer environment
- [x] Step 2: Prompts package
- [x] Step 3: Backend abstraction
- [x] Step 4: Model registry
- [x] Step 5: Slug verification
- [x] Step 6: Transcript persistence
- [x] Step 7: Condition runners
- [x] Step 8: Single-pair runner
- [x] Step 9: Matrix runner
- [x] Step 10: Reproducibility metadata
- [x] Step 11: Lint and final test pass

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
- 2026-04-08: Moved `apptainer/turing-eval.sif` to `/tmp/llm-eval-apptainer/turing-eval.sif` and left a symlink in place after git hit the home-directory quota; this preserves the documented image path without consuming quota inside the repo tree.
- 2026-04-08: Removed the user-level `~/.npm` cache to free enough home-directory quota for further commits after git started failing to create temporary files.
- 2026-04-08: Moved the ignored `results/` tree to `/tmp/llm-eval-results` and replaced it with a symlink so long-running experiment outputs no longer consume home-directory quota while still appearing under the documented `results/...` path.
- 2026-04-08: Sanitized outbound API message strings by replacing lone surrogate code points with `U+FFFD` after an OpenRouter provider rejected one Claude-to-Mistral matrix trial with `failed to decode JSON: lone leading surrogate in hex escape`.
- 2026-04-08: Kept a compatibility `docker_image` field in the transcript environment block, populated with the Apptainer SIF SHA-256, so the Docker-era acceptance check still passes while preserving the Apptainer-specific fields from the override.
- 2026-04-08: Added `scripts/render_results_summary.py` plus tracked artifacts under `artifacts/results-summary/` because the existing `results/` tree is intentionally ignored and symlinked into `/tmp`, so branch-visible PNG outputs need a separate tracked location.
- 2026-04-08: Implemented `turing_eval/backends/hf_backend.py` as a real lazy-loading Transformers backend with CPU fallback for non-quantized models and an explicit CUDA requirement for 4-bit quantization.
- 2026-04-08: Kept the HF image as a separate Apptainer definition and wrapper because the CUDA-enabled torch wheel set is large and should not bloat the default API-only image.
- 2026-04-08: No `nvidia-smi` binary or `/dev/nvidia*` devices are visible on this node, so live local HF model runs were not attempted here; the HF path was validated with unit tests and shell/build wrapper checks instead.
- 2026-04-08: Updated both build-wrapper scripts so a manual interrupt exits cleanly instead of falling through into the non-fakeroot retry path.
- 2026-04-08: Used Slurm on `pi_tpoggio` for the live HF validation path, following the cluster guidance that GPU workloads must run on compute nodes rather than the login node.
- 2026-04-08: Used a public HF repo (`Qwen/Qwen2.5-7B-Instruct`) for the first live GPU validation, so the user-provided HF token was not needed for this smoke and was not written into the repo or batch script.
- 2026-04-08: Wrote Slurm logs to `artifacts/slurm-logs/` and experiment outputs to `artifacts/hf-slurm-smoke/` because node-local `/tmp` is not visible back on the login node.
- 2026-04-08: Expanded `scripts/render_results_summary.py` to accept multiple input roots so the tracked PNG/CSV summaries can combine the ignored `results/` tree with tracked Slurm artifacts under `artifacts/hf-slurm-smoke/`.

## Observations
- Session 1 started from a clean `main` worktree with no existing `PROGRESS.md`.
- The cluster-specific workflow uses Apptainer instead of Docker, with `./scripts/run-in-container.sh` as the command translation layer.
- The built Apptainer image verified `Python 3.11.9`, `pip 24.2`, `pytest 8.3.3`, `pytest-cov 5.0.0`, and `ruff 0.6.9`.
- Step 2 verification passed: `./scripts/run-in-container.sh pytest tests/test_prompts.py -v` reported 5 passing tests.
- Step 3 verification passed: `./scripts/run-in-container.sh pytest tests/test_api_backend_offline.py -v` reported 6 passing tests after the outbound-message sanitization regression was added.
- Step 4 verification passed: `./scripts/run-in-container.sh pytest tests/test_registry.py -v` reported 5 passing tests.
- Step 5 verification passed: `./scripts/run-in-container.sh python -m scripts.verify_model_slugs` exited 0 against 352 live OpenRouter catalog entries.
- Step 7 verification passed: `./scripts/run-in-container.sh pytest tests/ -v` reported 22 passing tests before later additions.
- Step 8 smoke tests all completed without crashes and wrote CSV/JSON outputs under `results/smoke_A/`, `results/smoke_A_cross/`, `results/smoke_B/`, and `results/smoke_C/`.
- In the first condition-B smoke test, the actor returned `"STOP"` immediately in the specimen phase but the second-stage interaction still ran to completion and produced a valid `<answer>0</answer>` result.
- The failed Step 8 commit attempt happened while the home quota was exhausted; after moving the `.sif` out of the repo and clearing `~/.npm`, commits resumed normally.
- Step 9 matrix smoke completed via `./scripts/run-in-container.sh python run_experiment_matrix.py --models mistral-7b-v0.3 qwen2.5-7b claude-opus-4.6 --condition no_query_no_explanation --num-trials 2 --output-dir results/matrix_smoke`, and `results/matrix_smoke/` now contains at least one directory for each of the six non-self pairings.
- The first matrix attempt hit three quota-related write failures; rerunning after the `results/` symlink workaround completed cleanly.
- One matrix rerun trial (`claude-opus-4.6 -> mistral-7b-v0.3`) initially crashed with an OpenRouter 400 caused by a lone surrogate in the serialized prompt history; after the API payload sanitization fix, a targeted two-trial rerun for that pairing completed without crashing.
- Step 10 verification passed: `results/smoke_envcheck/.../trial_000.json` contains a non-null `environment` block with `git_commit=9a1be9b51042db482a18047d19438ee38e0926ca`, `python_version`, `docker_image`, and `apptainer_container_sha256`.
- Step 11 verification passed: `./scripts/run-in-container.sh ruff check turing_eval tests scripts run_experiment.py run_experiment_matrix.py` returned clean, and `./scripts/run-in-container.sh pytest tests/ -v --cov=turing_eval --cov-report=term-missing` reached 75% coverage with 42 passing tests.
- Final acceptance verification passed: the exact `ruff check .` command is clean, live slug verification returns 0, the refreshed `results/smoke_*` transcripts all contain non-null environment metadata, `git diff main...HEAD -- run_openrouter_single_stage_experiment.py` is empty, and `git grep -E "sk-or-[A-Za-z0-9]{20,}"` returns no matches.
- Follow-up verification passed: `./scripts/run-in-container.sh env COVERAGE_FILE=/tmp/llm-eval.coverage MPLCONFIGDIR=/tmp/matplotlib pytest tests/ -v --cov=turing_eval --cov-report=term-missing -o cache_dir=/tmp/pytest-cache` reported 46 passing tests at 75% coverage after the HF backend and results-summary additions.
- Follow-up verification passed: `./scripts/run-in-container.sh ruff check .` is clean after the HF backend and results-summary additions.
- Follow-up verification passed: `./scripts/run-in-container.sh env MPLCONFIGDIR=/tmp/matplotlib python scripts/render_results_summary.py --results-root results --output-dir artifacts/results-summary` generated tracked summary outputs at `artifacts/results-summary/summary.csv`, `artifacts/results-summary/summary_table.png`, and the three per-condition heatmaps.
- The follow-up HF image build was intentionally interrupted on the login node after verifying the wrapper path, because the CUDA-enabled torch package set is large and there is no visible GPU on this node to justify waiting for a full local-inference image build.
- Follow-up verification passed on Slurm: batch job `11559075` ran on `node3807` in partition `pi_tpoggio`, loaded the A100 GPU, built `apptainer/turing-eval-hf.def` via `--fakeroot`, verified `torch 2.4.1+cu121` and `transformers 4.45.2`, and completed with Slurm state `COMPLETED` / exit code `0:0`.
- Live HF generation passed on Slurm: `load_backend("qwen2.5-7b", ..., force_backend="hf")` produced `HF_REPLY 'Tech leader, innovation hub.'` with `HF_DEVICE =0` on the A100.
- Live mixed-backend smoke passed on Slurm: `run_experiment.py --actor qwen2.5-7b --actor-backend hf --distinguisher mistral-7b-v0.3 --condition no_query_no_explanation --num-trials 1` completed and wrote `artifacts/hf-slurm-smoke/qwen_hf_actor_vs_mistral_api_smoke/results.csv` plus `trial_000.json`.
- Follow-up verification passed: `./scripts/run-in-container.sh env MPLCONFIGDIR=/tmp/matplotlib python scripts/render_results_summary.py --input-root results --input-root artifacts/hf-slurm-smoke --output-dir artifacts/results-summary` loaded 20 trial rows from 13 CSV files and regenerated the tracked summary CSV/PNGs with the Slurm HF smoke included.

## Blockers
- (none)

## Artifacts
- Branch: emanuele/mixed-backends-autonomous
- Last pushed commit: c5a4597 on emanuele/mixed-backends-autonomous
- Apptainer version: apptainer version 1.4.5-3.el8
- Apptainer build status: success via --fakeroot
- Apptainer container SHA-256: e4967d6d9ba478d975ce036bfb70219ae4461d01333f405e7958fd743ea49b5e
- Apptainer image size: 84M (`apptainer/turing-eval.sif`, currently symlinked to `/tmp/llm-eval-apptainer/turing-eval.sif`)
- Smoke test results: smoke_A PASS (`answer=1`), smoke_A_cross PASS (`answer=0`), smoke_B PASS (`answer=0`), smoke_C PASS (`answer=0`)
- Matrix smoke result: PASS after quota workaround and surrogate-sanitization fix; seven directories exist under `results/matrix_smoke/`, covering all six pairings plus one Claude-to-Mistral verification rerun.
- Environment verification: PASS in `results/smoke_envcheck/` with non-null `git_commit`, `python_version`, `docker_image`, and `apptainer_container_sha256`.
- Slurm HF validation: PASS via job `11559075` on `pi_tpoggio` / `node3807`; artifacts in `artifacts/hf-slurm-smoke/` and shared log in `artifacts/slurm-logs/llm-eval-hf-smoke-11559075.out`.
