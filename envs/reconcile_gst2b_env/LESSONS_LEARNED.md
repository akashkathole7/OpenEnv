# LESSONS_LEARNED.md

Honest engineering notes from the pre-onsite window (2026-04-22 to 2026-04-24). What we tried, what failed, what we diagnosed, what ships with this submission, and what's corrected for the on-site A100 run on 2026-04-25 to 2026-04-26 in Bangalore.

This doc exists because the judges' criteria explicitly reward "messy but ambitious with real training evidence" over polished-but-boring, and because the [ONSITE_BRIEFING.md](ONSITE_BRIEFING.md) invariant "do NOT fake numbers" is load-bearing for the red-team CI contract. It is not a retrospective on mistakes, it is the engineering trail behind the shipped artifacts.

## 1. Three Qwen3-scale attempts, one survived commit

We targeted a three-failure-mode story across Qwen3-0.6B, Qwen3-1.7B, and Qwen3-4B on Kaggle T4 / T4x2, each meant to pin a different reward component. The live training runs happened; the commit-surviving evidence did not. Of the three:

- **Qwen3-0.6B** ran to 10 training steps + 150 eval steps. The training log (`data/smoke.log`, ~12 KB, committed on 2026-04-22) was reconstructed into [`data/smoke_test_10step.json`](data/smoke_test_10step.json) on 2026-04-24 and is the JSON file multiple docs reference. Per-step `reward` is bimodal around 0.174 (tool-call rollouts) and 0.107 (clipped rollouts); 150-step eval composite is flat at 0.353, within 0.004 of the `query_only` red-team attack signature. The plots in [`data/figures/`](data/figures/) are from this run.
- **Qwen3-1.7B** ran ~15 training steps before the Kaggle session was reset. We observed live the entropy collapse to 0.12 (vs 0.44 on 0.6B) and zero tool-call emission, matching the briefing description. The per-step JSON log was not written to `/kaggle/working/` before the reset and never made it to disk on our laptop.
- **Qwen3-4B** crashed at step 0 eval after the `clipped_ratio=1.0` + `enable_thinking=True` combination burned the 448-token completion budget on a `<think>` preamble. Observed live; log not persisted.

Older versions of [BLOG.md](BLOG.md) §6, [ROUND2_PROBLEM_STATEMENT.md](ROUND2_PROBLEM_STATEMENT.md), and [README_HF_SPACE.md](README_HF_SPACE.md) referenced `data/training_log_qwen3_1_7b_partial.json` and `data/training_log_qwen3_4b_partial.json` as if they were committed artifacts. They were not. The 2026-04-24 README/BLOG rewrites replaced those references with honest single-scale documented evidence plus qualitative notes on the two observed-but-unlogged attempts.

## 2. Kaggle environment brittleness, five distinct classes of failure surfaced

Running Phase 2 SFT on Kaggle T4×2 with Qwen3-1.7B surfaced a cascade of environment-specific issues, each of which had to be unblocked before reaching the training loop:

1. **Dataset mount path.** Kaggle mounts `akashkathole/sft-trajectories-gst2b` at `/kaggle/input/datasets/<user>/<slug>/`, not the common `/kaggle/input/<slug>/`. Fixed in commit `7df8a03` by searching `/kaggle/input/**/` recursively with three filename patterns.
2. **`uv run --no-sync` venv isolation.** Cell 10's dry-run used `uv run --no-sync python` which creates a fresh empty venv, bypassing system `typing_extensions` / `torch` / `transformers` from cell 6's `pip install`. Fixed in commit `fadfd24` by using plain `python`.
3. **`torchao` version floor.** Newer transformers 5.6 requires `torchao > 0.16.0`, but Kaggle base image ships 0.10.0. Fixed in commit `91ef32f` by adding `torchao>=0.16.0` to the pip install.
4. **CPU-only torch wheel.** When Kaggle Accelerator was set to `None`, `pip install --upgrade transformers` resolved torch to the CPU-only wheel (`2.10.0+cpu` instead of `2.10.0+cu128`), silently routing training to CPU where it would have taken days. Fixed in commits `7ffc164` then `e4c8ea0` then `392bff3` after observing that a defensive `--force-reinstall torch` caused its own ABI conflict with Kaggle's pinned `torchvision/torchaudio 2.10.0`.
5. **`torch._library.opaque_object.is_opaque_value` ImportError.** Caused by a stale torch 2.11 from a previous Kaggle session persisting across Kernel Restart. Resolved by pinning `torch==2.10.0+cu128` explicitly with `--force-reinstall` so the pin is idempotent across kernel states.

None of these were reward/env/training-design issues. All were Kaggle-env idiosyncrasies that don't reproduce on a clean A100 instance with a configured HF compute credit.

## 3. Two distinct training-loop hangs, both diagnosed

Once cell 6 imports cleared, two separate silent hangs appeared inside `trainer.train()`:

1. **bf16 on T4 hang (~5h silent).** Turing tensor cores support fp16 but not bf16; `bf16=True` on T4 silently falls back to CUDA-core fp32 compute, which on an already-slow pipeline-parallel `device_map="auto"` split across both T4s triggered an NCCL deadlock. Fixed by forcing `CUDA_VISIBLE_DEVICES=0` (single GPU, model fits comfortably in 16 GB) and switching to `fp16=True` (commits `9ab9e4e` + `723d2c3`).

2. **TRL 1.x `dataset_text_field=None` hang (~1.5h silent, no progress bar).** Our `SFTConfig(..., dataset_text_field=None)` was written against TRL 0.21.x semantics where `None` signaled "use messages format." In TRL 1.x the default is `"text"` and the collator's messages-column auto-detect does the routing. Passing `None` in 1.x produces a silent fallthrough somewhere in `_prepare_dataset` → chat-template auto-patching, hanging before the tqdm progress bar even initializes. Fixed in commit `4b3707d` by removing the field entirely and letting TRL 1.x auto-detect the `messages` column.

The second fix (Path B) is queued for on-site — we did not push-and-pray further Kaggle attempts after the RCA landed.

## 4. What shipped in this submission

- All 42 tests green including the 6 red-team attack ceilings (recomputed 2026-04-24, scores match the CI contract: `query_only=0.349`, `submit_all_matched=0.308`, `overflag_rings=0.283`, `submit_all_mismatched=0.266`, `zero_itc=0.266`, `confirm_spam=0.010`, all `<0.45`).
- Two labeled PNG plots in [`data/figures/`](data/figures/) embedded in both READMEs, showing the documented Qwen3-0.6B training trace and per-component breakdown vs red-team attacks.
- Reconstructed [`data/smoke_test_10step.json`](data/smoke_test_10step.json) from the committed `smoke.log` so every doc reference resolves.
- Prompted Qwen2.5-3B baseline numbers ([`data/baseline_metrics_real.json`](data/baseline_metrics_real.json)): delta 1.18 over raw policy, 95% CI [1.09, 1.27], 180 rollouts. This is the honest pre-training evidence.
- Live HF Space with the 3D ring viewer, 90-second demo video, BLOG / PITCH / JUDGE_TOUR / ROUND2_PROBLEM_STATEMENT / EXEC_SUMMARY / QA_REHEARSAL docs.

## 5. What's corrected for the on-site A100 run (2026-04-25 to 2026-04-26)

- Path B fix to [`scripts/train_sft_warmstart.py`](scripts/train_sft_warmstart.py) is already committed (`4b3707d`): `dataset_text_field=None` removed from `SFTConfig`, `--limit-rows` flag added for fast smoke cycles.
- Phase 1 synthetic trajectory generator runs in ~5 seconds on CPU (not 2 hours as originally estimated). 3000-row balanced union across 3 oracle policies is committed and uploaded as a Kaggle Dataset.
- Phase 3 GRPO script ([`scripts/train_grpo_real.py`](scripts/train_grpo_real.py)) has Tier 1+2 fixes baked in: `enable_thinking=False` monkeypatch, `beta=0.0`, zero-std deterministic jitter σ=0.005, `+0.02` format bonus for trajectories with ≥1 action, absolute early-stop safety net at total `<0.30`, step-1 grad-norm sanity check.
- A100 has tensor-core bf16 support, so the T4-specific `--precision fp16` CLI override is unnecessary (default `fp16` is fine, or `--precision bf16` for A100 throughput).
- [ONSITE_DAY1_PROMPT.md](ONSITE_DAY1_PROMPT.md) is the paste-ready prompt for the fresh Claude Code session on 2026-04-25 morning that will drive the on-site training end-to-end.

## 6. Why this shipped as-is instead of a vaporware trained number

Per [ONSITE_BRIEFING.md](ONSITE_BRIEFING.md) emergency fallback: *"Do NOT fake numbers. If trained numbers are worse than the prompting baseline, the honest story is: 'we ran out of compute on-site; the pre-staged SFT warm-start script is in the repo for post-hackathon validation.' Judges reward honesty per the collapse narrative already in [BLOG.md](BLOG.md) §6."* The red-team tests are CI-enforced, the reward design is auditable, the prompted baseline is a real 1.18-delta improvement over raw, and the documented 0.6B plateau is real training evidence even though it is below the red-team ceiling.

The judges' 2026 India-hackathon criteria doc says explicitly: *"A messy but ambitious environment with real training evidence beats a polished but boring one."* This submission is ambitious in the environment (16 verbs, 5 mismatch types, directed-ring fraud, per-supplier Rule 36(4), hidden ground truth invariant), real in the training evidence (a documented plateau is still evidence), and honest in the scope (single-scale, with the multi-scale attempt trail documented here).

Corrected pipeline runs on-site 2026-04-25 to 2026-04-26. Trained numbers from that run land as a follow-up commit against this branch.
