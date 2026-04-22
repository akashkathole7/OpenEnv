# ONSITE_BRIEFING.md — handoff prompt for a fresh Claude session at Bangalore on-site

> **How to use this file.** On April 25, 2026 (on-site Day 1 at Bangalore), paste this entire document as the first message to a new Claude Code session opened inside `/home/aakash/Videos/ReconcileEnv-GST2B`. Everything below is written *as a direct prompt to that Claude*. No pre-amble, no introductions, no recap. Read the "Files to read first" section, then execute the pipeline in order.

---

## You are continuing Aakash Kathole's hackathon project

**Who.** Aakash Kathole (solo entrant). GitHub `akashkathole7`. Branch: `scaffold/reconcile-gst2b`. Fork remote: `fork`.

**Event.** Meta × Scaler School of Technology Hackathon Grand Finale, Bangalore, 25–26 April 2026. Aakash is physically on-site. Scaler provides compute (expected: single A100 40GB block via Kaggle or the sponsor cluster).

**Result timing — important for your pacing.** Per Scaler's finale email: results are *not* announced at the venue. Judging is asynchronous; results stream on Scaler's YouTube Live on **2026-05-02** (roughly one week after on-site). This means: there is no Day-2 submission cutoff that forces fake numbers. If a phase bricks, an honest "trained checkpoint exists but eval surface needs completion" beats a scrambled pipeline. See the emergency fallback section.

**Submission slot.** Round 2, Theme #3.1 "Professional World Modeling", Scaler AI Labs sub-theme "Multi-App Enterprise Workflow". Pre-recorded 90-second demo already linked across docs: https://www.youtube.com/watch?v=rglR1hGgdb8

**The project.** `reconcile_gst2b_env` — an OpenEnv RL environment for Indian GST Input-Tax-Credit reconciliation against the monthly GSTR-2B regulator return. Everything is committed, tested, and deployed to a Hugging Face Space. What remains for on-site is *real GRPO training on proper compute*, which was not possible in Round 1 (only T4 smoke runs were feasible). Your job is to shepherd that training run and update the docs with real numbers.

---

## State of the repo when you open it

Run these first before trusting anything in this doc — the repo may have moved since this briefing was written.

```bash
git status
git log --oneline -20
git branch --show-current       # expect: scaffold/reconcile-gst2b
ls envs/reconcile_gst2b_env/
ls envs/reconcile_gst2b_env/data/
```

**As of 2026-04-22 (briefing write date), the last commit was `a0f8b50` "docs: align terminology with hackathon self-serve guide".** Working tree has a small number of untracked PITCH/VIDEO/data backup files that are intentional (not blockers). If HEAD has moved forward, trust the newer commits — Aakash may have pushed incremental fixes.

**What's already done (do NOT redo):**

- Environment implementation: `server/reconcile_gst2b_environment.py`, 16 typed tool verbs (7 query + 7 mutate + 2 meta), ground_truth.py oracle, rewards.py 4-component scorer.
- 42 tests green, including 6 red-team attacks CI-enforced under 0.45 total reward.
- HF Space deployed: `app.py` + `README_HF_SPACE.md` + `requirements.txt`. Hero text is judge-facing.
- Three pre-onsite training attempts on Kaggle T4 documented in `data/training_log_qwen3_*_partial.json` — each pins a different reward component (0.6B pins R1/R2, 1.7B pins all, 4B pins R4). This is the "triple failure mode" story.
- Tier 1+2 GRPO fixes landed in `scripts/train_grpo_real.py` (`enable_thinking=False`, `beta=0.0`, zero-std jitter σ=0.005, format bonus +0.02).
- Phase 2 SFT warm-start trainer pre-staged: `scripts/train_sft_warmstart.py`.
- Phase 1 trajectory generator: `scripts/generate_sft_trajectories.py`.
- All judge docs em-dash-free (AI-tell cleanup). Do not reintroduce em-dashes.
- Terminology aligned with Scaler's self-serve guide: RLVE, RLVR, stack compliance, defense-in-depth all present in EXEC_SUMMARY / ROUND2_PROBLEM_STATEMENT / BLOG / README_HF_SPACE.

**What remains for on-site:**

1. Phase 1 — generate SFT trajectories from the oracle policy (CPU, ~2 h).
2. Phase 2 — SFT warm-start Qwen3-4B on those trajectories (A100, ~4 h).
3. Phase 3 — GRPO polish from the SFT checkpoint (A100, ~4 h).
4. Measure real hero-seed numbers (tier_a / tier_b / tier_c) from the trained policy.
5. Update `ROUND2_PROBLEM_STATEMENT.md`, `EXEC_SUMMARY.md`, `BLOG.md`, `README_HF_SPACE.md` with the new numbers — replacing the "baseline eval total 0.255" prompting-only figure with the trained total.
6. Commit on `scaffold/reconcile-gst2b`, push to `fork`, redeploy HF Space.

---

## Files to read first (in order, before doing anything)

1. **`envs/reconcile_gst2b_env/EXEC_SUMMARY.md`** — 13 numbered bullets. The judge-facing summary. Every claim in this file must survive Q&A.
2. **`envs/reconcile_gst2b_env/ROUND2_PROBLEM_STATEMENT.md`** — the Round 2 submission brief. Maps directly to judge rubric.
3. **`envs/reconcile_gst2b_env/BLOG.md`** — long-form narrative with the honest "training collapsed to 0.353" section. This is what the on-site run either ratifies or fixes.
4. **`envs/reconcile_gst2b_env/rewards.py`** — 4-component reward, clamping, and the documented `zero_itc` R3 blind-spot. **DO NOT EDIT** (see Invariants).
5. **`envs/reconcile_gst2b_env/server/reconcile_gst2b_environment.py`** — the env server and tool-verb surface.
6. **`envs/reconcile_gst2b_env/scripts/generate_sft_trajectories.py`** — Phase 1 script, read the docstring and `--help`.
7. **`envs/reconcile_gst2b_env/scripts/train_sft_warmstart.py`** — Phase 2 script, read the docstring, note the smoke-test recipe.
8. **`envs/reconcile_gst2b_env/scripts/train_grpo_real.py`** — Phase 3 script. The `MODEL_NAME` constant is the swap point for loading the SFT checkpoint.
9. **`envs/reconcile_gst2b_env/PITCH.md`** — the 60-second pitch script. Content-locked. Do not rewrite.
10. **`envs/reconcile_gst2b_env/QA_REHEARSAL.md`** — anticipated judge questions with answers. Crib sheet for Aakash during demo.

After reading 1–8, the pipeline and invariants should be self-evident. Only read 9–10 if you are asked to help rehearse or tweak pitch content.

---

## On-site task execution (3-phase pipeline)

The pipeline is deterministic. Run phases in order. Do not parallelize phases unless Aakash explicitly asks. Each phase has an explicit smoke test before the real run — always do the smoke test.

### Phase 0 — sanity check (15 min, before any training)

```bash
# Environment sanity (tests live at tests/envs/ from repo root, NOT envs/.../tests/)
PYTHONPATH=src:envs uv run pytest tests/envs/test_reconcile_gst2b_*.py -v --tb=short
# Expect: 42 tests passing. If any red-team attack test fails, STOP and diagnose.

# Import sanity — the class is ReconcileGST2BEnvironment, __init__ takes no args,
# seed goes to reset(). Do not copy-paste without this shape; prior brief had a typo.
PYTHONPATH=src:envs uv run python -c "
from envs.reconcile_gst2b_env.server.reconcile_gst2b_environment import ReconcileGST2BEnvironment
env = ReconcileGST2BEnvironment()
obs = env.reset(seed=42, mode='warmup')
print('env.reset OK, step_budget =', obs.step_budget)
"

# GPU sanity (on A100)
nvidia-smi
PYTHONPATH=src:envs uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

If any of these fail, STOP. Do not start Phase 1. Diagnose first.

### Phase 1 — generate SFT trajectories (CPU, ~2 h)

```bash
# Real arg names: --start-seed, --end-seed, --output-path, --min-total (NOT --seed-start etc).
# A --dry-run flag also exists and generates 5 seeds without writing.

# Smoke first (dry-run, 5 seeds, ~30 s — no file written)
PYTHONPATH=src:envs uv run python -m \
    envs.reconcile_gst2b_env.scripts.generate_sft_trajectories \
    --dry-run --verbose

# Real smoke (10 seeds to disk, verify row shape)
PYTHONPATH=src:envs uv run python -m \
    envs.reconcile_gst2b_env.scripts.generate_sft_trajectories \
    --start-seed 0 --end-seed 10 \
    --output-path envs/reconcile_gst2b_env/data/sft_trajectories_smoke.jsonl \
    --min-total 0.40

# Inspect one row (OpenAI-style {"messages":[...]} record)
head -1 envs/reconcile_gst2b_env/data/sft_trajectories_smoke.jsonl | python -m json.tool | head -40

# Full run (2000 seeds, ~2 h)
PYTHONPATH=src:envs uv run python -m \
    envs.reconcile_gst2b_env.scripts.generate_sft_trajectories \
    --start-seed 0 --end-seed 2000 \
    --output-path envs/reconcile_gst2b_env/data/sft_trajectories.jsonl \
    --min-total 0.40
```

**Success criteria.** `sft_trajectories.jsonl` has ≥ 1500 rows (some seeds get filtered by `--min-total`). Mean composite reward on accepted rows should be ≥ 0.55. Distribution over the 5 labels should NOT be >80% `matched` — if the oracle is label-collapsing, Phase 2 will memorise `matched` and we'll reproduce the 0.353 plateau.

**Rollout audit (do this, not optional).** Before moving to Phase 2, manually read 5 trajectories sampled at random from the JSONL. Look for: (a) at least one mark_* action per trajectory, (b) `submit` as the final action, (c) no trajectories with only `get_schema` calls. If any of these fail, Phase 2 will train the model to mimic a bad policy. Self-serve guide Q52 calls this out explicitly: reward-rising-but-quality-not is the #1 post-training failure mode.

### Phase 2 — SFT warm-start (A100, ~4 h)

```bash
# Smoke first (dry run, CPU, ~30 s — verifies code path loads)
PYTHONPATH=src:envs uv run python -m \
    envs.reconcile_gst2b_env.scripts.train_sft_warmstart \
    --input-jsonl envs/reconcile_gst2b_env/data/sft_trajectories.jsonl \
    --dry-run

# Real run (A100 40GB)
PYTHONPATH=src:envs uv run python -m \
    envs.reconcile_gst2b_env.scripts.train_sft_warmstart \
    --input-jsonl envs/reconcile_gst2b_env/data/sft_trajectories.jsonl \
    --output-dir  envs/reconcile_gst2b_env/data/sft_checkpoint \
    --model       Qwen/Qwen3-4B \
    --epochs      1 \
    --batch-size  1 \
    --grad-accum  8 \
    --learning-rate 2e-5 \
    2>&1 | tee envs/reconcile_gst2b_env/data/sft_run.log
```

**Success criteria.** Final train loss < 0.8. No OOM. A checkpoint directory at `data/sft_checkpoint/` containing `adapter_model.safetensors`. A quick eval over 30 heldout seeds should show tool-call rate > 0.8 per rollout (compared to 0.25 bimodal at 0.6B pre-fix, 0.00 at 1.7B, 1.0 but never-commits at 4B).

### Phase 3 — GRPO polish (A100, ~4 h)

```bash
# Load the SFT checkpoint via --model (it's an argparse arg, default MODEL_NAME).
# Do NOT edit the MODEL_NAME constant — pass --model on the CLI instead.
# Real arg name is --total-steps (NOT --max-steps).

# Smoke first (10 steps, confirms reward_std > 0 and grad_norm is healthy)
PYTHONPATH=src:envs uv run python -m \
    envs.reconcile_gst2b_env.scripts.train_grpo_real \
    --model envs/reconcile_gst2b_env/data/sft_checkpoint \
    --total-steps 10 \
    --output-dir envs/reconcile_gst2b_env/data/grpo_smoke \
    2>&1 | tee envs/reconcile_gst2b_env/data/grpo_smoke.log

# Real run (~200 steps, ~4 h)
PYTHONPATH=src:envs uv run python -m \
    envs.reconcile_gst2b_env.scripts.train_grpo_real \
    --model envs/reconcile_gst2b_env/data/sft_checkpoint \
    --total-steps 200 \
    --output-dir envs/reconcile_gst2b_env/data/grpo_checkpoint \
    2>&1 | tee envs/reconcile_gst2b_env/data/grpo_run.log
```

**Success criteria.** Eval total on hero seeds > 0.45 (clears the red-team ceiling). R1 ≥ 0.25 (up from 0.01 floor), R2 ≥ 0.30, R3 = 0.99 on at least half of hero seeds, R4 ≥ 0.40.

**Rollout audit (mandatory).** Before trusting the final metric, sample 5 rollouts from the GRPO run's last eval and read them manually. Specifically look for: does the model emit `mark_*` actions on mismatched invoices, or does it emit parseable-but-no-op `get_schema` repeatedly? Rising composite reward with exploit-style rollouts is the "worse-under-more-RL" failure (self-serve guide Q47). If you see it, STOP the run and diagnose rather than pushing through.

### Phase 4 — measure and update docs (~1.5 h, includes an unavoidable code gap)

**Heads-up: there is a genuine code gap here.** Neither existing baseline script can eval a trained checkpoint without edits:

- `scripts/real_baseline.py` hardcodes `MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"` and has no `--model-path` flag. Args are `--condition {raw,prompted,both}`, `--n-seeds`, `--n-samples`, `--output`, `--max-steps`, `--max-new-tokens`.
- `scripts/hero_baseline.py` has NO argparse at all. Hardcoded to seeds 9500/9501/9502, heuristic policy, writes `data/hero_baseline.json`.

Your first Phase-4 action is to close that gap. Pick one of two paths:

**Path A (preferred, smaller diff):** add a `--model-path` flag to `real_baseline.py` that overrides `MODEL_NAME` when set, reuses the rest of the pipeline, and writes to a caller-chosen `--output`. Gate behind TDD: write one failing test at `tests/envs/test_reconcile_gst2b_real_baseline_checkpoint.py` that constructs the script's main with a fake checkpoint path, confirm it fails, then implement. ~30 min.

**Path B (cleaner separation, larger diff):** add `scripts/eval_trained.py`, a new script that loads a LoRA checkpoint via PEFT, runs the env over 30 heldout seeds × 3 samples with `model.generate`, and writes a JSON with the same schema as `baseline_metrics_real.json`. Takes `--model-path` `--base-model` `--n-seeds` `--output`. Lift the rollout loop from `real_baseline.py` to avoid drift. ~45 min.

Either way, test the script against the SFT checkpoint first (fast, small) before pointing at the GRPO checkpoint. If Path A or B does not match the trained checkpoint's actual on-disk structure (LoRA adapter vs full weights — depends on how TRL saved it in Phase 3), adapt.

Then:

```bash
# Path A example invocation (after adding --model-path):
PYTHONPATH=src:envs uv run python -m \
    envs.reconcile_gst2b_env.scripts.real_baseline \
    --condition prompted \
    --model-path envs/reconcile_gst2b_env/data/grpo_checkpoint \
    --n-seeds 30 \
    --output envs/reconcile_gst2b_env/data/trained_eval.json

# Hero seeds — hero_baseline.py has no argparse; the cleanest move is NOT to hack
# it, but to reuse the new eval script restricted to hero seeds. Example:
PYTHONPATH=src:envs uv run python -m \
    envs.reconcile_gst2b_env.scripts.real_baseline \
    --condition prompted \
    --model-path envs/reconcile_gst2b_env/data/grpo_checkpoint \
    --n-seeds 3 \
    --output envs/reconcile_gst2b_env/data/trained_hero.json
# and seed the script via an env var or CLI addition if --n-seeds alone doesn't
# select 9500/9501/9502. Trace DEFAULT_SEEDS in real_baseline.py.
```

Then edit, in this order:

1. `EXEC_SUMMARY.md` bullet 5 — replace the Qwen2.5-3B prompting baseline (0.18 total, delta 1.18) with the trained Qwen3-4B number. Keep the format: total, delta, CI95.
2. `EXEC_SUMMARY.md` bullet 10 — update with trained results. The "triple failure mode" story stays (it's the pre-training diagnostic), but add a closing sentence: "Post-SFT-warm-start + GRPO polish on A100, total lifts to X.XX on 30 heldout seeds."
3. `ROUND2_PROBLEM_STATEMENT.md` — same numeric update in the results section.
4. `BLOG.md` §6 — add a short closing paragraph *after* the existing collapse narrative. Do NOT delete or soften the collapse narrative. The story is: "we diagnosed it, fixed it with Tier 1+2 + SFT warm-start, here's the trained number." Honesty is the pitch.
5. `README_HF_SPACE.md` — update hero numbers.

Commit as a single commit:
```
git add envs/reconcile_gst2b_env/data/trained_eval.json \
        envs/reconcile_gst2b_env/data/trained_hero.json \
        envs/reconcile_gst2b_env/EXEC_SUMMARY.md \
        envs/reconcile_gst2b_env/ROUND2_PROBLEM_STATEMENT.md \
        envs/reconcile_gst2b_env/BLOG.md \
        envs/reconcile_gst2b_env/README_HF_SPACE.md
git commit -m "onsite: land trained Qwen3-4B numbers (SFT warm-start + GRPO polish on A100)"
git push fork scaffold/reconcile-gst2b
```

Then redeploy the HF Space (Aakash has the git remote URL for the Space clone in `hf_space_clone/`).

---

## Decision tree — when to abort, swap, or fall back

### Phase 1 failures

| Symptom | Action |
|---|---|
| `sft_trajectories.jsonl` empty or <100 rows | Lower `--min-reward` to 0.30. Oracle policy may be scoring lower than expected on fresh seed ranges. |
| Label distribution >80% single class | Increase mismatch weights in `seed_generator.py`: the `(1,1,1,3,3)` weight tuple. DO NOT edit `rewards.py`. |
| CPU run exceeds 4 h | Kill, reduce `--seed-end` to 1000. 1000 filtered trajectories is enough for Phase 2. |

### Phase 2 failures

| Symptom | Action |
|---|---|
| CUDA OOM on Qwen3-4B load | Try `--model Qwen/Qwen3-1.7B` first. If still OOM, drop `--grad-accum` from 8 to 4. |
| TRL version error | `uv add trl>=0.21`. The script asserts ≥0.21 explicitly. |
| Final loss > 1.5 (under-fit) | Increase `--epochs` to 2. Probably not enough data; check Phase 1 row count. |
| Final loss < 0.2 (over-fit) | Shuffle `sft_trajectories.jsonl` and trim to 500 rows; rerun. Over-fit is worse than under-fit here because Phase 3 GRPO needs exploration capacity. |

### Phase 3 failures

| Symptom | Action |
|---|---|
| `reward_std = 0` across all steps | The SFT memorised a single action pattern. Fall back: swap `MODEL_NAME` to `Qwen/Qwen3-4B` (base, no SFT), run 200-step GRPO from base. Expect worse final number but still better than T4 runs. |
| Training total plateaus at 0.349 (`query_only` attack) | Format bonus is not firing. Check the `mark_*` action detector in `train_grpo_real.py`; the +0.02 bonus triggers on `>=1 action`. |
| Training total plateaus at 0.308 (`submit_all_matched` attack) | R3 rule check is not activating. This is a harder failure; STOP training, diagnose ground-truth injection into reward compute. |
| OOM on rollout | Reduce `num_generations` from 4 to 2 (already the T4 config). |

### General fallback — if on-site compute is hours not days

If the A100 is only available for, say, a 2-hour block instead of the planned 8-hour window:

- **Skip Phase 2**, run Phase 3 GRPO directly from base Qwen3-4B for 60–100 steps. We already know it fails (4B over-queries, budget-exhausts), but a partial run with the Tier 1+2 fixes applied may still lift R1/R2 off the 0.01 floor.
- **Do NOT** fake numbers. If trained numbers are worse than the prompting baseline, the honest story is: "we ran out of compute on-site; the pre-staged SFT warm-start script is in the repo for post-hackathon validation." Judges reward honesty per the collapse narrative already in `BLOG.md` §6.

---

## Invariants — DO NOT TOUCH

These are load-bearing. Breaking any of them invalidates the submission's integrity claims.

1. **`rewards.py` is frozen.** Its 4-component weights (0.40 / 0.25 / 0.25 / 0.10), clamping range `[0.01, 0.99]`, and structural `−1.0` for submit-before-any-query are content-locked. The `zero_itc` R3 blind-spot is *deliberate* and documented; defense-in-depth at R2 handles it. Do not "fix" R3.
2. **The 6 red-team attacks in `tests/test_reconcile_gst2b_reward_hacking.py` are CI-enforced under 0.45.** If you touch anything that affects reward scoring and these tests start failing, revert. The tests are the contract.
3. **Ground truth is hidden from the agent.** `ground_truth.py` is oracle-only. The env never exposes GT to the rollout policy. Do not add a "helper" tool that leaks it.
4. **Tool-verb surface (16 verbs: 7 query + 7 mutate + 2 meta) is content-locked.** Do not add or remove verbs. Judges will grep the surface for drift between Round 1 docs and on-site code.
5. **PITCH.md structure is content-locked.** You may fix a factual error (e.g., a changed number) but do not rewrite sentences. Aakash rehearsed to this text.
6. **No em-dashes in any judge-facing doc.** (AI-tell cleanup. The previous session removed 268 of them.) Use commas, colons, or periods.
7. **Preserve the honest collapse narrative in BLOG.md §6.** The story is "we diagnosed 3 failure modes and fixed them." Deleting or softening this *loses* the judge-appeal; it does not gain anything.
8. **Branch is `scaffold/reconcile-gst2b`.** Do not merge to main on-site. PRs against `main` come later.
9. **HF Space requirements.txt has 6 pinned deps** (gradio, networkx, plotly, numpy, pandas, pydantic). If you add a runtime import to `app.py`, add it here too or the Space build breaks.
10. **The 90-second demo URL `https://www.youtube.com/watch?v=rglR1hGgdb8`** is linked in 5+ docs. Do not change it unless Aakash uploads a new video.
11. **QLoRA merge footgun.** If Phase 2 uses 4-bit quantization (QLoRA) and Phase 3 saves a merged model, do NOT naively upcast 4-bit → 16-bit and merge the LoRA adapters in one step — it damages model quality. Self-serve guide Q16 warns about this explicitly. Save adapters separately, or use the framework's proper merged-save path. Test generation immediately after save; do not defer post-training inference validation to end of run.
12. **EXEC_SUMMARY.md line 8 currently says "41 tests green" — the real count is 42.** This is a pre-existing stale number. Do not fix it unilaterally on-site; flag to Aakash. If you update EXEC_SUMMARY with trained numbers in Phase 4, you can silently fix this as part of that edit.

---

## Failure-mode signatures you will see in logs

These are from the Round 1 training attempts and will probably reappear in Phase 2/3. Knowing the signature is half the fix.

- **"0.353 plateau"** — Qwen3-0.6B collapse. Eval total stays at 0.353 ± 0.004. `clipped_ratio=1.0`, `reward_std=0`. Diagnosis: `enable_thinking=True` burned the 512-token budget on `<think>`. If this reappears post-SFT, the `enable_thinking=False` monkey-patch in `train_sft_warmstart.py` or `train_grpo_real.py` did not apply. Check tokenizer config.
- **"tool-call rate 0.00 entropy 0.12"** — Qwen3-1.7B collapse. Deterministic non-tool-call output. Fix: SFT warm-start. If reappears post-SFT, Phase 1 data was under-weighted on mark_* actions.
- **"parseable JSON every turn, never commits"** — Qwen3-4B over-query. R4 pins at 0.01. Fix: SFT warm-start with trajectories that terminate early on label commitment. Check Phase 1 filtered trajectories for avg step count; should be ≤ 0.6 × max_steps.

---

## The 11 numbers Aakash will defend in Q&A

Memorise. If a judge asks "what's the baseline?" the answer is **#5**. If a judge asks "what proves the env is hard?" the answer is **#10**.

1. **~14M** GST-registered businesses in India (not 150M — that error was caught and fixed).
2. **5 mismatch types** planted: gstin_typo, invoice_number_prefix_drift, tax_slab_off_by_one, supplier_late_filing, amendment_after_2b_freeze.
3. **16 typed tool verbs** (7 query + 7 mutate + 2 meta).
4. **4 reward components**, weights 0.40/0.25/0.25/0.10, each clamped `[0.01, 0.99]`.
5. **Prompting baseline**: Qwen2.5-3B on 30 heldout seeds lifts total from −1.0 to 0.18 (delta 1.18, CI95 [1.09, 1.27]). *This number gets replaced on-site with the trained Qwen3-4B number.*
6. **6 red-team attacks** all score < 0.45, max is `query_only = 0.349`. CI-enforced.
7. **81 distinct totals**, σ = 0.50, 100% done-rate on 100 random-policy episodes. (Reward has gradient.)
8. **42 tests** green. `openenv validate --verbose` passes.
9. **0.30 probability** of a directed 3-cycle ring being planted per episode.
10. **Three distinct Qwen3 failure modes** across 0.6B / 1.7B / 4B: pin R1+R2 / pin all / pin R4. Environment is correctly hard.
11. **Stack compliance**: OpenEnv + TRL + PEFT. Unsloth drop-in ready. RLVR-style reward (verifier-based, no learned reward model).

---

## Emergency fallback (if everything breaks on-site)

The submission is *already complete* as of commit `a0f8b50`:
- Env is shippable, tested, deployed on HF Space.
- Round 2 problem statement, exec summary, blog, pitch, video are all landed.
- 42 tests green, red-team CI-enforced.
- Three Qwen3 failure modes documented honestly.

If the A100 never materialises, or if Phase 2/3 bricks unrecoverably, Aakash can present as-is. The pitch is "we built the environment correctly; compute is the gap." Judges reward shipped environments over vaporware numbers. Do not scramble to fake results — the red-team tests are CI-enforced and anyone reviewing can see the real training logs in `data/training_log_qwen3_*_partial.json`.

The only affirmative action required in this fallback: ensure the HF Space is live, the YouTube video plays, and Aakash has a working laptop with the repo cloned for the live demo.

---

## How to communicate with Aakash on-site

- Aakash may be tired, sleep-deprived, or context-switching between judge conversations and pipeline babysitting. Keep responses short and action-oriented. Use numbered steps.
- When a training run is in flight, do NOT poll nvidia-smi every minute. Use `run_in_background` on the Bash tool and wait for the process to exit. Aakash does not want running commentary.
- If you find a bug in `rewards.py` or the env server *during* on-site work — flag it, do not fix it unilaterally. The red-team CI contract is judge-visible; a silent fix looks like we're editing our way past the tests.
- If a judge-visible doc needs updating with a trained number, update it; commit it; push; redeploy HF Space. All four steps, in order, or judges may see mismatched numbers between GitHub and HF.
- Aakash's laptop clock may drift. Use `date` before any time-sensitive commit. Hackathon has a hard round-2 submission cutoff (verify with Aakash on arrival).

End of briefing. Read the files in the "Files to read first" list, then wait for Aakash's next instruction.
