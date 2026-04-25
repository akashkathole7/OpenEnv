---
title: reconcile_gst2b_env
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.12.0"
app_file: app.py
pinned: true
license: mit
short_description: Multi-turn enterprise workflow RL on OpenEnv
tags:
  - openenv
  - reinforcement-learning
  - enterprise-workflow
  - tool-use
  - multi-turn
  - scaler-ai-labs
---

# reconcile_gst2b_env

**An OpenEnv environment for the monthly GSTR-2B reconciliation that ~14M Indian businesses perform by hand. Multi-turn, partially observable, arithmetic reward, 6 CI-enforced red-team attacks all scoring under 0.45.** Targeting the **Scaler AI Labs, Multi-App RL Environment for Enterprise Workflows** sub-theme at the Meta × Scaler Hackathon, Bangalore 2026.

> 🎮 **Try the live demo below.** Scroll past this README, select the **"Circular-Ring Viewer"** tab, pick a hero seed (9500 easy / 9501 medium / 9502 hard), click **Render** to see a 3D supplier graph with a planted directed-cycle fraud ring highlighted in red.

🎥 [90-second demo video](https://www.youtube.com/watch?v=rglR1hGgdb8) · 🧑‍💻 [GitHub repo](https://github.com/akashkathole7/OpenEnv/tree/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env) · 📝 [BLOG.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/BLOG.md) · 📋 [ROUND2_PROBLEM_STATEMENT.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/ROUND2_PROBLEM_STATEMENT.md) · 🧪 [JUDGE_TOUR.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/JUDGE_TOUR.md) · 📓 [Training notebook](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/scripts/kaggle_phase2_sft.ipynb) ([Colab](https://colab.research.google.com/github/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/scripts/kaggle_phase2_sft.ipynb))

[![Open Demo Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/colab_demo.ipynb) Judge-runnable demo: env import + oracle episode + trained-checkpoint audit (n=5 mean 0.305) + embedded plots, all under 5 minutes on free Colab T4 (no training required).

---

## TL;DR: what landed against the judging rubric

- **Environment innovation (40%):** 16 typed verbs (7 query / 7 mutate / 2 meta), 5 planted mismatch types, 30%-probability directed 3-cycle ring fraud, partially observable, hidden ground truth unit-tested to never leak into observations.
- **Storytelling (30%):** BLOG, 90-sec video, PITCH, QA_REHEARSAL, this Space with the 3D ring viewer you see below, README front-loads plots per judges' guidance.
- **Training evidence (20%):** Pre-onsite Qwen3-0.6B GRPO 10-step smoke + 150-step eval plateau (plots below). On-site Day 1 (2026-04-25, A100 SXM4-80GB): full 375-step Qwen3-4B SFT (n=5 mean 0.280) + **100-step P3 GRPO with Tier 2c length-shaping bonus, n=5 mean 0.305 (+0.025 lift)** above prompted Qwen2.5-3B baseline 0.18. [LESSONS_LEARNED.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/LESSONS_LEARNED.md) §1 covers all 5 documented failure modes including the FM4 audit-OOD trap chain (generalizes across SFT/GRPO/eval/audit code paths) and FM5 reward-landscape inversion (partially mitigated by P3 shaping).
- **Reward & pipeline (10%):** 4-component arithmetic reward clamped to `[0.01, 0.99]`, 6 red-team attacks CI-enforced at `<0.45`, 42 tests green, Tier 1+2 GRPO fixes validated in [`data/smoke_test_10step.json`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/smoke_test_10step.json).

---

## At a glance

| Metric | Value |
|---|---:|
| Typed tool verbs | **16** (7 query / 7 mutate / 2 meta) |
| Label classes | **5** (matched, mismatched, only_in_books, only_in_2b, partial) |
| Planted mismatch types | **5** + directed 3-cycle circular-trading rings (30% per episode) |
| Reward components | **4** arithmetic, each clamped to `[0.01, 0.99]` |
| Red-team attacks CI-enforced `<0.45` | **6 / 6 pass**, max `query_only` = 0.349 |
| Test suite | **42 / 42 green** |
| **Trained Qwen3-4B SFT + P3 GRPO on A100 SXM4-80GB (Day 1 on-site)** | **n=5 mean composite reward 0.305** |
| SFT-only baseline (pre-GRPO) | 0.280 (n=5 mean) |
| Prompted vs raw baseline delta on Qwen2.5-3B-Instruct | 1.18 (95% CI [1.09, 1.27], 180 rollouts) |
| Trained-vs-prompted lift | +0.125 (0.305 − 0.18) |
| GRPO P3 vs SFT lift | +0.025 (0.305 − 0.280) |
| Documented pre-onsite training run | Qwen3-0.6B, 10-step smoke + 150-step eval (plots below) |
| `make reproduce` | bit-identical against committed artifacts |

---

## Why it matters (30-second hook)

Every registered Indian business with B2B purchases reconciles its purchase register against the regulator-generated GSTR-2B return monthly. It's rule-heavy, error-sensitive (over-claim triggers audit; under-claim forfeits money), still done by hand by junior accountants, and the regulator publishes the correctness rules. That makes it a cleaner RL target than "general web agent": wrong answers have crisp, auditable, regulator-specified correctness. An agent that clears this generalizes to every two-sided document reconciliation against a schema.

---

## Results

### Day 1 on-site (2026-04-25, A100 SXM4-80GB): trained Qwen3-4B SFT vs all references

![Day 1 baseline-comparison bar chart: trained Qwen3-4B SFT (purple, 0.280) vs oracle, 6 red-team attacks, prompted Qwen2.5-3B baseline, and 0.45 red-team ceiling](https://raw.githubusercontent.com/akashkathole7/OpenEnv/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/figures/day1_baseline_comparison.png)

*Day 1 on-site result. Trained Qwen3-4B SFT (purple, n=5 mean 0.280, min-max error bar 0.17 - 0.35) sits above the prompted Qwen2.5-3B baseline (blue, 0.18) and below the CI-enforced red-team ceiling (red dashed, 0.45). The 6 red-team attacks (red bars) all score under 0.45 by design. The `query_only` attack (0.349) scores ABOVE the trained Mode A marking trajectories. This is the reward-landscape inversion documented as Failure Mode 5 in [LESSONS_LEARNED.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/LESSONS_LEARNED.md) §1: GRPO advantage signal would point toward the attack, not away from it. Phase 3 GRPO deferred. Same chart as Tab 4 in this Space; live-recomputed via [`scripts/make_day1_figure.py`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/scripts/make_day1_figure.py).*

### Day 1 training progression: step 350 vs step 375 (final), empirical proof of FM5

![Day 1 training-progression bar chart: prompted baseline (0.18), trained SFT step 350 (0.314, light purple), trained SFT step 375 final (0.280, dark purple), with 0.45 red-team ceiling and 0.349 query_only Mode B exploit reference lines](https://raw.githubusercontent.com/akashkathole7/OpenEnv/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/figures/day1_training_progression.png)

*Day 1 training progression, n=5 audit on heldout seeds 9030-9034 at GRPO-matching sampling with `tools=` enabled. Step 350 mean (0.314) is HIGHER than step 375 final mean (0.280), but only because 4 of 5 seeds at step 350 collapsed to the Mode B `query_only` attack shape (R_total 0.353), while only 2 of 5 did at step 375. **The last 25 optimizer steps moved the policy distribution from "mostly attack" (1/5 marking) to "mostly marking" (3/5 marking) AND the composite score went DOWN.** This is empirical proof of Failure Mode 5 (reward-landscape inversion) in [LESSONS_LEARNED.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/LESSONS_LEARNED.md) §1: under the current arithmetic-reward + R3-R4-saturation design, marking trajectories (Mode A) score lower than the cheap query_only attack (Mode B). Even at n=5 the direction is structurally robust: any seed shifting from Mode B (0.353) to Mode A (0.17 - 0.26) mechanically pulls the mean down. GRPO advantage signal would push the policy back toward Mode B. Source: [data/audit_ckpt350_F_n5.json](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/audit_ckpt350_F_n5.json) + [data/audit_F_n5.json](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/audit_F_n5.json) via [`scripts/make_day1_progression_figure.py`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/scripts/make_day1_progression_figure.py).*

### P3 GRPO with Tier 2c length-shaping bonus: train reward + audit-OOD visualization

![P3 GRPO reward curve: top pane shows train reward over 100 steps with rolling mean; bottom pane visualizes the FM4 audit-OOD bug by contrasting the buggy fresh-context eval callback (flat at 0.354) with the true post-training audit (n=5 mean 0.305 with tools= enabled)](https://raw.githubusercontent.com/akashkathole7/OpenEnv/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/figures/grpo_reward_curve.png)

*Two-pane visualization of the P3 GRPO outcome. **Top pane:** TRL train-reward across 100 steps under the Tier 2c length-shaping bonus (`+0.10` if `n_marks >= 10` AND `distinct_verbs >= 4`, gates verified to fire on no red-team attack). 10-step rolling mean trends from ~0.18 (start) to ~0.23 (end), confirming GRPO is moving the policy in the shaped direction. **Bottom pane:** the buggy fresh-context eval callback in `train_grpo_real.py:_eval_episode` (Failure Mode 4) reports a flat 0.354 across all 6 logged eval steps, identical to the step-0 baseline, regardless of training progress. The true post-training audit (n=5, with `tools=` enabled, multi-turn accumulation) lands at **0.305** (+0.025 lift over the SFT baseline 0.280). The gap between the buggy callback's 0.354 and the true audit's 0.305 is the empirical measure of FM4: the audit-OOD bug masks ALL training progress, not just SFT, making the broken callback a non-signal regardless of training stage. Source: [`data/grpo_p3_run.log`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/grpo_p3_run.log) + [`data/grpo_p3_curves.json`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/grpo_p3_curves.json) + [`data/audit_grpo_p3_F_n5.json`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/audit_grpo_p3_F_n5.json) via [`scripts/make_grpo_curve_figure.py`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/scripts/make_grpo_curve_figure.py).*

### Pre-onsite training trace (Qwen3-0.6B + Kaggle T4)

![Qwen3-0.6B training reward vs red-team attack ceilings](https://raw.githubusercontent.com/akashkathole7/OpenEnv/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/figures/three_scales_reward.png)

*Qwen3-0.6B + LoRA + GRPO on Kaggle T4. Per-step training reward is bimodal around 0.174 (rollouts emitting a tool call) and 0.107 (clipped rollouts), reflecting the 0.25 tool-call frequency at this scale. 150-step eval composite pins at 0.353, within 0.004 of the `query_only` red-team attack signature (0.349). Pure GRPO drifts the policy into attack-signature territory without SFT warm-start. 1.7B and 4B runs were attempted on Kaggle but the logs were lost across session resets; see [LESSONS_LEARNED.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/LESSONS_LEARNED.md).*

![Per-component R1/R2/R3/R4 breakdown: trained 0.6B vs top red-team attacks](https://raw.githubusercontent.com/akashkathole7/OpenEnv/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/figures/three_scales_components.png)

*Defense-in-depth visualized: trained Qwen3-0.6B pins R1 and R2 at clamp floor 0.01, same cells as `submit_all_matched`. Each red-team attack pins a different component subset; no single-component exploit clears the 0.45 composite ceiling, and the CI test battery fails any future reward change that would.*

![Qwen3-0.6B 10-step SFT smoke run loss trajectory](https://raw.githubusercontent.com/akashkathole7/OpenEnv/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/figures/loss_curve.png)

*Qwen3-0.6B 10-step SFT smoke run loss trajectory; full training logs in [data/training_log_qwen3_\*_partial.json](https://github.com/akashkathole7/OpenEnv/tree/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data). TRL GRPO policy-gradient-style loss oscillates near zero (advantage-weighted log-prob deltas), with the step-8 excursion reflecting a high-variance rollout batch. Source: [data/smoke_test_10step.json](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/smoke_test_10step.json).*

---

## How to try

### Right here in this Space (no install)

Scroll past this README → **"Circular-Ring Viewer"** tab → pick a hero seed → click **Render**. Red nodes are GSTINs detected as members of a directed 3-cycle ring, the env's structural fraud signature.

### Locally (~2 minutes)

```bash
git clone -b scaffold/reconcile-gst2b https://github.com/akashkathole7/OpenEnv.git
cd OpenEnv
uv sync --all-extras
PYTHONPATH=src:envs uv run pytest tests/envs/test_reconcile_gst2b_*.py -v
```

Expected: 42 tests pass in ~3 seconds, including the 6 red-team attack ceilings.

---

## Deep dive (reviewer matrix)

| You are | Read this |
|---|---|
| Screener with 3-5 minutes | This README + the Day 1 baseline-comparison chart above |
| Reviewer with 10 minutes | [JUDGE_TOUR.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/JUDGE_TOUR.md) |
| Reviewer with 30 minutes | [BLOG.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/BLOG.md) §6 closing block (Day 1 outcome) + [ROUND2_PROBLEM_STATEMENT.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/ROUND2_PROBLEM_STATEMENT.md) |
| Reviewer reproducing the trained number | "Verify the trained number" section below + [`scripts/audit_sft_rollout_quality.py`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/scripts/audit_sft_rollout_quality.py) + [`data/audit_F_n5.json`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/audit_F_n5.json) |
| Researcher | [rewards.py](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/rewards.py) + [tests/envs/](https://github.com/akashkathole7/OpenEnv/tree/scaffold/reconcile-gst2b/tests/envs) |
| Fellow finalist | [LESSONS_LEARNED.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/LESSONS_LEARNED.md) §1, all 5 documented failure modes including FM4 (audit-OOD trap chain) and FM5 (reward-landscape inversion) from Day 1 on-site |

---

## Environment design (below the fold)

**`ReconcileAction`**, typed Pydantic action with 16 verbs:
- **Query (7, read-only):** `get_schema`, `list_gstins`, `fuzzy_search_gstin`, `get_invoice`, `list_invoices_by_supplier`, `get_2b_row`, `get_hsn_slab`
- **Mutate (7, state-changing):** `mark_matched`, `mark_mismatched`, `mark_only_in_books`, `mark_only_in_2b`, `mark_partial_match`, `flag_circular_ring`, `request_amendment`
- **Meta (2):** `confirm_with_user`, `submit`

**`ReconcileObservation`**: `invoices_remaining_count`, `step_budget`, `last_tool_result`, `user_request`. **Hidden ground truth is a first-class invariant**: zero `true_*` / `gt_*` fields, unit-tested to never leak.

**Episode:** 20-100 invoices, 50-step action budget. Deterministic in seed. `make reproduce` is bit-identical across processes.

**RLVE-aligned:** `generate_episode(seed)` procedurally generates fresh episodes over an unbounded seed space; the 20-100 invoice range and warmup vs hardened mode give natural curriculum difficulty scaling; reward is verifier-based (arithmetic, no learned reward model).

**Five mismatch types** with weights `(1, 1, 1, 3, 3)`:
- `gstin_typo`, 1-char PAN edit with recomputed Luhn-like checksum
- `invoice_number_prefix_drift`, `INV/24-25/001` vs `INV-24-25-001`
- `tax_slab_off_by_one`, 18% vs 5% / 40% (post-GST-2.0 wider gaps)
- `supplier_late_filing`, present in books, absent from 2B
- `amendment_after_2b_freeze`, value edited after the 2B snapshot

**Circular-trading rings:** with probability 0.3, a directed 3-cycle A→B→C→A is planted across three invoices' counterparty GSTINs. Detectable only via `networkx.simple_cycles` over the supplier graph, not via naive GSTIN reassignment. The Ring Viewer tab in this Space shows this live.

---

## Reward design

Four components, every one clamped to `[0.01, 0.99]`:

| Component | Weight | Formula | Defends against |
|---|---:|---|---|
| **R1 reconciliation_f1** | 0.40 | Macro-F1 over 5 labels | "label all matched" accuracy trick |
| **R2 itc_delta_accuracy** | 0.25 | `1 − min(1, |claimed − true| / max(true, 1))` | 2× over-claim AND zero-claim both tank to 0 |
| **R3 rule_36_4_compliance** | 0.25 | `0.99` iff per-supplier cap honored AND ≥1 query verb called | `confirm_spam`, mutate without inspecting |
| **R4 step_efficiency** | 0.10 | `1 − (steps / 50)²` | Query-flooding to game budget |

Plus a **structural `−1.0` penalty** (not clamped) for `submit` before any query verb.

### Red-team CI battery (2026-04-24 recompute)

Every PR runs [`test_reconcile_gst2b_reward_hacking.py`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/tests/envs/test_reconcile_gst2b_reward_hacking.py). Six attacks, all asserted `<0.45`:

| Attack | Score | Caught by |
|---|---:|---|
| `submit_all_matched` | 0.308 | R3 overclaim + R4 budget |
| `submit_all_mismatched` | 0.266 | R1 (1-of-5 classes) + R2 (zero claim) |
| `confirm_spam` | 0.010 | R3 (no query) + R4 (budget) |
| `zero_itc` | 0.266 | R2 (claim vs true) |
| `query_only` | **0.349** | R1 + R2 (no mutate) |
| `overflag_rings` | 0.283 | R1 + R2 (no mutate) |

Any reward-function change that lifts any attack above 0.45 fails CI. The test is a permanent guardrail against reward-design drift.

---

## Pre-onsite training evidence

Qwen3-0.6B + LoRA rank 16 + GRPO on Kaggle T4. 10-step smoke + 150-step eval, plotted above. Trained policy plateaus at 0.353, within 0.004 of the `query_only` red-team attack signature. Pure GRPO without SFT warm-start drifts into attack-signature territory at this scale.

Two additional attempts on Kaggle (Qwen3-1.7B 15-step partial, Qwen3-4B step-0 eval) were observed live with the reported qualitative signatures (1.7B: zero tool-call entropy collapse; 4B: budget-exhausting over-query). The corresponding JSON logs were lost across Kaggle session resets and were never committed. See [LESSONS_LEARNED.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/LESSONS_LEARNED.md) §1 for the honest diagnosis and what changed for the on-site run.

## Day 1 on-site outcome (2026-04-25, A100 SXM4-80GB)

Pipeline that ran:

1. **Phase 1 trajectory verification** (5 s, CPU). 3000-row balanced trajectory file (1000 oracle + 1000 inspect_then_label + 1000 supplier_cap_aware, round-robin interleaved) verified against the env. Mean total 0.690, mean R3 0.791.
2. **Phase 2 SFT** (31:12, A100). [`train_sft_warmstart.py`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/scripts/train_sft_warmstart.py) on Qwen/Qwen3-4B + LoRA rank 16 + bf16 + 1 epoch over 3000 rows = 375 optimizer steps. Final aggregate train_loss 0.341 (final per-step 0.207, monotonic descent). Full log: [`data/sft_full_run.log`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/sft_full_run.log).
3. **5-metric rollout audit** on held-out seeds 9030-9034 at GRPO-matching sampling. After resolving a 4-bug audit-OOD trap chain (Failure Mode 4), n=5 mean composite reward **0.280** with 3/5 seeds emitting 32-46 mark verbs per trajectory. Diagnostic tool: [`scripts/audit_sft_rollout_quality.py`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/scripts/audit_sft_rollout_quality.py). Headline data: [`data/audit_F_n5.json`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/audit_F_n5.json).
4. **Phase 3 GRPO deferred.** Reward-landscape inversion (Failure Mode 5): the `query_only` attack at 0.353 scores ABOVE the trained Mode A marking trajectories at max 0.26. GRPO advantage would push toward the attack. Resolution requires either modifying the frozen `rewards.py` (Guardrail 1) or multi-hour reward-shaping validation in the trainer.

Two failure modes surfaced past Phase 2 and are the substantive findings of Day 1 (full text in [LESSONS_LEARNED.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/LESSONS_LEARNED.md) §1 + [BLOG.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/BLOG.md) §6 closing block):

- **Failure Mode 4: audit-script OOD trap chain.** Four sequential bugs in audit infrastructure (greedy decoding, fresh-context prompts, format false-alarm, missing `tools=` parameter) each fabricated a different false collapse signature before real model behavior could be observed. Lesson: diagnostic infrastructure is as load-bearing as the model itself.
- **Failure Mode 5: reward-landscape inversion at Phase 2 → Phase 3 boundary.** The same arithmetic-reward + multi-component-clamping design that makes the 6 red-team attacks defensible (the central claim of this submission) also creates an exploit-favoring asymmetry at this policy quality. Documented as the post-hackathon agenda.

## Verify the trained number (judges' reproducibility path)

Day 1 headline `n=5 mean 0.280` is reproducible end-to-end from these artifacts:

| Artifact | Purpose |
|---|---|
| [`scripts/audit_sft_rollout_quality.py`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/scripts/audit_sft_rollout_quality.py) | The diagnostic tool that produced the n=5 mean. |
| [`data/sft_full_run.log`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/sft_full_run.log) | Full 375-step SFT training log, 31:12 wall. |
| [`data/sft_summary.json`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/sft_summary.json) | Trained checkpoint metadata. |
| [`data/sft_trajectories.jsonl`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/sft_trajectories.jsonl) | The 3000-row balanced SFT input. |
| [`data/audit_F_n5.json`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/audit_F_n5.json) | Headline n=5 audit at step 375 final. Per-seed totals + reward breakdown + raw model completions. |
| [`data/audit_ckpt350_F_n5.json`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/audit_ckpt350_F_n5.json) | Same audit at step 350 (25 steps before final). Empirical FM5 proof: 0.314 mean comes from 4/5 Mode B collapse, while step 375 trades reward for marking. |
| [`data/audit_seeds_9031_9034.json`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/audit_seeds_9031_9034.json) | Pre-`tools=` collapse evidence (Failure Mode 4 audit-OOD chain). |
| [`data/audit_step100_n5.json`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/audit_step100_n5.json) | Early-stop control ruling out the over-training hypothesis. |
| [`data/audit_grpo_p3_F_n5.json`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/audit_grpo_p3_F_n5.json) | **Headline post-GRPO audit (n=5 mean 0.305).** SFT-merged + 100 GRPO steps with Tier 2c length-shaping bonus. |
| [`data/grpo_p3_run.log`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/grpo_p3_run.log) | 100-step GRPO training log with per-step rewards. |
| [`data/grpo_p3_curves.json`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/data/grpo_p3_curves.json) | Buggy eval callback's flat trace, kept as empirical FM4 confirmation across training stages. |

---

## OpenEnv alignment

- ✅ Latest OpenEnv release
- ✅ TRL GRPO training script (`environment_factory` pattern)
- ✅ Hosted live on Hugging Face Spaces (this page)
- ✅ `openenv validate --verbose` passes
- ✅ Hidden ground truth invariant (unit-tested via `test_state_hides_ground_truth_from_observation`)
- ✅ Rewards inside the environment (not external)
- ✅ Client-server separation enforced
- ✅ `SUPPORTS_CONCURRENT_SESSIONS = True` (regression-tested)
- ✅ 42-test suite, including red-team CI battery
- ✅ Reward follows OpenEnv's composable-rubrics pattern in spirit: each R1-R4 is an independent clamped function, `composite_reward()` composes them with weights + per-component audit breakdown

---

## Submission context

- **Author**: Aakash Kathole ([@akashkathole7](https://github.com/akashkathole7)), solo finalist
- **Event**: Meta × Scaler Hackathon, Bangalore 2026 (on-site Apr 25-26)
- **Round 2 theme**: #3.1 Professional Tasks / World Modeling
- **Sub-theme (bonus prize target)**: Scaler AI Labs, Multi-App RL Environment for Enterprise Workflows

## License

MIT, see [LICENSE](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/LICENSE). Upstream OpenEnv is BSD-3-Clause.

## Citation

```bibtex
@software{kathole2026reconcile_gst2b_env,
  title   = {reconcile_gst2b_env: An OpenEnv environment for multi-turn enterprise compliance workflows},
  author  = {Kathole, Aakash},
  year    = {2026},
  url     = {https://github.com/akashkathole7/OpenEnv/tree/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env},
  note    = {Meta × Scaler Hackathon Bangalore 2026, targeting Scaler AI Labs Enterprise Workflow sub-theme}
}
```
