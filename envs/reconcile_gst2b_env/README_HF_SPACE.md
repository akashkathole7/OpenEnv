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

---

## TL;DR: what landed against the judging rubric

- **Environment innovation (40%):** 16 typed verbs (7 query / 7 mutate / 2 meta), 5 planted mismatch types, 30%-probability directed 3-cycle ring fraud, partially observable, hidden ground truth unit-tested to never leak into observations.
- **Storytelling (30%):** BLOG, 90-sec video, PITCH, QA_REHEARSAL, this Space with the 3D ring viewer you see below, README front-loads plots per judges' guidance.
- **Training evidence (20%):** Pre-onsite Qwen3-0.6B GRPO 10-step smoke + 150-step eval plateau (plots below). On-site Day 1 (2026-04-25, A100 SXM4-80GB): full 375-step Qwen3-4B SFT, n=5 mean composite reward **0.280** above prompted Qwen2.5-3B baseline 0.18. GRPO Phase 3 deferred per reward-landscape analysis. [LESSONS_LEARNED.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/LESSONS_LEARNED.md) §1 covers all 5 documented failure modes.
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
| **Trained Qwen3-4B SFT on A100 SXM4-80GB (Day 1 on-site)** | **n=5 mean composite reward 0.280** |
| Prompted vs raw baseline delta on Qwen2.5-3B-Instruct | 1.18 (95% CI [1.09, 1.27], 180 rollouts) |
| Trained-vs-prompted lift | +0.10 (0.280 − 0.18) |
| Documented pre-onsite training run | Qwen3-0.6B, 10-step smoke + 150-step eval (plots below) |
| `make reproduce` | bit-identical against committed artifacts |

---

## Why it matters (30-second hook)

Every registered Indian business with B2B purchases reconciles its purchase register against the regulator-generated GSTR-2B return monthly. It's rule-heavy, error-sensitive (over-claim triggers audit; under-claim forfeits money), still done by hand by junior accountants, and the regulator publishes the correctness rules. That makes it a cleaner RL target than "general web agent": wrong answers have crisp, auditable, regulator-specified correctness. An agent that clears this generalizes to every two-sided document reconciliation against a schema.

---

## Results

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
| Screener with 3-5 minutes | This README + the two plots above |
| Reviewer with 10 minutes | [JUDGE_TOUR.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/JUDGE_TOUR.md) |
| Reviewer with 30 minutes | [BLOG.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/BLOG.md) + [ROUND2_PROBLEM_STATEMENT.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/ROUND2_PROBLEM_STATEMENT.md) |
| Researcher | [rewards.py](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/rewards.py) + [tests/envs/](https://github.com/akashkathole7/OpenEnv/tree/scaffold/reconcile-gst2b/tests/envs) |
| Fellow finalist | [LESSONS_LEARNED.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/LESSONS_LEARNED.md) |

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

Two additional attempts on Kaggle (Qwen3-1.7B 15-step partial, Qwen3-4B step-0 eval) were observed live with the reported qualitative signatures (1.7B: zero tool-call entropy collapse; 4B: budget-exhausting over-query). The corresponding JSON logs were lost across Kaggle session resets and were never committed. See [LESSONS_LEARNED.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/LESSONS_LEARNED.md) for the honest diagnosis and what changes for the on-site A100 run.

## On-site training strategy (Apr 25-26, Scaler HF compute credits)

1. **Synthetic expert trajectories** (2 h, CPU). Use the ground-truth-aware heuristic policy as an oracle; 3 policy variants already staged in [`scripts/_policies.py`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/scripts/_policies.py).
2. **Tool-SFT warm-start** (4 h, A100). LoRA-SFT Qwen3-4B (or 1.7B fallback) on the 3000-row balanced union. Target: teach `mark_*` action preference over `get_*` exploration.
3. **GRPO polish** (4 h, A100). Continue training from the SFT checkpoint using [`train_grpo_real.py`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/scripts/train_grpo_real.py) (Tier 1+2 reward shaping already baked in). Target: composite total > 0.45 on held-out seeds 9030-9039, clearly separating from both the `query_only` attack (0.349) and the prompted baseline (0.179).

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
