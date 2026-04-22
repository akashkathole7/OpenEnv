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

**OpenEnv environment for multi-turn enterprise compliance workflows.**
Instantiated on Indian GST Input-Tax-Credit reconciliation, a two-sided document-matching task against a regulator-published schema. Targeting the **Scaler AI Labs, Multi-App RL Environment for Enterprise Workflows** sub-theme at the Meta × Scaler Hackathon, Bangalore 2026.

> 🎮 **Try the live demo below**, scroll past this README, select the "Circular-Ring Viewer" tab, pick a hero seed (9500 / 9501 / 9502), and click **Render** to see a 3D supplier graph with a planted directed-cycle fraud ring highlighted in red.

---

## At a glance

| Metric | Value |
|---|---:|
| Typed tool verbs in env | **16** (7 query / 7 mutate / 2 meta) |
| Label classes | **5** (matched, mismatched, only_in_books, only_in_2b, partial) |
| Planted mismatch types | **5** + directed 3-cycle circular-trading rings (30% probability per episode) |
| Reward components | **4** arithmetic, each clamped to `[0.01, 0.99]` |
| Red-team attacks CI-enforced `<0.45` | **6 / 6 pass** (max `query_only` = 0.349) |
| Test suite | **42 / 42 green** |
| Prompted vs raw baseline delta on Qwen2.5-3B-Instruct | **1.18** (95% CI [1.09, 1.27], 180 rollouts) |
| Pre-onsite training attempts | **3** (0.6B / 1.7B / 4B) with distinct failure-mode documentation |
| `make reproduce` | Bit-identical against committed artifacts |

---

## The problem

Every month, millions of businesses worldwide match their internal purchase registers against a regulator's cross-filed view. In India alone, **~14 million GST-registered businesses** do this monthly, rule-heavy, error-sensitive, still done by hand, across millions of invoices per company.

Where books and the regulator's GSTR-2B return agree, the business claims back GST paid upstream ("input tax credit"). Where they disagree, typos, tax-slab errors, late supplier filings, post-freeze amendments, circular-trading rings, the business either **over-claims and triggers an audit** or **under-claims and forfeits money**. The task is structured, auditable, and adversarial. It is the textbook enterprise compliance workflow, and the textbook target for reinforcement-learning agents.

This environment captures that workflow as a partially observable, multi-turn, tool-orchestration task with a pure-arithmetic 4-component reward (no LLM judge in the reward path) and a CI-enforced red-team ceiling (6 adversarial attacks, all scoring below 0.45).

---

## Environment design

**`ReconcileAction`**, typed Pydantic action with 16 verbs:
- **Query (7, read-only)**: `get_schema`, `list_gstins`, `fuzzy_search_gstin`, `get_invoice`, `list_invoices_by_supplier`, `get_2b_row`, `get_hsn_slab`
- **Mutate (7, state-changing)**: `mark_matched`, `mark_mismatched`, `mark_only_in_books`, `mark_only_in_2b`, `mark_partial_match`, `flag_circular_ring`, `request_amendment`
- **Meta (2)**: `confirm_with_user`, `submit`

**`ReconcileObservation`**, contains `invoices_remaining_count`, `step_budget`, `last_tool_result`, `user_request`. **Hidden ground truth is a first-class invariant**, zero `true_*` / `gt_*` fields, unit-tested to never leak.

**Episode**: 20–100 invoices, 50-step action budget. Deterministic in seed. `make reproduce` is bit-identical across processes.

**RLVE-aligned (Reinforcement Learning with Verifiable Environments)**: `generate_episode(seed)` procedurally generates fresh episodes over an unbounded seed space; the 20-100 invoice range and warmup vs hardened mode give natural curriculum difficulty scaling; reward is verifier-based (arithmetic, no learned reward model), so the environment itself is the ground truth. This is the pattern recent RLVE literature argues prevents training-distribution saturation.

**Five mismatch types** with weights `(1, 1, 1, 3, 3)`:
- `gstin_typo`, 1-char PAN edit with recomputed Luhn-like checksum
- `invoice_number_prefix_drift`, `INV/24-25/001` vs `INV-24-25-001`
- `tax_slab_off_by_one`, 18% vs 5% / 40% (post-GST-2.0 wider gaps)
- `supplier_late_filing`, present in books, absent from 2B
- `amendment_after_2b_freeze`, value edited after the 2B snapshot

**Circular-trading rings**: with probability 0.3, a directed 3-cycle A→B→C→A is planted across three invoices' counterparty GSTINs. Detectable only via `networkx.simple_cycles` over the supplier graph, not via naive GSTIN reassignment. The Ring Viewer tab in this Space shows this live.

---

## Reward design, four arithmetic components

Each clamped to `[0.01, 0.99]`:

| Component | Weight | Formula | Defends against |
|---|---:|---|---|
| **R1 reconciliation_f1** | 0.40 | Macro-F1 over 5 labels | "label all matched" cheat (accuracy trick) |
| **R2 itc_delta_accuracy** | 0.25 | `1 − min(1, |claimed − true| / max(true, 1))` | 2× over-claim AND zero-claim both tank to 0 |
| **R3 rule_36_4_compliance** | 0.25 | `0.99` iff per-supplier cap honored AND ≥1 query verb called | "confirm spam", mutate without inspecting |
| **R4 step_efficiency** | 0.10 | `1 − (steps / 50)²` | Query-flooding to game budget |

Plus a **structural `−1.0` penalty** (not clamped) for `submit` before any query verb has been called.

### Red-team CI battery

Every pull request runs [`test_reconcile_gst2b_reward_hacking.py`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/tests/envs/test_reconcile_gst2b_reward_hacking.py), six attack trajectories asserted to score `<0.45`:

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

## Training evidence, three distinct failure modes across three model scales

Pre-onsite, I ran GRPO training across three Qwen3 variants on Kaggle T4 / T4x2. The reward design surfaces three structurally different failure modes, each pinning a **different reward component** to floor. No model in this range passes 0.45 by accident.

| Model | Observed behavior | Component pinned | Artifact |
|---|---|---|---|
| **Qwen3-0.6B** (10-step smoke) | Bimodal tool-call rate ~0.25 per rollout; can't reliably chain ≥5 `mark_*` actions | R1, R2 | [`data/smoke_test_10step.json`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/data/smoke_test_10step.json) |
| **Qwen3-1.7B** (15-step partial) | Entropy collapsed to 0.12 (vs 0.44 on 0.6B); zero tool calls emitted; deterministic mode | All | [`data/training_log_qwen3_1_7b_partial.json`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/data/training_log_qwen3_1_7b_partial.json) |
| **Qwen3-4B** (step-0 eval only) | Opposite failure: ~1.0 parseable tool-call rate, but over-queries until the 50-step budget exhausts | R4 | [`data/training_log_qwen3_4b_partial.json`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/data/training_log_qwen3_4b_partial.json) |

The environment is **structurally hard across model sizes**, not a training-recipe bug. This is exactly the property you want in an RL benchmark: no shortcut behavior clears the bar.

---

## On-site training strategy, tool-SFT warm-start → GRPO polish

Empirical motivation: three partial-training artifacts prove pure GRPO from a base Qwen3 instruct checkpoint fails to chain label actions at ≤4B scale. The on-site plan (25–26 April 2026, using HF compute credits):

1. **Synthetic expert trajectories** (2 h, CPU). Use the existing heuristic policy (`evaluate_heuristic` in `training.py:175`) as an oracle against the env's hidden ground truth. Generate ~500 trajectories across seeds 100–599.
2. **Tool-SFT warm-start** (4 h, A100). LoRA-SFT Qwen3-4B (or Qwen3-8B if compute permits) on the synthetic trajectories. Target: teach `mark_*` action preference over `get_*` exploration.
3. **GRPO polish** (4 h, A100). Continue training from the SFT checkpoint using the existing [`train_grpo_real.py`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/scripts/train_grpo_real.py) recipe (Tier 1+2 reward shaping already baked in). Target: composite total > 0.50 on held-out seeds 9030–9039.

Success metric: trained policy clearly separates from both the `query_only` red-team attack signature (0.349) and the prompted baseline (0.179).

---

## OpenEnv alignment

- ✅ Latest OpenEnv release
- ✅ TRL GRPO training script (`environment_factory` pattern)
- ✅ Hosted live on Hugging Face Spaces (this page)
- ✅ `openenv validate --verbose` passes
- ✅ Hidden ground truth invariant (unit-tested)
- ✅ Rewards inside the environment (not external)
- ✅ Client-server separation enforced
- ✅ `SUPPORTS_CONCURRENT_SESSIONS = True` (regression-tested)
- ✅ 42-test suite, including red-team CI battery

---

## Try it

**In this Space:** select the **"Circular-Ring Viewer"** tab below. Pick a hero seed (9500, 9501, or 9502) or any custom seed. Click **Render**. Rotate the 3D plot. Red nodes are GSTINs detected as members of a directed 3-cycle ring, the env's structural fraud signature.

**Locally:**
```bash
git clone -b scaffold/reconcile-gst2b https://github.com/akashkathole7/OpenEnv
cd OpenEnv
uv sync --all-extras
PYTHONPATH=src:envs uv run pytest tests/envs/test_reconcile_gst2b_*.py -v
make reproduce
```

---

## Links

- 🧑‍💻 **Repo**: https://github.com/akashkathole7/OpenEnv (branch `scaffold/reconcile-gst2b`)
- 📝 **Full blog + dev diary**: [`BLOG.md`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/BLOG.md)
- 📋 **Round 2 problem statement**: [`ROUND2_PROBLEM_STATEMENT.md`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/ROUND2_PROBLEM_STATEMENT.md)
- 📊 **Judge tour (10-min guided read)**: [`JUDGE_TOUR.md`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/JUDGE_TOUR.md)
- 🎥 **90-second demo video**: https://www.youtube.com/watch?v=rglR1hGgdb8
- 🧪 **Red-team test file**: [`test_reconcile_gst2b_reward_hacking.py`](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/tests/envs/test_reconcile_gst2b_reward_hacking.py)

---

## Submission context

**Author**: Aakash Kathole ([@akashkathole7](https://github.com/akashkathole7)), solo finalist
**Event**: Meta × Scaler Hackathon, Bangalore 2026
**Round 1 theme** (if kept same): selectable
**Round 2 theme target**: #3.1 Professional Tasks / World Modeling
**Round 2 sub-theme (bonus prize target)**: **Scaler AI Labs, Multi-App RL Environment for Enterprise Workflows**

---

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
