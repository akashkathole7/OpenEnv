---
title: reconcile_gst2b_env
emoji: 🥊
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# reconcile_gst2b_env: GST Input-Tax-Credit Reconciliation

**An OpenEnv environment for the monthly GSTR-2B reconciliation that ~14M Indian businesses perform by hand. 16 typed tool verbs, 4-component arithmetic reward, 6 red-team attacks all scoring under a CI-enforced 0.45 ceiling.**

🎥 [90-second demo video](https://www.youtube.com/watch?v=rglR1hGgdb8) · 🚀 [Live HF Space](https://huggingface.co/spaces/akashkathole/reconcile_gst2b_env) · 📝 [BLOG.md](BLOG.md) · 📋 [ROUND2_PROBLEM_STATEMENT.md](ROUND2_PROBLEM_STATEMENT.md) · 🧪 [JUDGE_TOUR.md](JUDGE_TOUR.md)

---

## TL;DR: what landed against the judging rubric

- **Environment innovation (40%):** 16 typed verbs (7 query / 7 mutate / 2 meta), 5 planted mismatch types, 30%-probability directed 3-cycle ring fraud, partially observable, hidden ground truth unit-tested to never leak into observations.
- **Storytelling (30%):** [BLOG.md](BLOG.md), 90-sec video, [PITCH.md](PITCH.md), [QA_REHEARSAL.md](QA_REHEARSAL.md), live HF Space with 3D ring viewer, this README front-loads plots per judges' guidance.
- **Training evidence (20%):** 10-step Qwen3-0.6B GRPO smoke + 150-step eval plateau documented with per-step metrics. Trained numbers from on-site A100 (Apr 25-26) land here as a follow-up commit. See [LESSONS_LEARNED.md](LESSONS_LEARNED.md) for the honest multi-scale attempt diagnosis.
- **Reward & pipeline (10%):** 4-component arithmetic reward clamped to `[0.01, 0.99]`, 6 red-team attacks CI-enforced at `<0.45`, 42 tests green, Tier 1+2 GRPO fixes validated in [`data/smoke_test_10step.json`](data/smoke_test_10step.json).

---

## Why it matters (30-second hook)

Every registered Indian business with B2B purchases reconciles its purchase register against a regulator-generated return (GSTR-2B) monthly. It's rule-heavy, error-sensitive (over-claim triggers audit; under-claim forfeits money), still done by hand by junior accountants, and the regulator publishes the correctness rules. That makes it a cleaner RL target than "general web agent": wrong answers have crisp, auditable, regulator-specified correctness. An agent that clears this generalizes to every two-sided document reconciliation against a schema.

In India alone, ~14M GST-registered businesses run this loop monthly. The task is **rule-bound** (regulator-published), **multi-turn** (dozens of tool calls per invoice set), **adversarially shaped** (over-claim, under-claim, and zero-claim attack shapes are all financially motivated in the real world), and **partially observable** (ground truth hidden from the agent).

---

## Results

![Qwen3-0.6B training reward vs red-team attack ceilings](data/figures/three_scales_reward.png)

*Qwen3-0.6B + LoRA + GRPO on Kaggle T4: per-step training reward is bimodal around 0.174 (rollouts that emit a tool call) and 0.107 (clipped rollouts), reflecting the 0.25 tool-call frequency observed at this scale. 150-step eval composite pins at 0.353, within 0.004 of the `query_only` red-team attack signature (0.349). Pure GRPO without SFT warm-start drifts the trained policy into attack-signature territory. Source: [data/smoke_test_10step.json](data/smoke_test_10step.json), [data/curves_150step_backup.json](data/curves_150step_backup.json). 1.7B and 4B runs were also attempted pre-onsite on Kaggle but the logs were lost across session resets; [LESSONS_LEARNED.md](LESSONS_LEARNED.md) documents what we observed live.*

![Per-component R1/R2/R3/R4 breakdown: trained 0.6B vs top red-team attacks](data/figures/three_scales_components.png)

*The defense-in-depth reward contract visualized. Trained Qwen3-0.6B pins R1 (reconciliation_f1) and R2 (itc_delta_accuracy) at the clamp floor 0.01: structurally the same cells as `submit_all_matched`. Because each red-team attack pins a different subset of components, no single-component exploit clears the 0.45 composite ceiling, and the CI test battery fails any future reward change that would.*

### Headline numbers

| Metric | Value |
|---|---:|
| Red-team attacks under CI-enforced 0.45 ceiling | 6 of 6, max = 0.349 (`query_only`) |
| Test suite | 42 / 42 green |
| Prompted Qwen2.5-3B baseline delta | **1.18** (95% CI [1.09, 1.27], 180 rollouts) |
| Reward-distribution gradient | 81 distinct totals, σ=0.50, 100% done-rate across 100 random-policy episodes |
| `make reproduce` | bit-identical against committed artifacts |

---

## How to try (pick one)

### In the HF Space (no install, ~10 seconds)

[huggingface.co/spaces/akashkathole/reconcile_gst2b_env](https://huggingface.co/spaces/akashkathole/reconcile_gst2b_env) → scroll past the README → **"Circular-Ring Viewer"** tab → pick a hero seed (9500 easy / 9501 medium / 9502 hard) → click **Render** → see the 3D supplier graph with planted directed-cycle fraud ring highlighted in red.

### Locally (~2 minutes)

```bash
git clone -b scaffold/reconcile-gst2b https://github.com/akashkathole7/OpenEnv.git
cd OpenEnv
uv sync --all-extras
PYTHONPATH=src:envs uv run pytest tests/envs/test_reconcile_gst2b_*.py -v
```

Expected: 42 tests pass in ~3 seconds, including the 6 red-team attack ceilings.

### Reproduce the plots above

```bash
PYTHONPATH=src:envs uv run python -m \
    envs.reconcile_gst2b_env.scripts.make_training_figures
# writes data/figures/three_scales_{reward,components}.png
```

---

## Deep dive (reviewer matrix)

| You are | Read this |
|---|---|
| Screener with 3-5 minutes | This README + the two plots above |
| Reviewer with 10 minutes | [JUDGE_TOUR.md](JUDGE_TOUR.md) (guided repo walk) |
| Reviewer with 30 minutes | [BLOG.md](BLOG.md) + [ROUND2_PROBLEM_STATEMENT.md](ROUND2_PROBLEM_STATEMENT.md) + the two plots |
| Researcher | [PRD.md](PRD.md) + [rewards.py](rewards.py) + [tests/envs/test_reconcile_gst2b_*.py](../../tests/envs/) |
| Fellow finalist | [LESSONS_LEARNED.md](LESSONS_LEARNED.md) (what didn't work on Kaggle, and why the submission ships anyway) |

---

## Reward design (arithmetic, no LLM judge)

Every component is clamped to `[0.01, 0.99]`. Boundary values (0.0 / 1.0) fail many public validators by design.

| ID | Component | Weight | Formula | Why this shape |
|----|-----------|-------:|---------|-----|
| R1 | reconciliation_f1 | 0.40 | Macro-F1 over 5 labels | Macro (not accuracy): "label all matched" caps around 0.17 because 4 of 5 per-class F1 scores go to zero. |
| R2 | itc_delta_accuracy | 0.25 | `1 − min(1, |claimed − true| / max(true, 1))` | Absolute (not relative): "claim 0 ITC to dodge Rule 36(4)" and "2× over-claim" both tank R2 to 0.01. |
| R3 | rule_36_4_compliance | 0.25 | `0.99` iff per-supplier claim ≤ 2B cap AND ≥1 query verb called, else `0.01` | Query prerequisite blocks `confirm_spam` (mutate-without-inspect). Per-supplier (not aggregate) catches `supplier_late_filing` overclaims. |
| R4 | step_efficiency | 0.10 | `1 − (steps / 50)²` | Quadratic: penalty spikes near budget, early steps score ≈ 1.0. Discourages exhaustive exploration without rewarding premature submit. |

Plus a **structural `−1.0` penalty** (not clamped) for `submit` before any query verb. Submit after `≥1 query` returns the composite. Step budget 50 exhausted also returns the composite. Hardened mode terminates mid-episode on any Rule 36(4) violation.

## Anti-gaming red-team battery (CI-enforced)

Every pull request runs [`test_reconcile_gst2b_reward_hacking.py`](../../tests/envs/test_reconcile_gst2b_reward_hacking.py). Six adversarial trajectories, all asserted `<0.45`. Measured on seed 0, re-computed 2026-04-24:

| Attack | Total | Caught primarily by | Secondary defense |
|--------|------:|---------------------|-------------------|
| `submit_all_matched`   | 0.308 | R3 per-supplier overclaim | R4 budget exhaustion |
| `submit_all_mismatched`| 0.266 | R1 (4 of 5 class F1 at 0) | R2 (zero claim vs true) |
| `confirm_spam`         | 0.010 | R3 (no query)             | R4 (budget), R1 (no mutate) |
| `zero_itc`             | 0.266 | R2 (claim-vs-true delta) | R3 false positive by design, documented |
| `query_only`           | 0.349 | R1 (no labels)            | R2 (no claim) |
| `overflag_rings`       | 0.283 | R1 (no mark labels)       | R2 (no claim) |

No single-component exploit clears the composite ceiling: each attack is caught by at least two independent components. Any future reward change that lifts any of these above 0.45 fails CI. This is the single strongest anti-reward-hacking guardrail in the submission.

## Differentiation

| Env | Task class | Correctness source | Reward signal | Adversarial hardening |
|-----|-----------|--------------------|---------------|----------------------|
| **reconcile_gst2b_env** (this) | Rule-bound accounting reconciliation | Indian GST regulation (regulator-published) | 4-component arithmetic, clamped | 6 red-team attacks CI-enforced, all `<0.45` |
| [Gaia2](https://huggingface.co/datasets/gaia-benchmark/GAIA) | Open-ended web / real-world questions | Human-verified answers | LLM-judge or exact-match | None built-in |
| [MEMTRACK](https://arxiv.org/abs/2410.08182) | Multi-turn memory consistency | Synthetic hand-designed | Trajectory-level match | Ontology-drift only, not adversarial |
| [AppWorld](https://appworld.dev/) | Multi-app tool use | Hand-scripted app states | Task completion booleans | None |
| [τ-bench](https://arxiv.org/abs/2406.12045) | Customer-service tool use | Simulated business rules | LLM-judge + rules | Limited |

What's distinct here:
1. **Reward is purely arithmetic and composite**, no LLM in the reward path, matches ARE paper §B.3.1 concerns about judge-driven reward hacking.
2. **Red-team ceiling enforced in tests**, 6 attacks × CI `<0.45`, failing the build if any crosses.
3. **Domain has crisp wrong answers**, over-claim ITC has a regulator, "general web agent" does not.
4. **Hidden ground truth is a first-class invariant**, `ReconcileObservation` has zero `true_*` / `gt_*` fields, verified by `test_state_hides_ground_truth_from_observation`.

## How our reward maps to OpenEnv Rubrics

OpenEnv's composable-rubrics philosophy says "one reward function is one brittle contract; multiple independent rubrics composed into a composite are auditable and defensible." Our [`rewards.py`](rewards.py) implements this pattern through composition rather than class inheritance:

- Each of R1-R4 is a pure function `(state, trajectory) -> float` with a weight and an independent clamp.
- `composite_reward()` composes them into a weighted sum with per-component breakdown persisted for audit.
- The red-team battery tests each component's independence (attack X tanks component Y, attack Z tanks component W).

We compose rewards as clamped, weighted independent components — the Rubric pattern in spirit if not yet in class-inheritance form. A v2 could wrap each component in an OpenEnv `Rubric` subclass for framework-native composition; functional behavior is identical.

## Scope fence (what I deliberately cut)

See [scope_card.md](scope_card.md). **IN**: B2B domestic supply of goods, GST 2.0 slabs, GSTR-2B matching. **OUT**: RCM, ISD, SEZ, imports, composition dealers, e-invoicing, e-way bill. **Synthetic**: HSN→slab mapping (labeled as such at table head). The env is deliberately narrow so the RL problem stays well-defined; regime expansion is a natural v2 (ISD first, RCM second).

## Citations

- **OpenEnv**: Meta PyTorch OpenEnv, https://github.com/meta-pytorch/OpenEnv
- **Qwen3**: https://huggingface.co/Qwen/Qwen3-1.7B
- **Qwen2.5-3B-Instruct** (prompted baseline): https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- **Unsloth**: https://github.com/unslothai/unsloth (drop-in ready for Phase 2/3 efficiency)
- **TRL / GRPO**: https://github.com/huggingface/trl
- **GSTN portal + Rule 36(4)**: https://www.gst.gov.in/ (Central Goods and Services Tax Rules, 2017)
- **ARE paper**, §B.3.1 on judge-driven reward hacking, cited as design rationale for arithmetic-only reward.
- **MEMTRACK**: Shen et al., 2024, https://arxiv.org/abs/2410.08182
- **τ-bench**: Yao et al., 2024, https://arxiv.org/abs/2406.12045

## License

MIT for this env contribution, see [LICENSE](LICENSE). Upstream OpenEnv is BSD-3-Clause at [../../LICENSE](../../LICENSE); both are compatible.
