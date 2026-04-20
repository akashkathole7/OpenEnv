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

# reconcile_gst2b_env — GST Input-Tax-Credit Reconciliation

> Scope fence: see [scope_card.md](scope_card.md). **IN**: B2B domestic supply of goods, GST 2.0 slabs, GSTR-2B matching. **OUT**: RCM, ISD, SEZ, imports, composition dealers, e-invoicing, e-way bill.

An OpenEnv environment for training an agent to reconcile a company's purchase register against its monthly **GSTR-2B** return: label each invoice across 5 categories, compute ITC delta in INR, flag Rule 36(4) violations, and detect planted circular-trading rings — all with a 16-verb tool surface and a 4-component arithmetic reward.

## Problem

Every registered Indian business with B2B purchases does this reconciliation monthly. It's rule-heavy, error-sensitive (over-claim → notice; under-claim → forfeit), and every CA firm pays juniors to do it by hand. The failure modes are well-cataloged (GSTIN typos, format drift, slab errors, supplier-late-filing, post-freeze amendments, circular trading) and the regulator publishes the ground truth. This is a cleaner RL target than "general web agent" because wrong answers have crisp, regulated correctness.

## Quick start

```bash
git clone -b scaffold/reconcile-gst2b https://github.com/akashkathole7/OpenEnv.git && cd OpenEnv
pip install -e . && pip install -e envs/reconcile_gst2b_env
PYTHONPATH=src:envs python -m envs.reconcile_gst2b_env.scripts.diversity_test
```

That runs the reward-diversity diagnostic (100 random-policy episodes × 2 modes); it writes `data/reward_dist.json` with 6 signal-quality criteria, all of which pass. For the video demo, `python envs/reconcile_gst2b_env/app.py` launches the 3D ring-viewer Gradio app on seed 9502.

## Reward design

4 components, all arithmetic, no LLM judge. Every component is clamped to `[0.01, 0.99]` — boundary values (0.0 / 1.0) fail many public validators.

| ID | Component | Weight | Formula | WHY |
|----|-----------|-------:|---------|-----|
| R1 | reconciliation_f1 | 0.40 | macro-F1 over 5 labels | Macro (not accuracy) — "label all matched" caps ~0.17 because 4 of 5 per-class F1s go to 0. |
| R2 | itc_delta_accuracy | 0.25 | 1 − min(1, \|claimed − true\| / max(true, 1)) | Absolute delta, not relative — "claim 0 ITC to dodge Rule 36(4)" tanks R2 to 0.01. |
| R3 | rule_36_4_compliance | 0.25 | 0.99 iff per-supplier claim ≤ 2B cap AND ≥1 query action, else 0.01 | Query prerequisite blocks confirm-spam. Per-supplier (not aggregate) catches supplier_late_filing overclaims. |
| R4 | step_efficiency | 0.10 | 1 − (steps / max_steps)² | Quadratic: penalty spikes near budget, early steps ≈ 1.0. Discourages exhaustive exploration without rewarding premature submit. |

Structural penalty: `submit()` before ≥1 query returns **−1.0** (not clamped). Submit + ≥1 query → composite. Step budget 50 exhausted → composite. Hardened mode terminates mid-episode on Rule 36(4) violation.

### Anti-gaming (red-team battery, ceiling 0.45)

Every attack scores under 0.45. Measured on seed 0:

| Attack | total | How the design catches it |
|--------|------:|---------------------------|
| submit_all_matched | 0.308 | R3 tanks (per-supplier overclaim); R4 tanks (budget exhausted). |
| submit_all_mismatched | 0.266 | R1 tanks (only 1 of 5 classes predicted); R2 tanks (zero claim). |
| confirm_spam | 0.010 | R3 tanks (no query); R4 tanks; no mutate → R1 zero. |
| zero_itc | 0.266 | R2 tanks (claim = 0 vs true > 0). R3 gives a free pass by design — defense-in-depth via R2 is documented in rewards.py. |
| query_only | 0.349 | No mutate → R1 zero, R2 zero. Highest attack on the board; still 0.10 under ceiling. |
| overflag_rings | 0.283 | No mutate marks → R1, R2 zero. |

`tests/envs/test_reconcile_gst2b_reward_hacking.py` asserts each of these < 0.45. 41 tests total across 4 test files (models, rewards, environment, reward-hacking).

## Diversity artifact

`data/reward_dist.json` + `data/reward_dist.html` from `scripts/diversity_test.py` on 100 random-policy episodes × 2 modes (warmup + hardened). All 6 criteria pass:

- ✅ ≥8 distinct totals (**81** observed)
- ✅ stdev ≥ 0.08 (**0.50** observed)
- ✅ modal outcome < 40% (**17%** observed)
- ✅ each R_i has ≥3 distinct values (R1=35, R2=66, R3=3, R4=31)
- ✅ heuristic median > random median by 0.05 (**0.62 vs 0.35**)
- ✅ done rate ≥ 95% (**100%**)

## Training

T4 dry-run (Colab, 20 steps, Qwen2.5-3B-Instruct + LoRA rank 32 via Unsloth) completed in **176 s** and saved a loadable LoRA adapter. Scaffold path runs the same policy across all eval points — curves are flat by design; the real GRPO step is a placeholder pending the pitch-demo training run. See `notebooks/dryrun_t4.ipynb`.

Baseline on 30 heldout seeds (mock proxies, real Qwen runs on Colab):

| Condition | total_mean | catastrophic_corruption | CI95 |
|-----------|-----------:|------------------------:|------|
| raw | −0.1302 | 32.2% | [−0.255, −0.010] |
| prompted | 0.2673 | 0.0% | [0.247, 0.289] |
| placeholder | 0.2042 | 8.9% | [0.117, 0.282] |

**prompted − raw = 0.397** (done-gate ≥ 0.05). Ablation: dropping R2 or R4 shifts total mean by ≥0.05 (done-gate ≥ 2 components).

## Differentiation

Agent benchmarks overlap in surface area but differ in domain shape:

| Env | Task class | Correctness source | Reward signal | Adversarial hardening |
|-----|-----------|--------------------|---------------|----------------------|
| **reconcile_gst2b_env** (this) | Rule-bound accounting reconciliation | Indian GST regulation (regulator-published) | 4-component arithmetic, clamped | 6 red-team attacks enforced in CI, all < 0.45 |
| [Gaia2](https://huggingface.co/datasets/gaia-benchmark/GAIA) | Open-ended web / real-world questions | Human-verified answers | LLM-judge or exact-match | None built-in |
| [MEMTRACK](https://arxiv.org/abs/2410.08182) | Multi-turn memory consistency | Synthetic but hand-designed | Trajectory-level match | Ontology-drift only; not adversarial |
| [AppWorld](https://appworld.dev/) | Multi-app tool use | Hand-scripted app states | Task completion booleans | None |
| [τ-bench](https://arxiv.org/abs/2406.12045) | Customer-service tool use | Simulated business rules | LLM-judge + rules | Limited |

What's distinct here:
1. **Reward is purely arithmetic and composite** — no LLM in the reward path (RLHF-style judge) — matches ARE paper §B.3.1 concerns about judge-driven reward hacking.
2. **Red-team ceiling enforced in tests** — 6 attacks × CI < 0.45, failing the test suite if any crosses.
3. **Domain has crisp wrong answers** — over-claim ITC has a regulator; "general web agent" does not.
4. **Hidden ground truth is a first-class invariant** — `ReconcileObservation` has zero `true_*` / `gt_*` fields, verified by a unit test.

## What I cut and why

- **Cut: LLM judge for open-ended "explain your reasoning" scoring.** WHY: ARE paper §B.3.1 documents judge reward-hacking; my 4-component arithmetic reward is boring but tamper-resistant.
- **Cut: RCM, ISD, SEZ, import, composition dealer, e-invoicing, e-way bill.** WHY: scope fence — each adds regulation complexity without proportionally increasing RL difficulty. See `scope_card.md`.
- **Cut: Real CBIC HSN table.** WHY: CBIC rate finder is offline from the sandbox; synthetic HSN-slab mapping is labeled as such at the table head. Not a tax-advice tool.
- **Cut: Multi-GSTIN-per-company (consolidated returns).** WHY: v1 is one GSTIN per episode; multi-GSTIN is a natural v2.
- **Cut: Real Qwen numbers for the pitch — using heuristic-mock proxies.** WHY: 5-day build with no local GPU. Colab scaffold validated; real run happens the night before the demo.

## Judge tour

See [JUDGE_TOUR.md](JUDGE_TOUR.md) for a guided 10-minute read that hits the highest-signal parts of the repo in order. [PRD.md](PRD.md) is the 1-page product-requirements view. [EXEC_SUMMARY.md](EXEC_SUMMARY.md) is the ≤10-line version for senior reviewers.

## Citations

- **OpenEnv**: Meta PyTorch OpenEnv — https://github.com/meta-pytorch/OpenEnv
- **Qwen2.5-3B-Instruct**: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- **Unsloth**: https://github.com/unslothai/unsloth
- **TRL / GRPO**: https://github.com/huggingface/trl
- **GSTN portal + Rule 36(4)**: https://www.gst.gov.in/ (Central Goods and Services Tax Rules, 2017)
- **ARE (Anthropic Research Evals) paper**, §B.3.1 on judge-driven reward hacking — cited as design rationale for arithmetic-only reward.
- **MEMTRACK**: Shen et al., 2024 — https://arxiv.org/abs/2410.08182
- **τ-bench**: Yao et al., 2024 — https://arxiv.org/abs/2406.12045

## License

MIT for this env contribution — see [LICENSE](LICENSE). Upstream OpenEnv repo is BSD-3-Clause at [../../LICENSE](../../LICENSE); both are compatible.
