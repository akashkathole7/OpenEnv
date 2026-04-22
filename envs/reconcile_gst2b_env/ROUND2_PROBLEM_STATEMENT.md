# Round 2 Problem Statement, `reconcile_gst2b_env`

**Theme:** #3.1 Professional Tasks / World Modeling
**Sub-theme target (bonus prize):** Scaler AI Labs, *Multi-App RL Environment for Enterprise Workflows*
**Submission author:** Aakash Kathole (solo)
**Repo:** [github.com/akashkathole7/OpenEnv](https://github.com/akashkathole7/OpenEnv) · branch `scaffold/reconcile-gst2b`
**Live Space:** [huggingface.co/spaces/akashkathole/reconcile_gst2b_env](https://huggingface.co/spaces/akashkathole/reconcile_gst2b_env)
**90-second demo video:** [youtube.com/watch?v=rglR1hGgdb8](https://www.youtube.com/watch?v=rglR1hGgdb8)

---

## At a glance

| Metric | Value | Evidence |
|---|---:|---|
| Typed tool verbs in env | **16** (7 query / 7 mutate / 2 meta) | `envs/reconcile_gst2b_env/models.py` |
| Label classes | **5** | macro-F1 scoring, `rewards.py` |
| Planted mismatch types | **5** + directed 3-cycle circular-trading rings | `seed_generator.py` |
| Reward components | **4**, each clamped to `[0.01, 0.99]` | `rewards.py` |
| Red-team attacks CI-enforced `<0.45` | **6 / 6 pass** (max `query_only` = 0.349) | `tests/envs/test_reconcile_gst2b_reward_hacking.py` |
| Test suite | **42 / 42 green** | `tests/envs/test_reconcile_gst2b_*.py` |
| Prompted vs raw baseline delta on Qwen2.5-3B-Instruct | **1.18** (95% CI [1.09, 1.27], 180 rollouts) | `data/baseline_metrics_real.json` |
| Pre-onsite training attempts | **3** (0.6B / 1.7B / 4B) with distinct failure-mode documentation | `data/training_log_qwen3_*.json` |
| `make reproduce` | **bit-identical** against committed artifacts | `Makefile:30` |
| 90-second demo video | 🎥 [youtube.com/watch?v=rglR1hGgdb8](https://www.youtube.com/watch?v=rglR1hGgdb8) | on YouTube |

---

## Why this matters (30-second hook)

Enterprise compliance workflows are the least glamorous and largest untapped RL target. Tens of millions of businesses worldwide repeat the same multi-system reconciliation loop every single month: pull records from their internal books, pull the regulator's cross-filed view, compare, commit labels, claim credit, log amendments. The task is **rule-bound** (regulator-published correctness), **multi-turn** (dozens of look-ups per document set), and **adversarially shaped** (over-claim = audit notice, under-claim = forfeit money). In India alone, ~**14 million GST-registered businesses** do this monthly, mostly by hand, by junior accountants. An RL agent that clears this workflow generalizes to every two-sided document reconciliation against a schema, invoice matching in any jurisdiction, supply-chain cross-filing, audit trails.

`reconcile_gst2b_env` is the instantiation. GST is the concrete domain because the rules are public and the failure modes are enumerable. The framework is domain-general: a partially observable multi-turn tool-orchestration env with hidden regulator-published ground truth, arithmetic reward over terminal labels, and a CI-enforced adversarial ceiling.

---

## 1. Problem statement

Given a company's monthly **purchase register** (what it claims it bought) and the regulator-generated **GSTR-2B view** (what its suppliers claimed they sold to it after cross-filing), the agent must reconcile each invoice into one of five labels, `matched`, `mismatched`, `only_in_books`, `only_in_2b`, `partial`, commit to an **Input-Tax-Credit (ITC) claim amount in INR**, and respect **Rule 36(4) per-supplier compliance** (claimed ≤ supplier's 2B-reflected cap). Episodes contain 20–100 invoices; the agent has a 50-step action budget per episode.

This captures the enterprise-workflow pattern Scaler AI Labs targets: **business rules published by a regulatory authority**, **multiple query and mutation tools** representing different back-office systems, **adversarial failure modes** (over-claim, under-claim, zero-ITC evasion, ring-trading), and **crisp measurable outcomes** (correct label + correct INR claim).

---

## 2. Environment

Built on **OpenEnv** (Meta PyTorch, latest release) as a first-class typed environment:

- **ReconcileAction** (Pydantic): one of 16 verbs with typed payloads. Unknown verbs rejected at the env boundary.
- **ReconcileObservation** (Pydantic): invoices-remaining count, step budget, last tool result, user request. **Zero `true_*` fields**, hidden ground truth is a unit-tested invariant (`test_state_hides_ground_truth_from_observation`).
- **ReconcileGST2BEnvironment**: 16 verbs split 7 query / 7 mutate / 2 meta. Two modes: `warmup` (lenient) and `hardened` (mid-episode termination on Rule 36(4) violation).
- **Concurrent sessions** supported (`SUPPORTS_CONCURRENT_SESSIONS = True`), verified by `test_concurrent_episodes_are_state_isolated`.
- **Deterministic in seed.** `generate_episode(seed)` produces identical purchase register + 2B + ground truth across processes. `make reproduce` is bit-identical.

**Mismatch type generator** plants five failure modes with weights `(1, 1, 1, 3, 3)` to keep the non-matched distribution balanced:
- `gstin_typo`, 1-char PAN edit with recomputed checksum
- `invoice_number_prefix_drift`, `INV/24-25/001` vs `INV-24-25-001`
- `tax_slab_off_by_one`, 18% vs 5% / 40% (post-GST-2.0 wider gaps)
- `supplier_late_filing`, books has it, 2B doesn't
- `amendment_after_2b_freeze`, value edited post-snapshot

With probability 0.3, a **directed 3-cycle** A→B→C→A is planted across three invoices' counterparty GSTINs, detectable only via `networkx.simple_cycles` over the supplier graph, not via naive GSTIN reassignment.

---

## 3. Agent capabilities

The agent interacts through **16 typed tool verbs**, each with a JSON-schema'd signature:

- **Query** (7, read-only): `get_schema`, `list_gstins`, `fuzzy_search_gstin`, `get_invoice`, `list_invoices_by_supplier`, `get_2b_row`, `get_hsn_slab`
- **Mutate** (7, state-changing): `mark_matched`, `mark_mismatched`, `mark_only_in_books`, `mark_only_in_2b`, `mark_partial_match`, `flag_circular_ring`, `request_amendment`
- **Meta** (2): `confirm_with_user`, `submit` (episode terminator)

The agent sees only the observation (never ground truth), must chain tool calls across multiple turns within the 50-step budget, and commits to terminal decisions via `submit`. Partial observability + multi-turn state tracking + regulator-specified correctness together form the World Modeling task.

---

## 4. Tasks

Per episode:
- **T1, Label every invoice correctly** across the 5 classes (macro-F1 scored).
- **T2, Claim correct ITC amount in INR**, respecting partial-match deltas.
- **T3, Honor Rule 36(4) per-supplier**, sum of claimed ITC per supplier ≤ that supplier's 2B-side cap.
- **T4, Use step budget efficiently**, quadratic penalty on excess queries.
- **T5, Detect circular-trading rings** when planted (via `flag_circular_ring`).

All five tasks must succeed together for a high composite score. No single reward component alone suffices, the design explicitly tests structured multi-objective reasoning.

---

## 5. Reward model & evaluation logic

Reward is **pure arithmetic** over the terminal trajectory. No LLM judge anywhere in the reward path, addresses ARE paper §B.3.1 concerns about judge-driven reward hacking.

Four components, each clamped to `[0.01, 0.99]`, weighted to sum to 1.0:

| Component | Weight | Formula | Why this shape |
|---|---:|---|---|
| **R1 reconciliation_f1** | 0.40 | Macro-F1 over 5 labels | All-matched cheats accuracy; macro-F1 caps it at ~0.17 |
| **R2 itc_delta_accuracy** | 0.25 | `1 − min(1, |claimed − true| / max(true, 1))` | Absolute delta; 2× over-claim = 0, zero-claim = 0 |
| **R3 rule_36_4_compliance** | 0.25 | `0.99` iff per-supplier cap honored AND ≥1 query, else `0.01` | Blocks confirm-spam (mutation without inspection) |
| **R4 step_efficiency** | 0.10 | `1 − (steps / 50)²` | Quadratic, penalty spikes near budget, early steps stay near ceiling |

Plus a **structural −1.0 penalty** (not clamped) for `submit` before any query verb has been called.

**Red-team CI enforcement.** `test_reconcile_gst2b_reward_hacking.py` runs six attack trajectories and asserts all score `<0.45`. Currently: `submit_all_matched = 0.308`, `submit_all_mismatched = 0.266`, `confirm_spam = 0.010`, `zero_itc = 0.266`, `query_only = 0.349`, `overflag_rings = 0.283`. Any reward-function change that breaks any ceiling fails CI.

**Evaluation logic.** Held-out seeds 9030–9039, evaluated every N training steps via forced-`get_schema` turn 1 (to isolate R3 prerequisites from the LM's ability to emit a query) plus LM-driven subsequent turns with robust 4-fallback JSON parsing (`_parse_action_robust`). 4-component breakdown persisted to `curves.json` after every eval.

---

## 6. Post-training / self-improvement strategy

This section is the differentiator. Most submissions hand-wave ("we'll GRPO"). This one has **empirical pre-onsite evidence** that pure GRPO from a base instruct model **does not solve the task at ≤4B scale**, plus a concrete fix.

### Empirical motivation: three pre-onsite training attempts on Kaggle T4 / T4x2

| Model | Observed behavior | Reward component pinned | Artifact |
|---|---|---|---|
| **Qwen3-0.6B** (10-step smoke) | Bimodal 0.25 tool-call rate per rollout; can't chain 5 `mark_*` actions reliably | R1, R2 | `data/smoke_test_10step.json` |
| **Qwen3-1.7B** (15-step partial) | Entropy collapsed to 0.12 (vs 0.44 on 0.6B); zero tool calls; deterministic mode | All components | `data/training_log_qwen3_1_7b_partial.json` |
| **Qwen3-4B** (step-0 eval only) | ~1.0 parseable tool-call rate per turn but over-queries until 50-step budget exhausts | R4 | `data/training_log_qwen3_4b_partial.json` |

**Three distinct failure modes across three model scales.** Each pins a different reward component to floor. The reward design is working, each of the four components is acting as an independent defense. But **pure GRPO from the base instruct checkpoint is not the right training recipe**; the policy does not have enough prior preference for `mark_*` actions over `get_*` actions to benefit from GRPO's advantage signal.

### On-site strategy: two-phase SFT → GRPO

**Phase 1, Synthetic expert trajectory generation (off-line, pre-training)**

Use the existing `evaluate_heuristic` policy in `training.py:175` as an oracle. It runs rule-based heuristics against the env's hidden ground truth (available only to scripts, not to the agent) and produces a sequence of (observation, action) pairs that scores well. Generate ~500 trajectories across seeds 100–599. Each trajectory has 10–50 turns of multi-turn tool-call demonstrations.

Output: a dataset of ~15,000 (prompt, tool-call, tool-response, label) quadruples.

**Phase 2, Tool-SFT warm-start**

Fine-tune Qwen3-4B (or Qwen3-8B if compute permits) on the synthetic trajectories for 1–2 epochs. LoRA rank 16, q/k/v/o targets. Standard supervised next-token loss. Target: the post-SFT model emits `<tool_call>` blocks in the correct format with a preference for `mark_*` actions when evidence supports them, rather than over-querying.

**Phase 3, GRPO polish on the SFT checkpoint**

Reuse the existing `train_grpo_real.py` with `MODEL_NAME` pointed at the Phase-2 SFT output directory. All existing Tier 1+2 reward-shaping (zero-std jitter, +0.02 format bonus), safety nets (absolute early-stop, grad-norm sanity, adapter check), and red-team invariants carry over unchanged. Target: 100–200 GRPO steps to lift the R1/R2 components off the 0.01 floor.

**Evaluation end-to-end**

Same held-out seeds (9030–9039), 4-component breakdown, `curves.json` persisted after each eval. Success criterion: composite total > **0.50** (well above the 0.349 `query_only` red-team ceiling, comfortably above the prompted baseline of 0.179, and a clear demonstration that trained ≠ attacked policy).

**What self-improvement means here.** The Phase 1 oracle policy is derived from the env's hidden ground truth, it's a teacher the agent cannot query at inference time. Phase 2 distills that teacher into the policy. Phase 3 then explores beyond the teacher via GRPO's reward signal, potentially finding label+amendment strategies the heuristic teacher doesn't encode. This is the "self-improvement after bootstrap" loop that Theme #4 also rewards.

---

## Real-world complexity & OpenEnv alignment

- **OpenEnv compliance.** Environment subclasses `openenv.core.env_server.interfaces.Environment`, registers via `openenv.yaml`, passes `openenv validate --verbose`. Tested against OpenEnv latest release.
- **Hidden ground truth is a first-class invariant.** `ReconcileObservation` has zero `true_*` / `gt_*` fields, verified by `test_state_hides_ground_truth_from_observation`. Any future code change that leaks GT into an observation fails CI.
- **Client-server separation.** Client never imports from `server/` directory. Unit-test-enforced.
- **Rewards inside the environment.** Domain knowledge encapsulated in `rewards.py`, not external, matches the OpenEnv principle that reward belongs to the env, not the trainer.
- **Deterministic + seedable + reproducible.** `make reproduce` produces bit-identical `data/` artifacts against committed versions.
- **Adversarial hardening is pre-shipped.** 6 red-team attacks CI-enforced at `<0.45`; any reward-design regression fails the test suite.

---

## Pre-onsite evidence (links into the repo)

- [BLOG.md](BLOG.md), full narrative including the triple-failure-mode analysis (§6)
- [README.md](README.md), quick-start + reward shape + differentiation table
- [EXEC_SUMMARY.md](EXEC_SUMMARY.md), 10-line ≤-senior-reviewer summary
- [PRD.md](PRD.md), product-requirements view with done-gates + success criteria
- [JUDGE_TOUR.md](JUDGE_TOUR.md), guided 10-minute repo walk
- [QA_REHEARSAL.md](QA_REHEARSAL.md), 22 pre-answered judge probes with rebuttals
- `data/smoke_test_10step.json` · `data/training_log_qwen3_1_7b_partial.json` · `data/training_log_qwen3_4b_partial.json`, three pre-onsite training attempts with structured failure-mode analysis
- `tests/envs/test_reconcile_gst2b_*.py`, 42 tests covering env contract, reward math, red-team battery, concurrent sessions, hardened-mode termination

---

## On-site compute plan (25–26 April, Bangalore)

Assuming Meta provides HF compute credits for A100-class GPUs:

1. **Hour 0–2:** Run synthetic expert trajectory generation (`evaluate_heuristic` × 500 seeds). CPU-bound, launches while waiting on GPU.
2. **Hour 2–6:** Phase 2 SFT on Qwen3-4B (or Qwen3-8B if A100-80GB). LoRA rank 16. 1 epoch over ~15K examples.
3. **Hour 6–10:** Phase 3 GRPO polish. 100–200 steps, num_generations=4, eval every 25 steps.
4. **Hour 10–12:** Generate final `curves.json` + plot, update BLOG.md + HF Space README with post-training numbers, record 90-second video walkthrough.

**Fallback if training collapses:** ship the diagnosis-and-fix narrative already documented (three distinct failure modes → SFT warm-start motivation). That story is already evidence-grounded and defensible without needing a successful curve.

---

## What distinguishes this submission from the 800-project field

1. **Domain-specific vs. generic.** Not another chess / negotiation / web-agent. A real enterprise compliance workflow with regulator-published correctness and 14M-business monthly scale.
2. **Red-team CI.** Six attack trajectories hard-coded into the test suite at `<0.45` ceiling. Reward-hacking regressions fail the build. Very few submissions will have this.
3. **Hidden ground truth as a first-class invariant.** Not just "agents don't see the answer", unit-tested that `ReconcileObservation` has zero ground-truth fields.
4. **Empirical pre-onsite evidence of training challenges.** Three distinct training attempts documented. Failure-mode matrix across model scales. Most submissions will have zero prior training evidence.
5. **Concrete on-site strategy with specific phases and time budget.** Not "we plan to GRPO", a two-phase SFT→GRPO plan with explicit trajectory-generation oracle, compute-hour breakdown, and fallback narrative.
6. **Minimum requirements pre-satisfied.** Latest OpenEnv, TRL GRPO training script, 42 tests, live HF Space, bit-identical reproducibility, sub-2-min video path ready.

---

## Contact

**Aakash Kathole**, [github.com/akashkathole7](https://github.com/akashkathole7), solo finalist, Meta × Scaler Hackathon Bangalore 2026
