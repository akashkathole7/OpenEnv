# reconcile_gst2b_env — PRD

## Problem

Indian GST Input-Tax-Credit reconciliation — matching a company's purchase
register against its regulator-generated GSTR-2B monthly — is rule-heavy,
error-sensitive, and done by hand across 150M+ registered businesses.
Failure modes (GSTIN typos, format drift, slab errors, late filings,
post-freeze amendments, circular trading) are well-cataloged and the
regulator publishes ground truth. Crisp regulated correctness at monthly
scale makes this an unusually clean RL target.

## User

Not the tax-payer. Two downstream users:

- **RL practitioner** needing a non-game, non-code benchmark with hidden
  ground truth and adversarial reward hardening.
- **Enterprise agent team** training / evaluating a reconciliation agent
  that will run against real 2B data (real HSN table swapped in post-v1).

## Task shape

- **Episode** = one synthetic company's monthly reconciliation against GSTR-2B.
- **Action space** = 16 typed verbs (7 query, 7 mutate, 2 meta).
- **Labels** = 5 (matched, mismatched, only_in_books, only_in_2b, partial).
- **Budget** = 50 steps. Submit before ≥1 query → structural −1.0.
- **Modes** = warmup (default) and hardened (terminates mid-episode on Rule 36(4) per-supplier cap violation).
- **Hidden state** = ground-truth labels, books, 2B, company_gstin — never surfaced in `ReconcileObservation`.

## Success criteria

| # | criterion | target | measured | artifact |
|---|-----------|-------:|---------:|----------|
| 1 | red-team attack ceiling | < 0.45 | max 0.349 | `tests/envs/test_reconcile_gst2b_reward_hacking.py` |
| 2 | reward diversity stdev | ≥ 0.08 | 0.50 | `data/reward_dist.json` |
| 3 | prompted − raw baseline delta | ≥ 0.05 | 1.18, CI95 [1.09, 1.27] | `data/baseline_metrics_real.json` |
| 4 | episode done-rate | ≥ 95% | 100% | `data/reward_dist.json` |

## Non-goals

- Reverse Charge Mechanism (RCM)
- Input Service Distributor (ISD)
- Supply to / from SEZ units
- Import of services or goods
- Composition dealers (GSTR-4 regime)
- **Not tax advice.** HSN → slab table is synthetic, labeled as such in
  `ground_truth.py:96` and `scope_card.md`.
- **Not a compliance product.** The regulator is in the reward shape, not
  in any guarantee of correctness over real-world invoices.

## Verifiers

- **Reward (programmatic, no LLM)** — `rewards.py`, ~180 lines of arithmetic, auditable top-to-bottom.
- **Adversarial ceiling (CI-enforced)** — `tests/envs/test_reconcile_gst2b_reward_hacking.py`, 6 attacks each asserted < 0.45.
- **Bit-identical regeneration** — `make reproduce` runs all scripts and byte-compares outputs against committed JSONs.
- **Human spot-check** — `data/audit.html`, 10 seeds side-by-side; hand-audited on seeds 3, 5, 9 before freeze.
