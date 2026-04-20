# QA rehearsal — 20 judge probes + rebuttals

Two lines max per rebuttal. Cite a file or a number.

## Probe 1: Why no LLM judge in the reward path?

ARE paper §B.3.1 documents judge-driven reward hacking as the dominant
failure mode for LLM-rubric rewards. `rewards.py` is 180 lines of arithmetic;
6 red-team attacks score <0.45 with this design (test_reconcile_gst2b_reward_hacking.py).

## Probe 2: Your HSN → slab table is wrong on entry X.

Probably. `ground_truth.py` line 96 tags it `# synthetic training data only — not a
tax-advice tool`; `scope_card.md` carries the same disclaimer. This is an RL
reconciliation env — the regulator is in the reward shape, not in the HSN table.

## Probe 3: How is this different from MEMTRACK?

MEMTRACK scores ontology-drift over turns; reconcile_gst2b_env scores content-drift
(books vs 2B) against a regulator-published schema. Reward is arithmetic over a
terminal label + INR delta, not trajectory-level consistency.

## Probe 4: How do you know the 0.397 prompted−raw delta survives real Qwen?

You don't yet. `data/baseline_metrics_*.json` is flagged `"execution_mode": "mock"`;
NOTES.md logs "real Qwen runs happen in Colab before the pitch demo." Dry-run
loaded real Qwen2.5-3B in 176s as a plumbing test (notebooks/dryrun_t4.ipynb).

## Probe 5: Why are hero seeds still tier_a/b/c not easy/medium/hard?

Measured totals on 9500/9501/9502 are 0.33 / 0.46 / 0.19 (`data/hero_baseline.json`).
Ordering is 9501 > 9500 > 9502, not monotonic — done-gate #4 says promote only on
monotonicity. Spec-correct behavior; commit 2e336e7.

## Probe 6: zero_itc lets R3 hit 0.99 by claiming 0 ITC. Is that a bug?

Intentional defense-in-depth, documented in `rewards.py` module docstring lines
28–36. R2 catches it: claiming 0 vs true > 0 tanks R2 to 0.01. Composite total
for attack_zero_itc is 0.266, well under the 0.45 ceiling.

## Probe 7: GST reconciliation is too niche for a global hackathon.

150M+ GST-registered Indian businesses do this monthly; every CA firm employs
juniors for it. Task shape generalizes (two-sided document matching against a
regulated schema); see README § "Differentiation" table for overlap with τ-bench.

## Probe 8: If the HSN table is synthetic, why trust anything downstream?

The HSN table is exposed via `get_hsn_slab` as an agent tool — the agent uses
whatever the env returns. Ground-truth labels are derived from the same table, so
the reward is internally consistent regardless of whether CBIC would ratify the map.

## Probe 9: Why is `env_name="reconcile_gst2b_env"` not `"reconcile_gst2b"` per spec?

Consistency with `openenv.yaml` `name:` field beats spec literalism; logged in
NOTES.md § "Section D — deviations". `openenv validate` passes with this choice.

## Probe 10: `InvoiceGroundTruth` has 6 `true_*` fields, not 5. Spec violation?

5 core `true_*` required by spec are present; `true_mismatch_type` was added in
Section C on user approval (see commit 00b63bf). `invoice_id` is a non-`true_*`
join key required by the reward scorer.

## Probe 11: Why "50 queries exhaust budget" for no-op instead of "1 query + submit"?

Literal "1 query + submit" computes to 0.353 — outside the done-gate's
[0.20, 0.32] band with the spec's R3/R4 formulas. "50 queries → budget exhaust"
yields 0.255 ∈ band; flagged in NOTES.md and accepted by user.

## Probe 12: `n_mismatches` scales with invoice count, not literal 3–8. Justify.

Spec literal is infeasible against joint done-gates (matched 60–80% AND no-label
>40% of non-matched) on 20–100 invoice episodes. NOTES.md § "Section C deviations"
shows the constraint-satisfaction math; user approved pre-commit.

## Probe 13: Hero seeds score 90% matched, outside the 60–80% band.

Hero seeds have fixed small mismatch counts (2/5/8) for pitch-demo reproducibility;
the 60–80% band is a property of the training distribution, not hero demos.
NOTES.md § "Section C — Hero seed matched-share" logs this explicitly.

## Probe 14: R1 counts unlabeled invoices as FN but never FP. Asymmetric — bug?

Real weakness; logged as NOTES.md § "Section E watch list" item 1. An agent that
only labels when confident can over-score vs an all-wrong agent. Ablation target
for a future high-precision-lazy-labeler baseline.

## Probe 15: Looping `get_schema` 50 times satisfies R3's query gate. Exploitable?

Yes — `attack_query_only` scores 0.349 (closest to the 0.45 ceiling). NOTES.md §
"Section E watch list" item 2 flags this; fix is require ≥2 distinct query verbs,
deferred until ablation data shows it matters.

## Probe 16: You added an `only_in_2b` planter — not in the original 5 types.

Required to populate the 5th label. Scales with `n_mismatches` (`round(n_mismatches
* uniform(0.2, 0.4))`); logged as deviation #3 in NOTES.md § "Section C".

## Probe 17: Mismatch-type weights (1,1,1,3,3) not uniform. Why?

3 of 5 types map to the "mismatched" label; uniform sampling would push
"mismatched" over the 40% no-single-type cap. Weights balance non-matched
distribution to 24 / 26 / 23 / 27% (seeds 0..99); see `data/reward_dist.json`.

## Probe 18: Your 4 synthetic GSTINs may not pass an external validator.

They pass the algorithm documented in `ground_truth.py:45` (base-36 Luhn-like,
right-to-left, weights 2/1). `27AAPFU0939F1ZV` is a public test vector used as
the algorithm anchor; all 4 synthetic entries recompute to a valid checksum.

## Probe 19: Python pinned to ≥3.11 — why not 3.12?

3.11 is what Colab T4 ships with; 3.12 would force us to either roll our own
Colab image or lose the free-tier demo. Colab default wins — pragmatic choice,
not a principled one.

## Probe 20: Training curves are flat across 20 steps. Is training actually happening?

No — scaffold eval uses the heuristic policy at every step (`evaluate_heuristic`
in `training.py:175`). The real GRPO step is a documented placeholder; Cell 4 of
`notebooks/dryrun_t4.ipynb` carries "flat curves are expected" as the caption.
