# ReconcileEnv-GST2B: training a 3B model to do India's monthly ITC reconciliation, on a free Colab T4

## 1. The problem, for a non-Indian reader

India's Goods and Services Tax (GST) is a federal value-added tax. Every registered business with B2B purchases has to match two documents every month: its own **purchase register** (what it claims it bought) and a regulator-generated document called **GSTR-2B** (what its suppliers claimed they sold to it, after cross-filing). Where the two agree, the business can claim back the GST its suppliers paid — "input tax credit" or ITC. Where they disagree — typos, date drift, tax-slab errors, late supplier filings, or outright fraud — the business either over-claims and gets a notice, or under-claims and forfeits money. In practice this is done by hand, by CA-firm juniors, on millions of invoices a month. It is rule-heavy, error-sensitive, and boring. An excellent target for RL.

## 2. Why this makes a good RL target

Most agent benchmarks live where correctness is fuzzy — "did the agent answer the user's open-ended question well?". GST reconciliation isn't that. The regulator publishes the rules; correctness is crisp. An overclaim against a supplier's 2B cap is wrong the way 2 + 2 = 5 is wrong. That means I can write the reward function as arithmetic over the trajectory's final label + ITC claim, with no LLM in the reward path — dodging the family of reward-hacking failures documented in the ARE paper §B.3.1. It also means failure modes are a finite, enumerable list: GSTIN typo, invoice-number format drift, tax-slab error, supplier late-filing, post-freeze amendment, circular trading. I plant all six in the generator and score the agent on whether it catches them.

## 3. Environment design — five mismatch types plus rings

Episodes are deterministic in their seed. `generate_episode(seed)` produces a purchase register, a GSTR-2B view, and hidden ground truth for ~20–100 invoices. Five mismatch types are sampled with weights `(1, 1, 1, 3, 3)` to keep the non-matched label distribution balanced: **gstin_typo** (1-char PAN edit with recomputed checksum), **invoice_number_prefix_drift** (`INV/24-25/001` vs `INV-24-25-001`), **tax_slab_off_by_one** (18% vs 5% / 40% — post-GST-2.0 gaps are wider than legacy), **supplier_late_filing** (in books, absent from 2B), and **amendment_after_2b_freeze** (value edited in books after the 2B snapshot). With probability 0.3, a directed 3-cycle A→B→C→A is planted across three invoices' `counterparty_gstin` fields. `networkx.simple_cycles` over the supplier graph detects it on the real graph, not on GSTIN reassignment.

![ring viewer](data/ring_viewer.png)

## 4. Reward surface — four components with the rationale baked in

The reward is a weighted sum of four components, each clamped to `[0.01, 0.99]`:

- **R1 reconciliation_f1 (0.40)** — macro-F1 over the 5 labels. WHY macro: "label all matched" caps around 0.17 because 4 of 5 per-class F1s go to zero. An accuracy-based reward would hand back 0.7 for free.
- **R2 itc_delta_accuracy (0.25)** — `1 − min(1, |claimed − true| / max(true, 1))`. WHY absolute delta: "claim 0 ITC to dodge Rule 36(4)" tanks R2 to 0.01. So does a 2× overclaim.
- **R3 rule_36_4_compliance (0.25)** — `0.99` iff per-supplier claim ≤ 2B cap AND at least one query action was taken, else `0.01`. WHY the query prerequisite: blocks a confirm-spam strategy that mutates labels without ever inspecting data.
- **R4 step_efficiency (0.10)** — `1 − (steps / max_steps)²`. WHY quadratic: penalty spikes near budget; early steps still score ≈1.0. Discourages exhaustive exploration without rewarding premature submit.

Submitting before any query returns a structural `−1.0` (not clamped).

## 5. Known reward-hacking attempts and defenses

I wrote six attacks and committed them to CI. Every one scores under 0.45; the tests fail if any crosses:

| attack | total | caught by |
|--------|------:|-----------|
| submit_all_matched | 0.308 | R3 (overclaim) + R4 (budget) |
| submit_all_mismatched | 0.266 | R1 (1-of-5 classes) + R2 (zero claim) |
| confirm_spam | 0.010 | R3 (no query) + R4 (budget) |
| zero_itc | 0.266 | R2 (claim vs true) |
| query_only | 0.349 | R1 + R2 (no mutate) |
| overflag_rings | 0.283 | R1 + R2 (no mutate) |

zero_itc is a deliberate R3 blind spot — claiming 0 trivially satisfies the cap. Defense-in-depth via R2 catches it at composite. That choice is documented in `rewards.py`.

## 6. Training dry-run — honest numbers

I ran a 20-step dry-run on a free Colab T4, via the committed scaffold. The Qwen2.5-3B download + `pip install -e .` + training + checkpoint save completed in **176 seconds**. The scaffold uses a CPU heuristic for periodic eval (not the model) so per-component curves are **flat** across steps 0/5/10/15/20: R1=0.092, R2=0.402, R3=0.353, R4=0.806, total=0.306. That flatness is intentional — the GRPO weight-update step is a placeholder pending the real training run. What the dry-run proves is that the plumbing works end-to-end, including the `PeftModel.from_pretrained` reload at the end. Real training swaps in before the pitch demo.

## 7. Known limitations

The environment is deliberately narrow. It covers **B2B domestic supply of goods only** — RCM, ISD, SEZ, imports, composition dealers, e-invoicing, and e-way bill are out of scope. The HSN→slab table is **synthetic**, not the real CBIC mapping; this is a reconciliation RL env, not a tax-advice tool, and the table header says so. The slab structure is locked to **GST 2.0** (effective Sep 2025) — pre-2.0 episodes aren't generated. Hero-seed labels remain **tier_a / tier_b / tier_c**: measured heuristic scores are 0.33 / 0.46 / 0.19, which do not monotonically support an easy/medium/hard promotion. Relabeling waits on real Qwen numbers.

## 8. What's next

Real Colab GRPO training (T4 first, then A10 if the signal looks good), then re-measuring hero-seed ordering from a trained policy. After that, a v2 that relaxes the scope fence one regime at a time — ISD first (common enough that CAs ask for it), then RCM. The generator is structured so each regime is a new mismatch type and a new label, not a rewrite. Pull requests welcome.
