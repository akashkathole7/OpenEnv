1. OpenEnv environment for Indian GST Input-Tax-Credit reconciliation against the monthly GSTR-2B return.
2. Regulated ground truth, 5 planted mismatch types, and directed 3-cycle circular-trading rings detectable by `networkx.simple_cycles`.
3. 4-component arithmetic reward, every component clamped to `[0.01, 0.99]`; 6 adversarial attacks all score <0.45, CI-enforced.
4. Tier 1+2 GRPO fixes diagnosed + validated on 10-step T4 smoke (`data/smoke_test_10step.json`); plateau at 0.6B scale; 3B + A100 × 4 h is next compute ask.
5. Prompted Qwen2.5-3B on 30 heldout seeds lifts total from **−1.0** (100% corruption) to **0.18** (12% corruption); delta **1.18**, CI95 [1.09, 1.27].
6. Reward has gradient: 81 distinct totals, σ = 0.50, 100% done-rate across 100 random-policy episodes (2 modes).
7. Differentiation: not Gaia2 / AppWorld / τ-bench (general tool-use) — tests adversarial domain reasoning against hidden ground truth.
8. Status: environment shippable (41 tests green, `openenv validate --verbose` passes); real GRPO training is the remaining work.
9. Sharpest self-attack: R1 and R2 stay at floor (~0.01) — prompting fixes surface behavior but not reconciliation reasoning. That's the compute ask on line 10: real training to lift R1/R2.
10. Next attempt: 60-step GRPO on Qwen3-4B across Kaggle T4x2 pipeline-parallel (`device_map="auto"`, `num_generations=2`) — 1.7B converged to zero tool-call emission (`data/training_log_qwen3_1_7b_partial.json`), ruling out "size fixes it" at this range; 4B is the last free-tier attempt before the A100 + tool-SFT path.
