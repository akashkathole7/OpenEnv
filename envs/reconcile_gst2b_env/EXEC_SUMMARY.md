1. OpenEnv environment for Indian GST Input-Tax-Credit reconciliation against the monthly GSTR-2B return.
2. Regulated ground truth, 5 planted mismatch types, and directed 3-cycle circular-trading rings detectable by `networkx.simple_cycles`.
3. 4-component arithmetic reward, every component clamped to `[0.01, 0.99]`; 6 adversarial attacks all score <0.45, CI-enforced.
4. 20-step GRPO + LoRA dry-run on Colab T4 completes in 176s; LoRA checkpoint reloads via `PeftModel.from_pretrained`.
5. Baseline on 30 heldout seeds: prompted − raw = **0.397** (mock proxies; real Qwen2.5-3B runs pending).
6. Reward has gradient: 81 distinct totals, σ = 0.50, 100% done-rate across 100 random-policy episodes (2 modes).
7. Differentiation: not Gaia2 / AppWorld / τ-bench (general tool-use) — tests adversarial domain reasoning against hidden ground truth.
8. Status: environment shippable (41 tests green, `openenv validate --verbose` passes); real GRPO training is the remaining work.
9. Sharpest self-attack: the 0.397 prompted − raw delta is mock; if real Qwen collapses this to <0.05, the storytelling arc breaks — the Colab A100 run is the critical-path check.
10. Next compute ask: Colab A100 × 4h to replace mock baseline numbers with real Qwen2.5-3B GRPO.
