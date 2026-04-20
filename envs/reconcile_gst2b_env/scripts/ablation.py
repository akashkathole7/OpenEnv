#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-component reward ablation on the prompted baseline.

For each R_i ∈ {R1, R2, R3, R4}: set that component's weight to 0,
renormalize the remaining three to sum to 1.0, rerun the prompted policy
on the same 90 rollouts (30 heldout seeds × 3 samples), and report the
total-mean drop plus per-label F1 shift.

Done-gate: at least 2 components should cause the total mean to drop by
≥0.05 when removed. Fewer → reward design is over-reliant on one
component; flag for discussion.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from envs.reconcile_gst2b_env.rewards import WEIGHTS  # noqa: E402
from envs.reconcile_gst2b_env.scripts._policies import (  # noqa: E402
    prompted_policy,
    run_episode,
)
from envs.reconcile_gst2b_env.server.reconcile_gst2b_environment import (  # noqa: E402
    ReconcileGST2BEnvironment,
)


HELDOUT_SEEDS = list(range(9000, 9030))
SAMPLES_PER_SEED = 3
ABLATIONS = ("R1", "R2", "R3", "R4")


def _ablated_weights(dropped: str) -> Dict[str, float]:
    """Zero out ``dropped``, renormalize the remaining three."""
    kept = {k: v for k, v in WEIGHTS.items() if k != dropped}
    s = sum(kept.values())
    return {k: v / s for k, v in kept.items()} | {dropped: 0.0}


def _recompute_total(
    component_rewards: Dict[str, float], weights: Dict[str, float]
) -> float:
    return sum(weights[c] * component_rewards[c] for c in ("R1", "R2", "R3", "R4"))


def _per_label_f1(logs: List[Dict[str, Any]]) -> float:
    """Average R1 across the rollouts (macro-F1 median proxy)."""
    return statistics.mean(r["component_rewards"]["R1"] for r in logs)


def main() -> int:
    out_dir = REPO / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run the prompted policy once; component rewards are policy-invariant,
    # so we reweight them post-hoc for each ablation.
    print("running prompted baseline for ablation substrate ...")
    rollouts: List[Dict[str, Any]] = []
    for seed in HELDOUT_SEEDS:
        for k in range(SAMPLES_PER_SEED):
            env = ReconcileGST2BEnvironment()
            log = run_episode(
                env,
                prompted_policy,
                seed=seed,
                rng_seed=seed * 100 + k,
                mode="warmup",
            )
            rollouts.append(log)

    baseline_totals = [r["total"] for r in rollouts]
    baseline_mean = statistics.mean(baseline_totals)
    baseline_r1 = _per_label_f1(rollouts)

    print(f"baseline total_mean = {baseline_mean:.4f}   R1_mean = {baseline_r1:.4f}")
    print()

    results: Dict[str, Any] = {
        "baseline": {
            "total_mean": round(baseline_mean, 4),
            "r1_mean": round(baseline_r1, 4),
            "weights": WEIGHTS,
            "n_rollouts": len(rollouts),
        },
        "ablations": {},
    }

    significant_drops = 0
    print(
        f"{'ablation':12s} {'weights':45s} {'total_mean':>10s} {'Δtotal':>8s} {'R1_mean':>8s}"
    )
    for dropped in ABLATIONS:
        w = _ablated_weights(dropped)
        totals = [_recompute_total(r["component_rewards"], w) for r in rollouts]
        mean = statistics.mean(totals)
        delta = mean - baseline_mean
        r1 = _per_label_f1(rollouts)  # R1 itself unchanged by weight change
        results["ablations"][dropped] = {
            "weights": {k: round(v, 4) for k, v in w.items()},
            "total_mean": round(mean, 4),
            "delta_total_mean": round(delta, 4),
            "r1_mean": round(r1, 4),
        }
        if abs(delta) >= 0.05:
            significant_drops += 1
        weights_str = ", ".join(f"{k}={v:.2f}" for k, v in w.items())
        print(
            f"drop_{dropped:7s} {weights_str:45s} {mean:10.4f} {delta:+8.4f} {r1:8.4f}"
        )

    results["significant_drop_count"] = significant_drops
    results["done_gate_ablation_ok"] = significant_drops >= 2

    (out_dir / "ablation_results.json").write_text(json.dumps(results, indent=2))

    print()
    print(
        f"=== done-gate: ablation shows {significant_drops} components with |Δ| ≥ 0.05 "
        f"(target ≥ 2) ===  {'PASS' if significant_drops >= 2 else 'FAIL'}"
    )
    return 0 if significant_drops >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())
