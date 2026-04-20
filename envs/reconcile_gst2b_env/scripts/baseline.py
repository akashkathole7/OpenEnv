#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base-model baseline on 30 heldout seeds × 3 samples = 90 rollouts per condition.

Conditions:
  raw         — Qwen2.5-3B-Instruct with a minimal system prompt
  prompted    — same model + schema-introspection prompt (≤200 tokens)
  placeholder — random-init marker (uniform-random policy)

Modes:
  --mock  (default): runs CPU-only heuristic proxies for raw / prompted /
                     placeholder. Fast; produces artifacts with
                     ``"execution_mode": "mock"`` flagged in the JSON.
  --real           : loads Qwen2.5-3B-Instruct via transformers and runs
                     real rollouts. Requires GPU; Colab-runnable. Not
                     exercised by the local CI; see NOTES.md for link.

Artifacts per condition (in ``data/``):
  baseline_metrics_{raw,prompted,placeholder}.json
  transcripts/{raw,prompted,placeholder}/seed_{s}_sample_{k}.json (first 3)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from envs.reconcile_gst2b_env.rewards import bootstrap_ci  # noqa: E402
from envs.reconcile_gst2b_env.scripts._policies import (  # noqa: E402
    placeholder_policy,
    prompted_policy,
    raw_policy,
    run_episode,
)
from envs.reconcile_gst2b_env.server.reconcile_gst2b_environment import (  # noqa: E402
    ReconcileGST2BEnvironment,
)


HELDOUT_SEEDS = list(range(9000, 9030))  # 30 seeds
SAMPLES_PER_SEED = 3

CONDITIONS: Dict[str, Callable] = {
    "raw": raw_policy,
    "prompted": prompted_policy,
    "placeholder": placeholder_policy,
}


def _run_condition(name: str, policy, mock: bool) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    for seed in HELDOUT_SEEDS:
        for k in range(SAMPLES_PER_SEED):
            env = ReconcileGST2BEnvironment()
            rng_seed = seed * 100 + k
            log = run_episode(env, policy, seed=seed, rng_seed=rng_seed, mode="warmup")
            log["sample_idx"] = k
            log["condition"] = name
            logs.append(log)
    return logs


def _write_transcripts(logs: List[Dict[str, Any]], out_dir: Path, name: str) -> None:
    td = out_dir / "transcripts" / name
    td.mkdir(parents=True, exist_ok=True)
    for log in logs[:3]:
        path = td / f"seed_{log['seed']}_sample_{log['sample_idx']}.json"
        path.write_text(json.dumps(log, indent=2))


def _summarize(name: str, logs: List[Dict[str, Any]], mock: bool) -> Dict[str, Any]:
    totals = [r["total"] for r in logs]
    lo, hi = bootstrap_ci(totals, n=1000, alpha=0.05, seed=0)
    component_totals = {
        c: [r["component_rewards"][c] for r in logs] for c in ("R1", "R2", "R3", "R4")
    }
    component_summary = {}
    for c, vals in component_totals.items():
        mean = statistics.mean(vals)
        clo, chi = bootstrap_ci(vals, n=500, alpha=0.05, seed=0)
        component_summary[c] = {
            "mean": round(mean, 4),
            "ci95_low": round(clo, 4),
            "ci95_high": round(chi, 4),
        }
    catastrophic_rate = sum(1 for r in logs if r["catastrophic_corruption"]) / len(logs)
    return {
        "condition": name,
        "execution_mode": "mock" if mock else "real",
        "n_rollouts": len(logs),
        "n_seeds": len(HELDOUT_SEEDS),
        "samples_per_seed": SAMPLES_PER_SEED,
        "total_mean": round(statistics.mean(totals), 4),
        "total_ci95": [round(lo, 4), round(hi, 4)],
        "total_stdev": round(statistics.stdev(totals) if len(totals) > 1 else 0.0, 4),
        "catastrophic_corruption_rate": round(catastrophic_rate, 4),
        "component_summary": component_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Base-model baseline.")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real Qwen2.5-3B-Instruct (Colab-only). Default runs mock.",
    )
    args = parser.parse_args()
    mock = not args.real

    if args.real:
        print(
            "ERROR: --real mode requires GPU + Qwen2.5-3B-Instruct weights.\n"
            "Run this script in the accompanying Colab notebook; see NOTES.md."
        )
        return 2

    out_dir = REPO / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    for name, policy in CONDITIONS.items():
        print(f"running condition: {name}")
        logs = _run_condition(name, policy, mock)
        summary = _summarize(name, logs, mock)
        results[name] = summary
        _write_transcripts(logs, out_dir, name)
        # Lean JSON: summary + per-rollout scorecards (no full trajectories).
        scorecards = [{k: v for k, v in r.items() if k != "trajectory"} for r in logs]
        (out_dir / f"baseline_metrics_{name}.json").write_text(
            json.dumps({"summary": summary, "rollouts": scorecards}, indent=2)
        )
        print(
            f"  {name:12s} total_mean={summary['total_mean']} "
            f"ci95={summary['total_ci95']} catastrophic={summary['catastrophic_corruption_rate']}"
        )

    # Done-gate #3: prompted − raw delta ≥ 0.05.
    raw_mean = results["raw"]["total_mean"]
    prompted_mean = results["prompted"]["total_mean"]
    delta = prompted_mean - raw_mean
    print()
    print(f"=== done-gate: prompted − raw delta = {delta:.4f} (target ≥ 0.05) ===")
    if delta < 0.05:
        print("FAIL: delta insufficient — see NOTES.md Section E watch list.")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
