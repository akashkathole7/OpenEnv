#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reward-diversity diagnostic.

Runs 100 random-policy episodes (seeds 500–599) in both warmup and hardened
modes and checks 6 signal-quality criteria. Exits non-zero on any failure.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from envs.reconcile_gst2b_env.scripts._policies import (  # noqa: E402
    heuristic_policy,
    random_policy,
    run_episode,
)
from envs.reconcile_gst2b_env.server.reconcile_gst2b_environment import (  # noqa: E402
    ReconcileGST2BEnvironment,
)


DIVERSITY_SEEDS = range(500, 600)


def _run_mode(policy, mode: str) -> List[Dict[str, Any]]:
    logs = []
    for seed in DIVERSITY_SEEDS:
        env = ReconcileGST2BEnvironment()
        logs.append(run_episode(env, policy, seed=seed, rng_seed=seed, mode=mode))
    return logs


def _evaluate(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = [r["total"] for r in logs]
    done_rate = sum(1 for r in logs if r["done"]) / len(logs)
    c_reward_fields = ("R1", "R2", "R3", "R4")
    per_component_unique = {
        c: len({r["component_rewards"].get(c) for r in logs}) for c in c_reward_fields
    }
    rounded_totals = [round(t, 3) for t in totals]
    modal_outcome_frac = Counter(rounded_totals).most_common(1)[0][1] / len(totals)
    std = statistics.stdev(totals) if len(totals) > 1 else 0.0
    return {
        "n": len(logs),
        "distinct_totals": len(set(totals)),
        "std": round(std, 4),
        "modal_outcome_frac": round(modal_outcome_frac, 4),
        "per_component_distinct_values": per_component_unique,
        "done_rate": round(done_rate, 4),
        "mean_total": round(statistics.mean(totals), 4) if totals else 0.0,
        "median_total": round(statistics.median(totals), 4) if totals else 0.0,
    }


def _render_html(warmup_stats: Dict, hardened_stats: Dict, heur_stats: Dict) -> str:
    def row(key, a, b, c):
        return f"<tr><th>{key}</th><td>{a}</td><td>{b}</td><td>{c}</td></tr>"

    rows = [
        row("n", warmup_stats["n"], hardened_stats["n"], heur_stats["n"]),
        row(
            "distinct totals",
            warmup_stats["distinct_totals"],
            hardened_stats["distinct_totals"],
            heur_stats["distinct_totals"],
        ),
        row("std", warmup_stats["std"], hardened_stats["std"], heur_stats["std"]),
        row(
            "modal frac",
            warmup_stats["modal_outcome_frac"],
            hardened_stats["modal_outcome_frac"],
            heur_stats["modal_outcome_frac"],
        ),
        row(
            "mean total",
            warmup_stats["mean_total"],
            hardened_stats["mean_total"],
            heur_stats["mean_total"],
        ),
        row(
            "median total",
            warmup_stats["median_total"],
            hardened_stats["median_total"],
            heur_stats["median_total"],
        ),
        row(
            "done rate",
            warmup_stats["done_rate"],
            hardened_stats["done_rate"],
            heur_stats["done_rate"],
        ),
    ]
    for c in ("R1", "R2", "R3", "R4"):
        rows.append(
            row(
                f"distinct {c}",
                warmup_stats["per_component_distinct_values"][c],
                hardened_stats["per_component_distinct_values"][c],
                heur_stats["per_component_distinct_values"][c],
            )
        )
    body = "".join(rows)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>reconcile_gst2b_env — reward diversity</title>"
        "<style>body{font:13px/1.4 ui-monospace,Menlo,monospace;margin:24px}"
        "table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:4px 10px;text-align:right}"
        "th:first-child{text-align:left}</style></head><body>"
        "<h1>reconcile_gst2b_env — reward diversity (seeds 500–599)</h1>"
        "<table><tr><th></th><th>random/warmup</th><th>random/hardened</th><th>heuristic/warmup</th></tr>"
        + body
        + "</table></body></html>"
    )


def main() -> int:
    warmup_logs = _run_mode(random_policy, "warmup")
    hardened_logs = _run_mode(random_policy, "hardened")
    heur_logs = _run_mode(heuristic_policy, "warmup")

    warmup_stats = _evaluate(warmup_logs)
    hardened_stats = _evaluate(hardened_logs)
    heur_stats = _evaluate(heur_logs)

    # Six criteria — evaluated on pooled warmup+hardened random logs.
    pooled = warmup_logs + hardened_logs
    pooled_stats = _evaluate(pooled)

    criteria = {
        "distinct_totals_ge_8": pooled_stats["distinct_totals"] >= 8,
        "std_ge_0_08": pooled_stats["std"] >= 0.08,
        "modal_outcome_lt_0_40": pooled_stats["modal_outcome_frac"] < 0.40,
        "per_component_ge_3_distinct": all(
            v >= 3 for v in pooled_stats["per_component_distinct_values"].values()
        ),
        "heuristic_beats_random_ge_0_05": (
            heur_stats["median_total"] - pooled_stats["median_total"]
        )
        >= 0.05,
        "done_rate_ge_0_95": pooled_stats["done_rate"] >= 0.95,
    }

    summary = {
        "warmup": warmup_stats,
        "hardened": hardened_stats,
        "heuristic_warmup": heur_stats,
        "pooled_random": pooled_stats,
        "criteria": criteria,
        "all_passed": all(criteria.values()),
    }

    out_dir = REPO / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reward_dist.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "reward_dist.html").write_text(
        _render_html(warmup_stats, hardened_stats, heur_stats)
    )

    print("=== diversity criteria ===")
    for k, v in criteria.items():
        print(
            f"  {k:40s} {'PASS' if v else 'FAIL'}  ({_summary_for(k, pooled_stats, heur_stats)})"
        )
    print()
    print(f"  all_passed: {summary['all_passed']}")
    return 0 if summary["all_passed"] else 1


def _summary_for(key: str, pooled: Dict, heur: Dict) -> str:
    if key == "distinct_totals_ge_8":
        return f"{pooled['distinct_totals']}"
    if key == "std_ge_0_08":
        return f"{pooled['std']}"
    if key == "modal_outcome_lt_0_40":
        return f"{pooled['modal_outcome_frac']}"
    if key == "per_component_ge_3_distinct":
        return str(pooled["per_component_distinct_values"])
    if key == "heuristic_beats_random_ge_0_05":
        return f"heur={heur['median_total']} pooled={pooled['median_total']}"
    if key == "done_rate_ge_0_95":
        return f"{pooled['done_rate']}"
    return ""


if __name__ == "__main__":
    sys.exit(main())
