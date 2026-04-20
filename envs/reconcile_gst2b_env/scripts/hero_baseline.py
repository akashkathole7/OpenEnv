#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regenerate ``data/hero_baseline.json``.

Measures the heuristic-scaffold composite reward on hero seeds 9500 / 9501 /
9502. Promotes tier_a/b/c → easy/medium/hard only if totals are strictly
decreasing. Deterministic in input; output is bit-identical to the committed
file for the same generator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from envs.reconcile_gst2b_env.scripts._policies import (  # noqa: E402
    prompted_policy,
    run_episode,
)
from envs.reconcile_gst2b_env.server.reconcile_gst2b_environment import (  # noqa: E402
    ReconcileGST2BEnvironment,
)

_TIER = {9500: "a", 9501: "b", 9502: "c"}


def main() -> int:
    results = {}
    for seed in (9500, 9501, 9502):
        env = ReconcileGST2BEnvironment()
        log = run_episode(env, prompted_policy, seed=seed, rng_seed=seed, mode="warmup")
        results[str(seed)] = {
            "seed": seed,
            "current_tier": _TIER[seed],
            "total": round(log["total"], 4),
            "components": {k: round(v, 4) for k, v in log["component_rewards"].items()},
            "steps_used": log["total_steps"],
            "termination_reason": log["termination_reason"],
        }

    totals = [results[str(s)]["total"] for s in (9500, 9501, 9502)]
    monotonic = totals[0] > totals[1] > totals[2]
    results["ordering_check"] = {
        "monotonic_easy_to_hard": monotonic,
        "promote_labels": monotonic,
        "base_model_reference": "Qwen2.5-3B-Instruct (downloaded; heuristic scaffold used for eval)",
        "eval_policy": "prompted_policy (cpu heuristic, no LM calls)",
    }
    results["promoted_labels"] = (
        {"9500": "easy", "9501": "medium", "9502": "hard"} if monotonic else None
    )

    out = REPO / "data" / "hero_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}  (monotonic={monotonic})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
