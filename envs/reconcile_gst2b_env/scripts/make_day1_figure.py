#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Render the Day 1 (on-site, 2026-04-25) baseline-comparison bar chart.

Reads:
  data/audit_F_n5.json   (n=5 audit headline; per-seed totals)
  envs/reconcile_gst2b_env/scripts/make_training_figures.py::_attack_scores
                                            (6 red-team attack composites)

Writes:
  data/figures/day1_baseline_comparison.png
                                            (judge-facing static recreation
                                            of HF Space Tab 4 bar chart)

Honest scope: composites are computed via the same `composite_reward`
pipeline as the live HF Space Tab 4. Static PNG so the README renders the
headline number 0.280 inline without requiring the Space to be reachable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

DATA_DIR = REPO / "envs" / "reconcile_gst2b_env" / "data"
FIG_DIR = DATA_DIR / "figures"


# Oracle range pre-verified locally on 2026-04-24 (mirrors app.py fallback).
ORACLE_SEEDS = [9500, 9501, 9502, 42, 0]
ORACLE_SCORES = [0.736, 0.814, 0.650, 0.936, 0.703]

# Prompted Qwen2.5-3B-Instruct on 30 held-out seeds × 6 samples = 180 rollouts.
# Source: data/baseline_metrics_real.json. Delta over raw policy = 1.18.
PROMPTED_BASELINE = 0.18

# Red-team attack composites — recomputed live below from the env so they
# always match the CI contract values.

# Trained Qwen3-4B SFT Day 1 — n=5 mean from data/audit_F_n5.json.
TRAINED_DAY1 = 0.280

# Red-team composite ceiling enforced by tests/test_reconcile_gst2b_reward_hacking.py.
REDTEAM_CEILING = 0.45


def _read_audit_n5_total() -> tuple[float, list[float]]:
    """Re-read data/audit_F_n5.json for the n=5 mean and per-seed totals.

    Falls back to TRAINED_DAY1 constant if the file is missing.
    """
    path = DATA_DIR / "audit_F_n5.json"
    if not path.exists():
        return TRAINED_DAY1, []
    blob = json.loads(path.read_text())
    rollouts = blob.get("rollouts", [])
    totals = [r["reward_breakdown"].get("total", 0.0) for r in rollouts]
    if not totals:
        return TRAINED_DAY1, []
    mean = sum(totals) / len(totals)
    return mean, totals


def _read_attack_scores() -> dict:
    """Recompute the 6 red-team attack composites from the env."""
    from envs.reconcile_gst2b_env.scripts.make_training_figures import _attack_scores

    return _attack_scores()


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    trained_mean, trained_per_seed = _read_audit_n5_total()
    attack_scores = _read_attack_scores()

    rows = []
    rows.append(
        {
            "label": "Oracle\n(mean, 5 seeds)",
            "score": float(np.mean(ORACLE_SCORES)),
            "err_lo": float(np.mean(ORACLE_SCORES) - min(ORACLE_SCORES)),
            "err_hi": float(max(ORACLE_SCORES) - np.mean(ORACLE_SCORES)),
            "color": "#2ca02c",
        }
    )
    for name, score_dict in sorted(
        attack_scores.items(), key=lambda kv: -kv[1]["total"]
    ):
        rows.append(
            {
                "label": f"attack:\n{name}",
                "score": float(score_dict["total"]),
                "err_lo": 0.0,
                "err_hi": 0.0,
                "color": "#d62728",
            }
        )
    rows.append(
        {
            "label": "Prompted\nQwen2.5-3B",
            "score": PROMPTED_BASELINE,
            "err_lo": 0.0,
            "err_hi": 0.0,
            "color": "#1f77b4",
        }
    )
    if trained_per_seed:
        err_lo = trained_mean - min(trained_per_seed)
        err_hi = max(trained_per_seed) - trained_mean
    else:
        err_lo = err_hi = 0.0
    rows.append(
        {
            "label": "Trained Qwen3-4B SFT\n(Day 1, A100 SXM4)",
            "score": trained_mean,
            "err_lo": float(err_lo),
            "err_hi": float(err_hi),
            "color": "#9467bd",
        }
    )

    labels = [r["label"] for r in rows]
    scores = [r["score"] for r in rows]
    err_lo_arr = [r["err_lo"] for r in rows]
    err_hi_arr = [r["err_hi"] for r in rows]
    colors = [r["color"] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
    x = np.arange(len(rows))
    bars = ax.bar(
        x,
        scores,
        yerr=[err_lo_arr, err_hi_arr],
        color=colors,
        edgecolor="#333",
        linewidth=0.8,
        ecolor="#555",
        capsize=4,
    )
    # Highlight trained bar with a thicker border.
    bars[-1].set_linewidth(2.2)
    bars[-1].set_edgecolor("#222")

    # Red-team ceiling line at 0.45.
    ax.axhline(
        REDTEAM_CEILING,
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=f"Red-team ceiling ({REDTEAM_CEILING:.2f}, CI-enforced)",
    )

    # Per-bar labels.
    for i, (xi, s) in enumerate(zip(x, scores)):
        ax.text(
            xi,
            s + max(err_hi_arr[i], 0.0) + 0.012,
            f"{s:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if i == len(rows) - 1 else "normal",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Composite reward (R1·0.40 + R2·0.25 + R3·0.25 + R4·0.10)")
    ax.set_title(
        "Day 1 on-site (2026-04-25, A100 SXM4-80GB): trained Qwen3-4B SFT vs oracle, "
        "red-team attacks, prompted baseline",
        fontsize=10.5,
        pad=12,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    fig.tight_layout()
    out_path = FIG_DIR / "day1_baseline_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)

    # Also print the data we plotted for posterity / reproducibility logs.
    print("\nbar values:")
    for r in rows:
        err = ""
        if r["err_lo"] > 0 or r["err_hi"] > 0:
            err = f"  err [-{r['err_lo']:.3f}, +{r['err_hi']:.3f}]"
        print(f"  {r['label'].replace(chr(10), ' ')}: {r['score']:.4f}{err}")
    print(f"\nred-team ceiling: {REDTEAM_CEILING}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
