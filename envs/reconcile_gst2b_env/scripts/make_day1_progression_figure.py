#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Render the Day 1 training-progression bar chart.

Plots, in order:
  - Prompted Qwen2.5-3B baseline (0.18, blue, baseline anchor)
  - Trained Qwen3-4B SFT @ step 350 (light purple, n=5 mean)
  - Trained Qwen3-4B SFT @ step 375 / final (dark purple, n=5 mean)
  - 0.45 red-team ceiling (red dashed)
  - 0.353 query_only Mode B exploit ceiling (orange dashed)

Reads:
  data/audit_ckpt350_F_n5.json   (n=5 audit at step 350)
  data/audit_F_n5.json           (n=5 audit at step 375)

Writes:
  data/figures/day1_training_progression.png

Honest scope: step 350 is only 25 optimizer steps before final (step 375).
The plot answers "did the last 25 steps move policy quality?" not
"how did training progress over all 375 steps". Per-step audit between
steps 100 and 350 was not run on-site (compute budget preserved for
Action 7 + judge-time investigation).
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

PROMPTED_BASELINE = 0.18
REDTEAM_CEILING = 0.45
QUERY_ONLY_PLATEAU = 0.349  # Mode B exploit ceiling, from CI red-team battery.


def _read_n5(path: Path) -> tuple[float, list[float]]:
    if not path.exists():
        return float("nan"), []
    blob = json.loads(path.read_text())
    totals = [r["reward_breakdown"].get("total", 0.0) for r in blob.get("rollouts", [])]
    if not totals:
        return float("nan"), []
    return sum(totals) / len(totals), totals


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    step350_mean, step350_per = _read_n5(DATA_DIR / "audit_ckpt350_F_n5.json")
    step375_mean, step375_per = _read_n5(DATA_DIR / "audit_F_n5.json")

    rows = [
        {
            "label": "Prompted\nQwen2.5-3B\n(no training)",
            "score": PROMPTED_BASELINE,
            "err_lo": 0.0,
            "err_hi": 0.0,
            "color": "#1f77b4",
            "highlight": False,
        },
        {
            "label": f"SFT step 350\n(n=5 audit)",
            "score": step350_mean,
            "err_lo": (step350_mean - min(step350_per)) if step350_per else 0.0,
            "err_hi": (max(step350_per) - step350_mean) if step350_per else 0.0,
            "color": "#c5b0d5",  # light purple
            "highlight": False,
        },
        {
            "label": f"SFT step 375 (final)\n(n=5 audit)",
            "score": step375_mean,
            "err_lo": (step375_mean - min(step375_per)) if step375_per else 0.0,
            "err_hi": (max(step375_per) - step375_mean) if step375_per else 0.0,
            "color": "#9467bd",  # dark purple
            "highlight": True,
        },
    ]

    labels = [r["label"] for r in rows]
    scores = [r["score"] for r in rows]
    err_lo = [r["err_lo"] for r in rows]
    err_hi = [r["err_hi"] for r in rows]
    colors = [r["color"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=200)
    x = np.arange(len(rows))
    bars = ax.bar(
        x,
        scores,
        yerr=[err_lo, err_hi],
        color=colors,
        edgecolor="#333",
        linewidth=0.8,
        ecolor="#555",
        capsize=5,
        width=0.55,
    )
    for b, r in zip(bars, rows):
        if r["highlight"]:
            b.set_linewidth(2.2)
            b.set_edgecolor("#222")

    # Reference lines.
    ax.axhline(
        REDTEAM_CEILING,
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label=f"Red-team ceiling ({REDTEAM_CEILING:.2f}, CI-enforced)",
    )
    ax.axhline(
        QUERY_ONLY_PLATEAU,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        label=f"query_only Mode B exploit ({QUERY_ONLY_PLATEAU:.3f})",
    )

    # Per-bar value labels.
    for xi, s, hi in zip(x, scores, err_hi):
        if not np.isnan(s):
            ax.text(
                xi,
                s + max(hi, 0.0) + 0.012,
                f"{s:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, 0.6)
    ax.set_ylabel("Composite reward (R1·0.40 + R2·0.25 + R3·0.25 + R4·0.10)")
    ax.set_title(
        "Day 1 training progression: SFT step 350 vs step 375 (final), "
        "n=5 audit on heldout seeds 9030-9034",
        fontsize=10.5,
        pad=12,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    fig.tight_layout()
    out_path = FIG_DIR / "day1_training_progression.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)

    print("\nbar values:")
    for r in rows:
        err = ""
        if r["err_lo"] > 0 or r["err_hi"] > 0:
            err = f"  err [-{r['err_lo']:.3f}, +{r['err_hi']:.3f}]"
        print(f"  {r['label'].replace(chr(10), ' ')}: {r['score']:.4f}{err}")
    print(f"\nred-team ceiling: {REDTEAM_CEILING}")
    print(f"query_only Mode B exploit: {QUERY_ONLY_PLATEAU}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
