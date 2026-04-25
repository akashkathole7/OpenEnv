#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Render the P3 GRPO reward-curve figure.

Two-pane visualization of FM4 (audit-OOD trap chain) and FM5 (reward-
landscape inversion) operating simultaneously:

  Top pane: TRL train-reward (per-step rewards/reward_func/mean from the
            grpo_p3_run.log). Bottom-line legitimate signal of GRPO's
            optimization trajectory under the Tier 2c length-shaping
            bonus. Shows the +0.046 trend during training.

  Bottom pane: buggy fresh-context eval callback (curves.json) flat at
            0.353 across all 6 logged steps, identical to step-0
            baseline. Compares against true post-training audit result
            (n=5 mean 0.305 from audit_grpo_p3_F_n5.json) to surface
            FM4's silent-failure shape.

Reference lines on both panes:
  - SFT baseline (0.280, n=5 audit mean of pre-GRPO checkpoint)
  - query_only Mode B exploit ceiling (0.349)
  - Red-team CI ceiling (0.45)

Reads:
  data/grpo_p3_run.log         (per-step train rewards)
  data/grpo_p3_curves.json     (eval callback per-step totals)
  data/audit_grpo_p3_F_n5.json (post-GRPO audit at n=5 with tools=)

Writes:
  data/figures/grpo_reward_curve.png
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

DATA_DIR = REPO / "envs" / "reconcile_gst2b_env" / "data"
FIG_DIR = DATA_DIR / "figures"

SFT_BASELINE_N5 = 0.280
QUERY_ONLY_CEILING = 0.349
REDTEAM_CEILING = 0.45


def _parse_train_rewards(log_path: Path) -> list[float]:
    means = []
    if not log_path.exists():
        return means
    for line in log_path.read_text().splitlines():
        m = re.search(r"rewards/reward_func/mean.:\s*.([0-9.eE+-]+)", line)
        if m:
            means.append(float(m.group(1)))
    return means


def _read_eval_curve(path: Path) -> tuple[list[int], list[float]]:
    if not path.exists():
        return [], []
    d = json.loads(path.read_text())
    return list(d.get("steps", [])), list(d.get("total", []))


def _read_audit_n5_mean(path: Path) -> float:
    if not path.exists():
        return float("nan")
    d = json.loads(path.read_text())
    totals = [r["reward_breakdown"].get("total", 0.0) for r in d.get("rollouts", [])]
    return sum(totals) / len(totals) if totals else float("nan")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train_rewards = _parse_train_rewards(DATA_DIR / "grpo_p3_run.log")
    eval_steps, eval_totals = _read_eval_curve(DATA_DIR / "grpo_p3_curves.json")
    audit_mean = _read_audit_n5_mean(DATA_DIR / "audit_grpo_p3_F_n5.json")

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 7), dpi=200, sharex=False)

    # --- Top pane: TRL train rewards ---
    if train_rewards:
        x_train = np.arange(1, len(train_rewards) + 1)
        ax_top.plot(
            x_train,
            train_rewards,
            color="#9467bd",
            linewidth=1.0,
            alpha=0.55,
            label="train reward (per-step)",
        )
        # Rolling mean for trend visibility.
        win = 10
        if len(train_rewards) >= win:
            roll = np.convolve(train_rewards, np.ones(win) / win, mode="valid")
            x_roll = np.arange(win, len(train_rewards) + 1)
            ax_top.plot(
                x_roll,
                roll,
                color="#9467bd",
                linewidth=2.2,
                label=f"train reward ({win}-step rolling mean)",
            )

    ax_top.axhline(
        SFT_BASELINE_N5,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label=f"SFT baseline (n=5 audit, {SFT_BASELINE_N5:.3f})",
    )
    ax_top.axhline(
        QUERY_ONLY_CEILING,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        label=f"query_only Mode B ({QUERY_ONLY_CEILING:.3f})",
    )
    ax_top.axhline(
        REDTEAM_CEILING,
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label=f"red-team ceiling ({REDTEAM_CEILING:.2f}, CI-enforced)",
    )
    ax_top.set_ylabel("Train reward\n(reward_func, includes Tier 2c shaping)")
    ax_top.set_xlabel("training step")
    ax_top.set_ylim(0.0, 0.55)
    ax_top.set_xlim(0, max(len(train_rewards), 100))
    ax_top.grid(axis="y", linestyle=":", alpha=0.35)
    ax_top.legend(loc="upper right", fontsize=8.5, framealpha=0.92, ncol=2)
    ax_top.set_title(
        "P3 GRPO with Tier 2c length-shaping bonus: train reward "
        "(includes shaping) vs eval callback (FM4-buggy fresh-context) vs "
        "audit (n=5 with tools=)",
        fontsize=10.5,
        pad=12,
    )

    # --- Bottom pane: eval callback vs true audit ---
    if eval_steps:
        ax_bot.plot(
            eval_steps,
            eval_totals,
            color="#d62728",
            marker="x",
            linewidth=1.5,
            markersize=10,
            label="eval callback (FM4-buggy fresh-context, no tools=)",
        )
    if not np.isnan(audit_mean):
        ax_bot.plot(
            [100],
            [audit_mean],
            marker="o",
            color="#9467bd",
            markersize=14,
            linestyle="none",
            markeredgecolor="#222",
            markeredgewidth=2.2,
            label=f"true audit (n=5 with tools=, mean {audit_mean:.3f})",
        )
        ax_bot.text(
            100,
            audit_mean + 0.025,
            f"{audit_mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#222",
        )

    ax_bot.axhline(
        SFT_BASELINE_N5,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label=f"SFT baseline (n=5 audit, {SFT_BASELINE_N5:.3f})",
    )
    ax_bot.axhline(
        QUERY_ONLY_CEILING,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        label=f"query_only Mode B ({QUERY_ONLY_CEILING:.3f})",
    )
    ax_bot.axhline(
        REDTEAM_CEILING,
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label=f"red-team ceiling ({REDTEAM_CEILING:.2f}, CI-enforced)",
    )
    ax_bot.set_ylabel("Composite reward")
    ax_bot.set_xlabel("training step")
    ax_bot.set_ylim(0.0, 0.55)
    ax_bot.set_xlim(0, max(eval_steps + [100]))
    ax_bot.grid(axis="y", linestyle=":", alpha=0.35)
    ax_bot.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    fig.tight_layout()
    out_path = FIG_DIR / "grpo_reward_curve.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)

    print(f"\ntrain rewards: {len(train_rewards)} per-step entries")
    if train_rewards:
        win = 10
        print(f"  first 10 mean: {sum(train_rewards[:win]) / win:.4f}")
        print(f"  last 10 mean:  {sum(train_rewards[-win:]) / win:.4f}")
    print(
        f"eval callback: {len(eval_steps)} steps, all at total={eval_totals[0] if eval_totals else 'n/a'}"
    )
    print(f"audit n=5 mean: {audit_mean:.4f}")
    print(f"SFT baseline:   {SFT_BASELINE_N5}")
    print(f"GRPO lift:      {audit_mean - SFT_BASELINE_N5:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
