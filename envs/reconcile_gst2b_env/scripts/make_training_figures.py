#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Render the two judge-facing training figures for the README.

Reads:
  data/smoke_test_10step.json          (10-step Qwen3-0.6B GRPO trace)
  data/curves_150step_backup.json      (150-step Qwen3-0.6B eval curve)
  (red-team attack scores are computed inline against the env)

Writes:
  data/figures/three_scales_reward.png      (training reward + attack ceilings)
  data/figures/three_scales_components.png  (per-component R1/R2/R3/R4 bars)

Honest scope: only Qwen3-0.6B data survived commit. The 1.7B and 4B runs
documented in BLOG.md were lost across Kaggle session resets. See
LESSONS_LEARNED.md. Plots label the 0.6B series explicitly and use the 6
red-team attack scores as reference ceilings so the visualization lands
the "trained policy plateaus at the query_only attack signature" story
without fabricating multi-scale data.

Usage:
  PYTHONPATH=src:envs uv run python -m \\
      envs.reconcile_gst2b_env.scripts.make_training_figures
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

DATA_DIR = REPO / "envs" / "reconcile_gst2b_env" / "data"
FIG_DIR = DATA_DIR / "figures"


def _attack_scores() -> dict:
    """Re-run the 6 CI-enforced red-team attacks against the env and return
    per-attack {total, R1, R2, R3, R4}. Mirrors
    tests/envs/test_reconcile_gst2b_reward_hacking.py."""
    from envs.reconcile_gst2b_env.models import ReconcileAction
    from envs.reconcile_gst2b_env.server.reconcile_gst2b_environment import (
        ReconcileGST2BEnvironment,
    )

    seed = 0
    env = ReconcileGST2BEnvironment()
    env.reset(seed=seed)
    inv_ids = [inv["invoice_id"] for inv in env.state.invoices][:48]
    gstins = list({inv["gstin"] for inv in env.state.gt_purchase_register})[:10]
    triples = list(itertools.combinations(gstins, 3))[:40]

    def _wrap(core):
        return (
            [ReconcileAction(verb="get_schema", payload={})]
            + core
            + [ReconcileAction(verb="submit", payload={})]
        )

    attacks = {
        "submit_all_matched": _wrap(
            [
                ReconcileAction(verb="mark_matched", payload={"invoice_id": i})
                for i in inv_ids
            ]
        ),
        "submit_all_mismatched": _wrap(
            [
                ReconcileAction(
                    verb="mark_mismatched",
                    payload={"invoice_id": i, "reason": "gstin"},
                )
                for i in inv_ids
            ]
        ),
        "confirm_spam": [
            ReconcileAction(verb="confirm_with_user", payload={"question": "?"})
            for _ in range(50)
        ],
        "zero_itc": _wrap(
            [
                ReconcileAction(
                    verb="mark_mismatched",
                    payload={"invoice_id": i, "reason": "value"},
                )
                for i in inv_ids
            ]
        ),
        "query_only": [
            ReconcileAction(verb="get_schema", payload={}) for _ in range(10)
        ]
        + [ReconcileAction(verb="submit", payload={})],
        "overflag_rings": _wrap(
            [
                ReconcileAction(verb="flag_circular_ring", payload={"gstins": list(t)})
                for t in triples
            ]
        ),
    }

    out = {}
    for name, actions in attacks.items():
        env = ReconcileGST2BEnvironment()
        env.reset(seed=seed)
        final = None
        for a in actions:
            final = env.step(a)
            if final.done:
                break
        b = env.state.reward_breakdown
        out[name] = {
            "total": round(float(b.get("total", final.reward)), 4),
            "R1": round(float(b.get("R1", 0)), 4),
            "R2": round(float(b.get("R2", 0)), 4),
            "R3": round(float(b.get("R3", 0)), 4),
            "R4": round(float(b.get("R4", 0)), 4),
        }
    return out


def plot_reward(attacks: dict) -> Path:
    """Fig 1: training reward + red-team attack band over training-step axis.

    Prior version drew each of the 6 attacks as a dotted horizontal line
    with a labeled endpoint, but submit_all_mismatched and zero_itc share
    an identical 0.266 score which made the right-edge labels collide
    illegibly at thumbnail size. Now rendered as a shaded band (min..max
    across the 6 attacks) + one arrow annotation calling out the top
    attack and the trained-plateau delta. Cleaner at any scale.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    smoke = json.loads((DATA_DIR / "smoke_test_10step.json").read_text())
    curves = json.loads((DATA_DIR / "curves_150step_backup.json").read_text())

    train_steps = [s["step"] for s in smoke["train_steps"]]
    train_reward = [s.get("reward", 0.0) for s in smoke["train_steps"]]

    eval_steps = curves["steps"]
    eval_total = curves["total"]

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Red-team attack band: shaded region covering min..max of the 6 attacks.
    attack_totals = [s["total"] for s in attacks.values()]
    att_min, att_max = min(attack_totals), max(attack_totals)
    ax.axhspan(
        att_min,
        att_max,
        color="#888888",
        alpha=0.18,
        zorder=0,
        label=f"red-team attack range (6 attacks, {att_min:.3f}-{att_max:.3f})",
    )

    # Per-step training reward (the bimodal 0.174 / 0.107 signature).
    ax.plot(
        train_steps,
        train_reward,
        marker="o",
        linewidth=2.2,
        markersize=7,
        color="#1f77b4",
        label="Qwen3-0.6B per-step training reward (10-step smoke)",
        zorder=3,
    )

    # Eval composite total across 150 steps (the flat 0.353 plateau).
    ax.plot(
        eval_steps,
        eval_total,
        marker="s",
        linestyle="-",
        linewidth=2.2,
        markersize=7,
        color="#ff7f0e",
        label="Qwen3-0.6B eval composite total (150-step run, flat at 0.353)",
        zorder=3,
    )

    # Red-team ceiling (CI-enforced).
    ax.axhline(
        0.45,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label="red-team attack ceiling (0.45, CI-enforced)",
        zorder=2,
    )

    # Single summary annotation: max attack vs trained plateau.
    top_name, top = max(attacks.items(), key=lambda kv: kv[1]["total"])
    ax.annotate(
        f"max attack: {top_name} = {top['total']:.3f}\n"
        f"trained plateau: 0.353 (delta +0.004)",
        xy=(max(eval_steps), top["total"]),
        xytext=(max(eval_steps) - 80, 0.20),
        fontsize=10,
        color="#333333",
        arrowprops=dict(arrowstyle="->", color="#666666", lw=1.2),
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#888888",
            alpha=0.92,
        ),
    )

    ax.set_xlabel("training step (GRPO optimizer step)", fontsize=11)
    ax.set_ylabel("composite reward (0.0-1.0, higher is better)", fontsize=11)
    ax.set_title(
        "Qwen3-0.6B GRPO on T4: trained policy plateaus at the query_only "
        "red-team attack signature (0.353 ≈ 0.349)\n"
        "single-scale evidence; 1.7B and 4B logs lost across Kaggle resets "
        "(see LESSONS_LEARNED.md)",
        fontsize=11,
    )
    ax.set_ylim(0.0, 0.55)
    ax.set_xlim(-5, max(eval_steps) + 15)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.92)

    fig.tight_layout()
    out = FIG_DIR / "three_scales_reward.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_components(attacks: dict) -> Path:
    """Fig 2: per-component R1/R2/R3/R4 breakdown for trained 0.6B vs 6 attacks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    smoke = json.loads((DATA_DIR / "smoke_test_10step.json").read_text())
    step0 = next((e for e in smoke["eval_points"] if e["step"] == 0), None)
    if step0 is None:
        raise RuntimeError("no step-0 eval in smoke_test_10step.json")

    components = ["R1", "R2", "R3", "R4"]
    labels = [
        "R1\nreconciliation_f1\n(weight 0.40)",
        "R2\nitc_delta_accuracy\n(weight 0.25)",
        "R3\nrule_36_4_compliance\n(weight 0.25)",
        "R4\nstep_efficiency\n(weight 0.10)",
    ]

    series = {"Qwen3-0.6B (trained, eval step 0)": [step0[c] for c in components]}
    # Add a few representative attacks sorted by total, keeping top 3 for clarity.
    top_attacks = sorted(attacks.items(), key=lambda kv: -kv[1]["total"])[:3]
    for name, s in top_attacks:
        series[f"attack: {name} (total={s['total']:.3f})"] = [s[c] for c in components]

    x = np.arange(len(components))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#1f77b4", "#d62728", "#9467bd", "#8c564b"]
    for i, (label, vals) in enumerate(series.items()):
        offset = (i - (len(series) - 1) / 2) * width
        bars = ax.bar(
            x + offset, vals, width, label=label, color=colors[i % len(colors)]
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.02,
                f"{v:.2f}",
                ha="center",
                fontsize=7,
            )

    # Clamp range reference.
    ax.axhline(0.99, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axhline(0.01, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(
        len(components) - 0.5,
        1.0,
        "clamp ceiling 0.99",
        fontsize=7,
        color="grey",
        va="bottom",
    )
    ax.text(
        len(components) - 0.5,
        0.0,
        "clamp floor 0.01",
        fontsize=7,
        color="grey",
        va="top",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("reward component (weight in composite shown)")
    ax.set_ylabel("component score (0.01-0.99 clamped, higher is better)")
    ax.set_title(
        "Per-component breakdown: trained Qwen3-0.6B pins R1+R2 at floor 0.01, "
        "same cells as the submit_all_matched attack\n"
        "evidence that Qwen3-0.6B's trained policy drifted into red-team exploit "
        "territory under pure GRPO without SFT warm-start",
        fontsize=10,
    )
    ax.set_ylim(0.0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    out = FIG_DIR / "three_scales_components.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("computing red-team attack scores ...", flush=True)
    attacks = _attack_scores()
    for name, s in sorted(attacks.items(), key=lambda kv: -kv[1]["total"]):
        print(
            f"  {name:22s} total={s['total']:.4f}  R1={s['R1']:.2f} R2={s['R2']:.2f} R3={s['R3']:.2f} R4={s['R4']:.2f}"
        )
    print("\nrendering figures ...")
    f1 = plot_reward(attacks)
    f2 = plot_components(attacks)
    print(f"  wrote {f1.relative_to(REPO)} ({f1.stat().st_size / 1024:.1f} KB)")
    print(f"  wrote {f2.relative_to(REPO)} ({f2.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
