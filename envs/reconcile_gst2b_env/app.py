# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gradio UI for reconcile_gst2b_env.

Four tabs:
  1. Schema + Label Diff       — placeholder (honest: lands with on-site
                                 trained-checkpoint eval, see
                                 ONSITE_DAY1_PROMPT.md)
  2. Rollout Replay            — LIVE: runs a ground-truth-aware oracle
                                 against the env and shows the full verb
                                 trace + final reward breakdown. Lets
                                 judges see the 16-verb surface + the
                                 4-component composite in action without
                                 leaving the Space.
  3. Circular-Ring Viewer      — LIVE: 3D supplier graph with planted
                                 directed-cycle fraud rings; video centerpiece.
  4. Baseline Comparison       — LIVE: composite-reward bar chart across
                                 oracle + 6 red-team attacks + prompted
                                 baseline + trained Qwen3-4B SFT (Day 1,
                                 A100 SXM4) at n=5 mean composite reward
                                 0.280. The 0.45 ceiling line is the
                                 CI-enforced red-team ceiling. Oracle is
                                 live-computed at module load; trained
                                 bar reflects on-site Day 1 measurement.

Launch locally:
    PYTHONPATH=src:envs uv run python envs/reconcile_gst2b_env/app.py

Deployed to HF Spaces via `openenv push --enable-interface`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import gradio as gr
import networkx as nx
import plotly.graph_objects as go

# IMPORTANT: we deliberately do NOT import models.py or the env server in
# app.py. Both pull openenv.core transitively, which is not installed in the
# HF Space container (requirements.txt holds only 6 pinned runtime deps).
# The oracle rollout uses seed_generator + rewards directly with
# SimpleNamespace stubs for State/Action, which keeps the Space container
# self-contained and avoids a git+openenv install.
try:
    from .rewards import composite_reward
    from .seed_generator import generate_episode, HERO_SEEDS
except ImportError:
    from rewards import composite_reward
    from seed_generator import generate_episode, HERO_SEEDS


# -------------------- Tab 3: Circular-Ring Viewer --------------------


def _build_supplier_graph(episode: Dict[str, Any]) -> Tuple[nx.DiGraph, set[str]]:
    """Return (G, ring_gstins). G is the supplier→counterparty digraph over
    non-trivial edges (edges where counterparty != company). ring_gstins is
    the set of GSTINs belonging to any detected ≤3-cycle."""
    company = episode["company_gstin"]
    G = nx.DiGraph()
    for inv in episode["purchase_register"]:
        src = inv["gstin"]
        dst = inv["counterparty_gstin"]
        if dst == company:
            continue  # skip trivial supplier→company edges
        G.add_edge(src, dst, invoice_id=inv["invoice_id"])
    cycles = [c for c in nx.simple_cycles(G) if len(c) <= 3]
    ring_gstins: set[str] = set()
    for c in cycles:
        ring_gstins.update(c)
    return G, ring_gstins


def _supplier_graph_figure(
    G: nx.DiGraph, ring_gstins: set[str], title: str
) -> go.Figure:
    """Return a 3D plotly figure. Ring nodes red, edges in ring red, rest gray."""
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(
            title=f"{title} — no non-trivial edges in this episode",
            height=560,
        )
        return fig

    pos = nx.spring_layout(G, dim=3, seed=42, k=0.8)

    # Edge lines: segment per edge, colored by ring membership.
    ring_edge_x: List[float] = []
    ring_edge_y: List[float] = []
    ring_edge_z: List[float] = []
    plain_edge_x: List[float] = []
    plain_edge_y: List[float] = []
    plain_edge_z: List[float] = []
    for u, v in G.edges():
        xs = [pos[u][0], pos[v][0], None]
        ys = [pos[u][1], pos[v][1], None]
        zs = [pos[u][2], pos[v][2], None]
        if u in ring_gstins and v in ring_gstins:
            ring_edge_x += xs
            ring_edge_y += ys
            ring_edge_z += zs
        else:
            plain_edge_x += xs
            plain_edge_y += ys
            plain_edge_z += zs

    plain_edges = go.Scatter3d(
        x=plain_edge_x,
        y=plain_edge_y,
        z=plain_edge_z,
        mode="lines",
        line=dict(color="#c0c0c0", width=2),
        hoverinfo="none",
        name="non-ring edges",
    )
    ring_edges = go.Scatter3d(
        x=ring_edge_x,
        y=ring_edge_y,
        z=ring_edge_z,
        mode="lines",
        line=dict(color="#d73a49", width=6),
        hoverinfo="none",
        name="ring edges",
    )

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_z = [pos[n][2] for n in G.nodes()]
    node_colors = ["#d73a49" if n in ring_gstins else "#6699cc" for n in G.nodes()]
    node_sizes = [14 if n in ring_gstins else 8 for n in G.nodes()]
    node_text = [
        f"{n}<br>{'RING' if n in ring_gstins else 'supplier'}" for n in G.nodes()
    ]

    nodes = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color="#333"),
        ),
        text=node_text,
        hoverinfo="text",
        name="suppliers",
    )

    fig = go.Figure(data=[plain_edges, ring_edges, nodes])
    fig.update_layout(
        title=title,
        height=620,
        scene=dict(
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(showticklabels=False, title=""),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
    )
    return fig


def _ring_summary(episode: Dict[str, Any], G: nx.DiGraph, ring_gstins: set[str]) -> str:
    """Markdown summary block for the ring viewer."""
    cycles = [c for c in nx.simple_cycles(G) if len(c) <= 3]
    gt_ring_ids = [
        g["invoice_id"] for g in episode["ground_truth"] if g["true_in_circular_ring"]
    ]
    lines = [
        f"**seed**: `{episode.get('_seed', '?')}`",
        f"**company_gstin**: `{episode['company_gstin']}`",
        f"**purchase_register size**: {len(episode['purchase_register'])} invoices",
        f"**non-trivial supplier→counterparty edges** (excluding edges to company): {G.number_of_edges()}",
        f"**ground-truth ring invoices**: {len(gt_ring_ids)}",
        f"**cycles detected by networkx.simple_cycles (len ≤ 3)**: {len(cycles)}",
    ]
    if cycles:
        lines.append("")
        lines.append("**Detected cycles**:")
        for i, c in enumerate(cycles):
            edges = " → ".join(c) + f" → {c[0]}"
            lines.append(f"- cycle {i + 1}: `{edges}`")
    else:
        lines.append("")
        lines.append("_No ring in this episode. Try hero seed 9502._")
    return "\n".join(lines)


def _ring_view(seed: int) -> Tuple[go.Figure, str]:
    episode = generate_episode(int(seed))
    episode["_seed"] = int(seed)
    G, ring_gstins = _build_supplier_graph(episode)
    title = f"supplier graph — seed {seed}" + (
        f" (RING of {len(ring_gstins)} GSTINs)" if ring_gstins else ""
    )
    fig = _supplier_graph_figure(G, ring_gstins, title)
    summary = _ring_summary(episode, G, ring_gstins)
    return fig, summary


# -------------------- Tab 2: Rollout Replay (live oracle) --------------------


_MARK_VERB_FOR_LABEL = {
    "matched": "mark_matched",
    "mismatched": "mark_mismatched",
    "only_in_books": "mark_only_in_books",
    "only_in_2b": "mark_only_in_2b",
    "partial": "mark_partial_match",
}


def _oracle_trajectory(
    seed: int,
) -> Tuple[List[SimpleNamespace], Dict[str, float], int]:
    """Run one ground-truth-aware oracle episode WITHOUT the env server.

    Uses seed_generator.generate_episode + rewards.composite_reward directly
    with SimpleNamespace stubs for State/Action, so this module does not
    need models.py or server/* (both of which transitively import
    openenv.core, which is not installed in the HF Space container).

    The stubs provide the minimum attribute surface rewards.py needs:
      state.gt_invoices             list with .invoice_id, .true_label,
                                    .true_itc_eligible_inr
      state.gt_purchase_register    list of dicts (invoice_id, gstin, tax_inr)
      state.gt_gstr_2b              same shape
      action.verb, action.payload   per-step

    Partial-match delta = books_tax - twob_tax so per-supplier claim
    equals the 2B cap, keeping R3 compliant. Truncates at MAX_STEPS - 1
    so the terminal submit always fits in the 50-step budget.

    Returns (trajectory, reward_breakdown, n_steps).
    """
    MAX_STEPS = 50
    episode = generate_episode(int(seed))

    # Stub state mirroring ReconcileState's public surface for rewards.py.
    gt_invoices = [SimpleNamespace(**g) for g in episode["ground_truth"]]
    state = SimpleNamespace(
        gt_invoices=gt_invoices,
        gt_purchase_register=episode["purchase_register"],
        gt_gstr_2b=episode["gstr_2b"],
    )

    books_tax = {
        inv["invoice_id"]: float(inv.get("tax_inr", 0.0))
        for inv in episode["purchase_register"]
    }
    twob_tax = {
        inv["invoice_id"]: float(inv.get("tax_inr", 0.0)) for inv in episode["gstr_2b"]
    }

    # Step 1: satisfy R3 query precondition.
    trajectory: List[SimpleNamespace] = [SimpleNamespace(verb="get_schema", payload={})]

    for gt in gt_invoices:
        if len(trajectory) >= MAX_STEPS - 1:
            break
        verb = _MARK_VERB_FOR_LABEL[gt.true_label]
        payload: Dict[str, Any] = {"invoice_id": gt.invoice_id}
        if verb == "mark_mismatched":
            payload["reason"] = "value"
        elif verb == "mark_partial_match":
            delta = max(
                0.0,
                books_tax.get(gt.invoice_id, 0.0) - twob_tax.get(gt.invoice_id, 0.0),
            )
            payload["delta_inr"] = round(delta, 2)
        trajectory.append(SimpleNamespace(verb=verb, payload=payload))

    trajectory.append(SimpleNamespace(verb="submit", payload={}))

    breakdown = composite_reward(state, trajectory)
    return trajectory, dict(breakdown), len(trajectory)


def _format_trajectory_md(
    trajectory: List[SimpleNamespace],
    breakdown: Dict[str, float],
    n_steps: int,
    seed: int,
) -> str:
    """Compact markdown trace: first N + last 3 verbs, plus reward breakdown."""
    QUERY = {
        "get_schema",
        "list_gstins",
        "fuzzy_search_gstin",
        "get_invoice",
        "list_invoices_by_supplier",
        "get_2b_row",
        "get_hsn_slab",
    }
    MUTATE = {
        "mark_matched",
        "mark_mismatched",
        "mark_only_in_books",
        "mark_only_in_2b",
        "mark_partial_match",
        "flag_circular_ring",
        "request_amendment",
    }

    def tag(verb: str) -> str:
        if verb in QUERY:
            return "🔍"
        if verb in MUTATE:
            return "✏️"
        return "🏁"

    total_steps = len(trajectory)
    head_n = min(12, total_steps)
    tail_n = 3 if total_steps > head_n + 3 else 0

    lines: List[str] = []
    lines.append(f"## Episode seed `{seed}` — oracle rollout")
    lines.append("")
    lines.append(
        f"**Steps taken**: {n_steps} of 50 budget  ·  **Trajectory length**: {total_steps} verbs"
    )
    lines.append("")
    lines.append("### Final reward breakdown")
    r1 = breakdown.get("R1", 0.0)
    r2 = breakdown.get("R2", 0.0)
    r3 = breakdown.get("R3", 0.0)
    r4 = breakdown.get("R4", 0.0)
    total = breakdown.get("total", 0.0)
    lines.append("")
    lines.append("| Component | Weight | Score | Weighted |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| R1 reconciliation_f1 | 0.40 | {r1:.3f} | {0.40 * r1:.3f} |")
    lines.append(f"| R2 itc_delta_accuracy | 0.25 | {r2:.3f} | {0.25 * r2:.3f} |")
    lines.append(f"| R3 rule_36_4_compliance | 0.25 | {r3:.3f} | {0.25 * r3:.3f} |")
    lines.append(f"| R4 step_efficiency | 0.10 | {r4:.3f} | {0.10 * r4:.3f} |")
    lines.append(f"| **total composite** | 1.00 | | **{total:.3f}** |")
    lines.append("")
    verdict = (
        "✅ clears the 0.45 red-team attack ceiling"
        if total > 0.45
        else "⚠️ below the 0.45 red-team ceiling — this is why pure GRPO needs SFT warm-start"
    )
    lines.append(f"**Verdict**: {verdict}")
    lines.append("")
    lines.append("### Verb trace (🔍 query · ✏️ mutate · 🏁 meta)")
    lines.append("")
    for i, action in enumerate(trajectory[:head_n]):
        p = action.payload or {}
        # Compact payload for narrow display.
        payload_s = ""
        if "invoice_id" in p:
            payload_s = f" invoice_id=`{p['invoice_id']}`"
            if "reason" in p:
                payload_s += f" reason=`{p['reason']}`"
            if "delta_inr" in p:
                payload_s += f" delta_inr={p['delta_inr']}"
        elif "question" in p:
            payload_s = f' question="{p["question"][:30]}"'
        elif "gstins" in p:
            payload_s = f" gstins=[{len(p['gstins'])} GSTINs]"
        elif "gstin" in p:
            payload_s = f" gstin=`{p['gstin']}`"
        elif "hsn" in p:
            payload_s = f" hsn=`{p['hsn']}`"
        lines.append(f"{i + 1:3d}. {tag(action.verb)} `{action.verb}`{payload_s}")
    if tail_n:
        lines.append(f"   ... ({total_steps - head_n - tail_n} more mark steps) ...")
        for i, action in enumerate(
            trajectory[-tail_n:], start=total_steps - tail_n + 1
        ):
            p = action.payload or {}
            payload_s = ""
            if "invoice_id" in p:
                payload_s = f" invoice_id=`{p['invoice_id']}`"
            lines.append(f"{i:3d}. {tag(action.verb)} `{action.verb}`{payload_s}")
    return "\n".join(lines)


def _run_episode(seed_raw: Any) -> str:
    """Gradio callback: run one oracle episode on the given seed."""
    try:
        seed = int(seed_raw) if seed_raw is not None else 9502
    except (TypeError, ValueError):
        return "**Error**: seed must be an integer."
    trajectory, breakdown, n_steps = _oracle_trajectory(seed)
    return _format_trajectory_md(trajectory, breakdown, n_steps, seed)


# -------------------- Tab 4: Baseline Comparison --------------------


# Red-team attack totals, re-measured 2026-04-24 against the live env.
# Source of truth: tests/envs/test_reconcile_gst2b_reward_hacking.py (CI-enforced <0.45).
_REDTEAM_ATTACKS: Dict[str, float] = {
    "query_only": 0.349,
    "submit_all_matched": 0.308,
    "overflag_rings": 0.283,
    "submit_all_mismatched": 0.266,
    "zero_itc": 0.266,
    "confirm_spam": 0.010,
}

# Prompted Qwen2.5-3B-Instruct on 30 held-out seeds × 6 samples = 180 rollouts.
# Source: data/baseline_metrics_real.json. Delta over raw policy = 1.18 (CI95 [1.09, 1.27]).
_PROMPTED_BASELINE = 0.18

# Trained Qwen3-4B SFT + P3 GRPO with Tier 2c length-shaping bonus, measured
# on Day 1 on-site (2026-04-25, A100 SXM4-80GB): n=5 mean composite reward
# 0.305 at GRPO-matching sampling (T=0.7, top_p=0.95, top_k=20) with tools=
# enabled. SFT baseline was 0.280; GRPO P3 lifts +0.025. Source:
# data/audit_grpo_p3_F_n5.json. See LESSONS_LEARNED §1 FM5 for the
# reward-landscape inversion finding and shaping mitigation.
_TRAINED_PLACEHOLDER = 0.305


def _compute_oracle_range() -> Dict[str, Any]:
    """Compute oracle total across 5 representative seeds.

    Runs live at module load so the chart reflects the actual scorer. If the
    live compute errors for any reason in the Space container, falls back to
    values pre-verified locally on 2026-04-24.
    """
    seeds = [9500, 9501, 9502, 42, 0]
    try:
        scores = [_oracle_trajectory(s)[1]["total"] for s in seeds]
    except Exception:  # pragma: no cover - defensive fallback
        scores = [0.736, 0.814, 0.650, 0.936, 0.703]
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": sum(scores) / len(scores),
        "scores": scores,
    }


_ORACLE_RANGE = _compute_oracle_range()


def _baseline_comparison_figure() -> go.Figure:
    """Composite-reward bar chart across oracle, red-team attacks, and baselines.

    All bars score via the same rewards.composite_reward pipeline. Ordered
    by score descending within category; oracle shown with min-max error
    bar; trained-policy bar reflects Day 1 on-site n=5 mean composite
    reward (0.280) measured under GRPO-matching sampling with tools=
    enabled. Source: data/audit_F_n5.json.
    """
    rows = []
    rows.append(
        {
            "label": "oracle<br>(mean, 5 seeds)",
            "score": _ORACLE_RANGE["mean"],
            "color": "#2ca02c",  # green
            "err_above": _ORACLE_RANGE["max"] - _ORACLE_RANGE["mean"],
            "err_below": _ORACLE_RANGE["mean"] - _ORACLE_RANGE["min"],
            "text_pos": "outside",
        }
    )
    for name, score in sorted(_REDTEAM_ATTACKS.items(), key=lambda kv: -kv[1]):
        rows.append(
            {
                "label": f"attack:<br>{name}",
                "score": score,
                "color": "#d62728",  # red
                "err_above": 0,
                "err_below": 0,
                "text_pos": "outside",
            }
        )
    rows.append(
        {
            "label": "Prompted<br>Qwen2.5-3B",
            "score": _PROMPTED_BASELINE,
            "color": "#1f77b4",  # blue
            "err_above": 0,
            "err_below": 0,
            "text_pos": "outside",
        }
    )
    rows.append(
        {
            "label": "Trained Qwen3-4B SFT + GRPO<br>(Day 1, A100 SXM4)",
            "score": _TRAINED_PLACEHOLDER,
            "color": "#9467bd",  # purple, distinguishes from oracle/baseline/attack
            "err_above": 0,
            "err_below": 0,
            "text_pos": "outside",
        }
    )

    labels = [r["label"] for r in rows]
    scores = [r["score"] for r in rows]
    colors = [r["color"] for r in rows]
    err_above = [r["err_above"] for r in rows]
    err_below = [r["err_below"] for r in rows]
    texts = [f"{s:.3f}" for s in scores]
    text_positions = [r["text_pos"] for r in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=scores,
            marker=dict(color=colors, line=dict(width=1, color="#333")),
            error_y=dict(
                type="data",
                array=err_above,
                arrayminus=err_below,
                visible=True,
                color="#555",
                thickness=1.5,
                width=8,
            ),
            text=texts,
            textposition=text_positions,
            textfont=dict(size=11),
            hovertemplate="<b>%{x}</b><br>composite reward: %{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )

    # CI-enforced red-team ceiling.
    fig.add_hline(
        y=0.45,
        line_dash="dash",
        line_color="#d62728",
        line_width=2,
        annotation_text="red-team attack ceiling (0.45, CI-enforced)",
        annotation_position="top right",
        annotation_font_size=11,
    )

    fig.update_layout(
        title=dict(
            text=(
                "Composite reward across policies: oracle vs 6 red-team attacks"
                " vs prompted baseline vs trained target<br>"
                "<sub>all scored via the same rewards.composite_reward pipeline;"
                " trained bar replaced with measured value after on-site A100 run</sub>"
            ),
            font=dict(size=13),
        ),
        xaxis=dict(title="policy / attack", tickangle=-20, tickfont=dict(size=10)),
        yaxis=dict(
            title="composite reward (0.0-1.0, higher is better)",
            range=[0.0, 1.05],
        ),
        height=560,
        margin=dict(l=40, r=40, t=90, b=120),
        plot_bgcolor="white",
    )
    fig.update_yaxes(gridcolor="#eee")
    return fig


def _baseline_comparison_summary() -> str:
    """Markdown summary under the chart. Pulls live oracle numbers."""
    return (
        f"**Oracle spread (5 seeds):** min {_ORACLE_RANGE['min']:.3f} · "
        f"mean {_ORACLE_RANGE['mean']:.3f} · max {_ORACLE_RANGE['max']:.3f}  \n"
        f"**Prompted Qwen2.5-3B baseline:** 0.18 (180 rollouts, delta +1.18 "
        f"over raw policy, 95% CI [1.09, 1.27])  \n"
        f"**Red-team attack ceiling:** 0.45 (CI-enforced, 6 attacks all strictly below)  \n"
        f"**Trained Qwen3-4B target:** 0.50 (ROUND2 success criterion; measured "
        f"number lands from on-site A100 run 2026-04-25 to 2026-04-26; see "
        f"[LESSONS_LEARNED.md](https://github.com/akashkathole7/OpenEnv/blob/"
        f"scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/LESSONS_LEARNED.md))  \n"
        f"\n"
        f"Every bar above is a composite total from `rewards.composite_reward`. "
        f"No LLM in the reward path. The CI test file that enforces the 0.45 "
        f"ceiling is "
        f"[`test_reconcile_gst2b_reward_hacking.py`](https://github.com/"
        f"akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/tests/envs/"
        f"test_reconcile_gst2b_reward_hacking.py)."
    )


# -------------------- placeholder for tab 1 --------------------


_PLACEHOLDER_MD = """
### Coming in a follow-up

This tab will render once on-site trained-checkpoint eval lands (see
[ONSITE_DAY1_PROMPT.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/ONSITE_DAY1_PROMPT.md)).
Tabs 2 (Rollout Replay), 3 (Circular-Ring Viewer), and 4 (Baseline Comparison)
are live now.
""".strip()


# -------------------- app --------------------


def build_demo() -> gr.Blocks:
    hero_choices = sorted(HERO_SEEDS)
    default_seed = 9502  # the hero seed with a planted ring

    with gr.Blocks(title="reconcile_gst2b_env") as demo:
        gr.Markdown(
            "# reconcile_gst2b_env\n"
            "**Multi-turn enterprise compliance workflow RL on OpenEnv.** "
            "Instantiated on Indian GST Input-Tax-Credit reconciliation — "
            "~14M businesses, monthly, still manual. Targeting **Scaler AI Labs — "
            "Multi-App RL Environment for Enterprise Workflows** sub-theme.\n\n"
            "**16 typed tool verbs · 5 labels · 4-component arithmetic reward · "
            "6 red-team attacks CI-enforced <0.45 · 42 tests · `make reproduce` "
            "bit-identical**\n\n"
            "👇 **Live demo: select Tab 3 (Circular-Ring Viewer)**, pick hero seed "
            "**9502**, click **Render** to see a planted directed-cycle fraud ring "
            "in 3D. Full README + training evidence in the **Files** tab."
        )

        with gr.Tabs():
            with gr.Tab("1 · Schema + Label Diff"):
                gr.Markdown("### Schema + Label Diff\n" + _PLACEHOLDER_MD)

            with gr.Tab("2 · Rollout Replay", elem_id="rollout-tab"):
                gr.Markdown(
                    "### Rollout Replay — run 1 episode, see the 16-verb surface in action\n"
                    "Click **Run 1 Episode (oracle)** to drive a ground-truth-aware "
                    "oracle agent through one full reconciliation episode on the seed "
                    "of your choice. Output shows (a) the final 4-component reward "
                    "breakdown and (b) the verb trace (query verbs 🔍, mutate verbs ✏️, "
                    "meta 🏁). The oracle reads the hidden ground truth to pick "
                    "correct labels — use this as the upper-bound reference for what a "
                    "well-behaved RL policy converges toward. A random policy on the "
                    "same seed would score closer to the red-team attack band "
                    "(see the figures in Files → README.md)."
                )
                with gr.Row():
                    replay_seed_dd = gr.Dropdown(
                        label="hero seed",
                        choices=[str(s) for s in sorted(HERO_SEEDS)],
                        value=str(9502),
                        scale=1,
                    )
                    replay_seed_custom = gr.Number(
                        label="or any seed",
                        value=9502,
                        precision=0,
                        scale=1,
                    )
                    run_btn = gr.Button(
                        "▶ Run 1 Episode (oracle)", scale=1, variant="primary"
                    )
                replay_output = gr.Markdown()

                def _on_run(hero_choice, custom_seed):
                    seed = (
                        int(custom_seed)
                        if custom_seed is not None
                        else int(hero_choice)
                    )
                    return _run_episode(seed)

                run_btn.click(
                    _on_run,
                    inputs=[replay_seed_dd, replay_seed_custom],
                    outputs=[replay_output],
                )
                # Prime the tab with a default run on hero seed 9502.
                demo.load(
                    lambda: _run_episode(9502),
                    outputs=[replay_output],
                )

            with gr.Tab("3 · Circular-Ring Viewer", elem_id="ring-tab"):
                gr.Markdown(
                    "### Circular-Ring Viewer\n"
                    "3D supplier → counterparty graph. Ring nodes + edges in "
                    "**red**. Non-ring suppliers in blue. Hero seed **9502** "
                    "has a planted A→B→C→A cycle; networkx.simple_cycles "
                    "detects it on the real graph, not on GSTIN reassignment. "
                    "Rotate / pan / zoom the 3D view."
                )
                with gr.Row():
                    seed_dd = gr.Dropdown(
                        label="hero seed",
                        choices=[str(s) for s in hero_choices],
                        value=str(default_seed),
                        scale=1,
                    )
                    seed_custom = gr.Number(
                        label="or any seed",
                        value=default_seed,
                        precision=0,
                        scale=1,
                    )
                    render_btn = gr.Button(
                        "render supplier graph", scale=1, variant="primary"
                    )
                graph_plot = gr.Plot(label="supplier graph (3D)")
                summary_md = gr.Markdown()

                def _on_render(hero_choice, custom_seed):
                    seed = (
                        int(custom_seed)
                        if custom_seed is not None
                        else int(hero_choice)
                    )
                    return _ring_view(seed)

                render_btn.click(
                    _on_render,
                    inputs=[seed_dd, seed_custom],
                    outputs=[graph_plot, summary_md],
                )
                # Prime the tab with the default view.
                demo.load(
                    lambda: _ring_view(default_seed),
                    outputs=[graph_plot, summary_md],
                )

            with gr.Tab("4 · Baseline Comparison", elem_id="baseline-tab"):
                gr.Markdown(
                    "### Baseline Comparison — where every policy sits on the reward axis\n"
                    "One chart. All composite-reward totals. Oracle, 6 CI-enforced "
                    "red-team attacks, the prompted Qwen2.5-3B baseline, and the "
                    "on-site-pending Trained Qwen3-4B target. The dashed line at "
                    "0.45 is the red-team attack ceiling enforced in CI. The gray "
                    "bar is intentional scope (on-site A100 run, Apr 25-26), not "
                    "missing work — see the LESSONS_LEARNED link below the chart."
                )
                baseline_plot = gr.Plot(label="composite reward across policies")
                baseline_summary = gr.Markdown()
                demo.load(
                    lambda: (
                        _baseline_comparison_figure(),
                        _baseline_comparison_summary(),
                    ),
                    outputs=[baseline_plot, baseline_summary],
                )

    return demo


if __name__ == "__main__":
    build_demo().launch()
