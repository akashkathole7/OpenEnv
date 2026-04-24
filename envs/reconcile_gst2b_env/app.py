# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gradio UI for reconcile_gst2b_env.

Four tabs:
  1. Schema + Label Diff       — placeholder
  2. Rollout Replay            — LIVE: runs a ground-truth-aware oracle
                                 against the env and shows the full verb
                                 trace + final reward breakdown. Lets
                                 judges see the 16-verb surface + the
                                 4-component composite in action without
                                 leaving the Space.
  3. Circular-Ring Viewer      — video centerpiece
  4. Baseline Comparison       — placeholder

Launch locally:
    PYTHONPATH=src:envs uv run python envs/reconcile_gst2b_env/app.py

Deployed to HF Spaces via `openenv push --enable-interface`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import gradio as gr
import networkx as nx
import plotly.graph_objects as go

try:
    from .models import ReconcileAction
    from .seed_generator import generate_episode, HERO_SEEDS
    from .server.reconcile_gst2b_environment import ReconcileGST2BEnvironment
except ImportError:
    from models import ReconcileAction
    from seed_generator import generate_episode, HERO_SEEDS
    from server.reconcile_gst2b_environment import ReconcileGST2BEnvironment


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
) -> Tuple[List[ReconcileAction], Dict[str, float], int]:
    """Run one ground-truth-aware oracle episode.

    Mirrors oracle_heuristic_policy from scripts/_policies.py but inlined
    here so app.py has no cross-module dependency on the scripts/ package
    (which uses repo-root absolute imports that break in the HF Space's
    flat module layout).

    Returns (trajectory, reward_breakdown, n_steps). The oracle reads the
    hidden ground truth from env.state to produce correct label actions,
    then submits. Partial-match delta is set so the claim equals the 2B
    side (books_tax - twob_tax), which keeps R3 per-supplier compliant.
    """
    env = ReconcileGST2BEnvironment()
    env.reset(seed=int(seed), mode="warmup")

    trajectory: List[ReconcileAction] = []
    # Step 1: satisfy the R3 query precondition.
    act = ReconcileAction(verb="get_schema", payload={})
    env.step(act)
    trajectory.append(act)

    # Build lookups for partial-match delta computation.
    books_tax = {
        inv["invoice_id"]: float(inv.get("tax_inr", 0.0))
        for inv in env.state.gt_purchase_register
    }
    twob_tax = {
        inv["invoice_id"]: float(inv.get("tax_inr", 0.0))
        for inv in env.state.gt_gstr_2b
    }

    # Step 2+: one mark per invoice via the hidden ground truth.
    for gt in env.state.gt_invoices:
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
        action = ReconcileAction(verb=verb, payload=payload)
        obs = env.step(action)
        trajectory.append(action)
        if obs.done or obs.step_budget <= 1:
            break

    # Terminal submit if not already done.
    if not env.state.reward_breakdown:
        submit = ReconcileAction(verb="submit", payload={})
        env.step(submit)
        trajectory.append(submit)

    breakdown = env.state.reward_breakdown or {}
    return trajectory, breakdown, env.state.step_count


def _format_trajectory_md(
    trajectory: List[ReconcileAction],
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


# -------------------- placeholders for tabs 1 / 4 --------------------


_PLACEHOLDER_MD = """
### Coming in a follow-up

This tab will render once on-site trained-checkpoint eval lands (see
[ONSITE_DAY1_PROMPT.md](https://github.com/akashkathole7/OpenEnv/blob/scaffold/reconcile-gst2b/envs/reconcile_gst2b_env/ONSITE_DAY1_PROMPT.md)).
Tabs 2 (Rollout Replay) and 3 (Circular-Ring Viewer) are live now.
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

            with gr.Tab("4 · Baseline Comparison"):
                gr.Markdown("### Baseline Comparison\n" + _PLACEHOLDER_MD)

    return demo


if __name__ == "__main__":
    build_demo().launch()
