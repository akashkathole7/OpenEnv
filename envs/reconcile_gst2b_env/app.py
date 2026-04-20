# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gradio UI for reconcile_gst2b_env.

Four tabs:
  1. Schema + Label Diff       — placeholder
  2. Rollout Replay            — placeholder
  3. Circular-Ring Viewer      — video centerpiece (implemented)
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
    from .seed_generator import HERO_SEEDS, generate_episode
except ImportError:
    from seed_generator import HERO_SEEDS, generate_episode


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


# -------------------- placeholders for tabs 1 / 2 / 4 --------------------


_PLACEHOLDER_MD = """
### Coming in Section G (priority items 4)

This tab will render once the Section G scope items 4 ship. Tab 3
(Circular-Ring Viewer) is the video centerpiece and is live now.
""".strip()


# -------------------- app --------------------


def build_demo() -> gr.Blocks:
    hero_choices = sorted(HERO_SEEDS)
    default_seed = 9502  # the hero seed with a planted ring

    with gr.Blocks(title="reconcile_gst2b_env") as demo:
        gr.Markdown(
            "# reconcile_gst2b_env\n"
            "GST Input-Tax-Credit reconciliation environment. "
            "See [scope_card.md](scope_card.md) for scope fence."
        )

        with gr.Tabs():
            with gr.Tab("1 · Schema + Label Diff"):
                gr.Markdown("### Schema + Label Diff\n" + _PLACEHOLDER_MD)

            with gr.Tab("2 · Rollout Replay"):
                gr.Markdown("### Rollout Replay\n" + _PLACEHOLDER_MD)

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
