# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Human audit HTML for seed inspection.

``generate_audit_html(seeds, output_path)`` renders one section per seed
with user request, register/2B totals, label counts, mismatch-type
breakdown, ring GSTINs, planted non-matched invoices side-by-side with
their ground-truth label, and 2 example clean invoices. Plain HTML +
inline CSS, no JS. Auditable in ≤10 minutes per file.
"""

from __future__ import annotations

import html
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

try:
    from .seed_generator import generate_episode
except ImportError:
    from seed_generator import generate_episode


_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>reconcile_gst2b_env — seed audit</title>
<style>
 body { font: 13px/1.4 ui-monospace, Menlo, monospace; background: #fafafa; color: #222; margin: 24px; }
 h1 { font-size: 20px; margin: 0 0 16px; }
 h3 { font-size: 13px; margin: 10px 0 4px; color: #555; }
 .seed { background: #fff; border: 1px solid #ddd; border-radius: 6px;
         padding: 12px 16px; margin: 0 0 14px; }
 .seed h2 { margin: 0 0 8px; font-size: 15px; color: #0366d6; }
 .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 20px; }
 .kv { display: flex; justify-content: space-between; border-bottom: 1px dotted #eee; padding: 2px 0; }
 .kv b { color: #555; font-weight: 600; }
 .req { font-style: italic; color: #444; margin: 6px 0 10px; }
 table { border-collapse: collapse; width: 100%; margin-top: 6px; font-size: 12px; }
 th, td { border: 1px solid #e5e5e5; padding: 3px 6px; text-align: left; vertical-align: top; }
 th { background: #f4f4f4; }
 .ring { color: #d73a49; font-weight: 700; }
 .hero { color: #6f42c1; font-weight: 700; }
 .ring-table { background: #fff5f5; }
 .planted { background: #fffaf0; }
 .label-matched { color: #2e7d32; }
 .label-mismatched { color: #b71c1c; }
 .label-partial { color: #e65100; }
 .label-only_in_books { color: #1565c0; }
 .label-only_in_2b { color: #6a1b9a; }
 code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }
</style>
</head>
<body>
<h1>reconcile_gst2b_env — seed audit</h1>
<p>Pick 3 seeds at random. For each: (a) verify a planted-mismatch row's data matches its
<code>true_label</code> + <code>true_mismatch_type</code>; (b) if a ring is present, trace the
3 GSTINs as a supplier → counterparty cycle; (c) eyeball matched-share is 60–80%.</p>
"""

_TAIL = "</body></html>\n"

_LABELS = ("matched", "mismatched", "only_in_books", "only_in_2b", "partial")


def _summarize_side(invoices: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "count": len(invoices),
        "total_value_inr": round(sum(i["value_inr"] for i in invoices), 2),
        "total_tax_inr": round(sum(i["tax_inr"] for i in invoices), 2),
    }


def _render_invoice_row(inv: Dict[str, Any], extra_cells: List[str] = []) -> str:
    cells = [
        inv["invoice_id"],
        inv["invoice_no"],
        inv["gstin"],
        inv.get("counterparty_gstin", ""),
        inv["hsn"],
        f"{inv['slab_pct']}%",
        f"{inv['value_inr']:,.2f}",
        f"{inv['tax_inr']:,.2f}",
    ] + list(extra_cells)
    return "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr>"


_BASE_HEADERS = (
    "<th>id</th><th>invoice_no</th><th>supplier_gstin</th>"
    "<th>counterparty_gstin</th><th>hsn</th><th>slab</th>"
    "<th>value</th><th>tax</th>"
)


def _render_seed(seed: int, ep: Dict[str, Any]) -> str:
    books = ep["purchase_register"]
    twob = ep["gstr_2b"]
    gt = ep["ground_truth"]
    company = ep["company_gstin"]

    label_counts = Counter(g["true_label"] for g in gt)
    mtype_counts = Counter(
        g["true_mismatch_type"] for g in gt if g["true_mismatch_type"] is not None
    )
    ring_ids = [g["invoice_id"] for g in gt if g["true_in_circular_ring"]]
    books_summary = _summarize_side(books)
    twob_summary = _summarize_side(twob)

    matched_share = label_counts.get("matched", 0) / max(1, sum(label_counts.values()))

    hero_label = (
        f' <span class="hero">hero-{ep["hero_tier"]}</span>'
        if ep.get("hero_tier")
        else ""
    )
    ring_label = f' <span class="ring">RING({len(ring_ids)})</span>' if ring_ids else ""

    parts = [
        f'<div class="seed"><h2>seed {seed}{hero_label}{ring_label} '
        f"· company = <code>{html.escape(company)}</code></h2>"
    ]
    parts.append(f'<div class="req">“{html.escape(ep["user_request"])}”</div>')
    parts.append('<div class="grid">')
    parts.append(
        f'<div class="kv"><b>purchase_register</b><span>{books_summary["count"]} invs · '
        f"₹{books_summary['total_value_inr']:,.2f} · tax ₹{books_summary['total_tax_inr']:,.2f}</span></div>"
    )
    parts.append(
        f'<div class="kv"><b>gstr_2b</b><span>{twob_summary["count"]} invs · '
        f"₹{twob_summary['total_value_inr']:,.2f} · tax ₹{twob_summary['total_tax_inr']:,.2f}</span></div>"
    )
    for lab in _LABELS:
        parts.append(
            f'<div class="kv"><b>label:{lab}</b>'
            f'<span class="label-{lab}">{label_counts.get(lab, 0)}</span></div>'
        )
    parts.append(
        f'<div class="kv"><b>matched share</b><span>{matched_share:.1%}</span></div>'
    )
    parts.append(
        f'<div class="kv"><b>in_circular_ring</b><span>{len(ring_ids)} invs</span></div>'
    )
    parts.append("</div>")

    # Mismatch-type breakdown
    parts.append("<h3>mismatch type breakdown</h3>")
    parts.append("<table><tr><th>type</th><th>count</th></tr>")
    for mt in (
        "gstin_typo",
        "invoice_number_prefix_drift",
        "tax_slab_off_by_one",
        "supplier_late_filing",
        "amendment_after_2b_freeze",
    ):
        parts.append(f"<tr><td>{mt}</td><td>{mtype_counts.get(mt, 0)}</td></tr>")
    parts.append("</table>")

    # Ring detail (if any): show the 3 invoices forming the cycle
    if ring_ids:
        parts.append("<h3>planted circular ring</h3>")
        parts.append('<table class="ring-table">')
        parts.append(f"<tr>{_BASE_HEADERS}</tr>")
        for rid in ring_ids:
            inv = next(i for i in books if i["invoice_id"] == rid)
            parts.append(_render_invoice_row(inv))
        parts.append("</table>")
        ring_gstins = [
            next(i["gstin"] for i in books if i["invoice_id"] == rid)
            for rid in ring_ids
        ]
        ring_counterparties = [
            next(i["counterparty_gstin"] for i in books if i["invoice_id"] == rid)
            for rid in ring_ids
        ]
        edges = " → ".join(f"{s}→{c}" for s, c in zip(ring_gstins, ring_counterparties))
        parts.append(f"<p>ring edges: <code>{html.escape(edges)}</code></p>")

    # Planted non-matched invoices — cross-check against ground truth.
    planted_ids = [g["invoice_id"] for g in gt if g["true_label"] != "matched"][
        :5
    ]  # cap at 5 per seed for auditability
    if planted_ids:
        parts.append("<h3>planted non-matched invoices (books side vs GT)</h3>")
        parts.append('<table class="planted">')
        parts.append(
            f"<tr>{_BASE_HEADERS}<th>true_label</th><th>true_mismatch_type</th><th>true_itc</th></tr>"
        )
        for pid in planted_ids:
            inv = next((i for i in books if i["invoice_id"] == pid), None)
            g = next(x for x in gt if x["invoice_id"] == pid)
            if inv is None:
                # 2B-only case
                inv = next(i for i in twob if i["invoice_id"] == pid)
            extras = [
                g["true_label"],
                g["true_mismatch_type"] or "—",
                f"{g['true_itc_eligible_inr']:,.2f}",
            ]
            parts.append(_render_invoice_row(inv, extras))
        parts.append("</table>")

    # Example clean invoices from each side for format sanity
    parts.append("<h3>example clean invoices (books)</h3>")
    matched_books = [
        inv
        for inv in books
        if next(
            (g for g in gt if g["invoice_id"] == inv["invoice_id"]),
            {"true_label": "matched", "true_in_circular_ring": False},
        )["true_label"]
        == "matched"
        and not next(
            (g for g in gt if g["invoice_id"] == inv["invoice_id"]),
            {"true_in_circular_ring": False},
        )["true_in_circular_ring"]
    ][:2]
    parts.append(f"<table><tr>{_BASE_HEADERS}</tr>")
    for inv in matched_books:
        parts.append(_render_invoice_row(inv))
    parts.append("</table>")

    parts.append("</div>")
    return "".join(parts)


def generate_audit_html(seeds: List[int], output_path: str) -> Path:
    """Render per-seed summaries to a single HTML file at ``output_path``."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    body_parts = [_HEAD]
    for seed in seeds:
        ep = generate_episode(seed)
        body_parts.append(_render_seed(seed, ep))
    body_parts.append(_TAIL)

    out.write_text("".join(body_parts), encoding="utf-8")
    return out
