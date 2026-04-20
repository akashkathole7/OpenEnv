# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reward providers for reconcile_gst2b_env.

All component scores pass through ``clamp`` (from ``ground_truth``) before
being returned — boundary values (0.0 / 1.0) fail many public validators.
``composite_reward`` persists the per-component breakdown for baseline
analysis.

Weights (sum to 1.0):
  R1 reconciliation_f1       0.40  — macro-F1 over 5 labels, terminal
  R2 itc_delta_accuracy      0.25  — absolute delta; overclaim tanks this
  R3 rule_36_4_compliance    0.25  — 0.99 iff per-supplier cap honored AND >=1 query
  R4 step_efficiency         0.10  — 1 − (steps / max_steps)^2

Structural penalty (-1.0, NOT clamped) for ``submit`` before any query is
applied by the environment at step time, not here.

Defense-in-depth note: a "zero-ITC" strategy (mark everything mismatched,
claim 0 ITC) trivially satisfies R3's per-supplier cap and scores 0.99 on
R3 by design. The design relies on R2 catching this (claiming 0 against a
non-zero true ITC tanks R2 to 0.01), so the composite score stays well
under the 0.45 red-team ceiling. Keep R3 simple; do not add a "must claim
non-zero" clause — it would couple R3 to R2 and produce brittle edges.
"""

from __future__ import annotations

import math
import random as _random_stdlib
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from .ground_truth import clamp


WEIGHTS: Dict[str, float] = {"R1": 0.40, "R2": 0.25, "R3": 0.25, "R4": 0.10}

MAX_STEPS: int = 50

QUERY_VERBS = frozenset(
    {
        "get_schema",
        "list_gstins",
        "fuzzy_search_gstin",
        "get_invoice",
        "list_invoices_by_supplier",
        "get_2b_row",
        "get_hsn_slab",
    }
)

MUTATE_VERBS = frozenset(
    {
        "mark_matched",
        "mark_mismatched",
        "mark_only_in_books",
        "mark_only_in_2b",
        "mark_partial_match",
        "flag_circular_ring",
        "request_amendment",
    }
)

_LABEL_CLASSES = ("matched", "mismatched", "only_in_books", "only_in_2b", "partial")

_MARK_VERB_TO_LABEL = {
    "mark_matched": "matched",
    "mark_mismatched": "mismatched",
    "mark_only_in_books": "only_in_books",
    "mark_only_in_2b": "only_in_2b",
    "mark_partial_match": "partial",
}


def _extract_agent_labels(trajectory: Iterable[Any]) -> Dict[str, str]:
    """Return the latest per-invoice label from any mark_* action in the trajectory."""
    labels: Dict[str, str] = {}
    for act in trajectory:
        label = _MARK_VERB_TO_LABEL.get(act.verb)
        if label is None:
            continue
        inv_id = (act.payload or {}).get("invoice_id")
        if inv_id is None:
            continue
        labels[inv_id] = label
    return labels


def _extract_partial_deltas(trajectory: Iterable[Any]) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for act in trajectory:
        if act.verb != "mark_partial_match":
            continue
        p = act.payload or {}
        inv_id = p.get("invoice_id")
        if inv_id is None:
            continue
        deltas[inv_id] = float(p.get("delta_inr", 0.0))
    return deltas


def _claimed_itc_by_invoice(
    trajectory: Iterable[Any],
    inv_tax_by_id: Dict[str, float],
) -> Dict[str, float]:
    """Per-invoice ITC the agent is effectively claiming.

    Rule: matched/only_in_2b → full tax; partial → tax − delta (≥0);
    mismatched/only_in_books → 0. Unlabeled invoices contribute 0.
    """
    labels = _extract_agent_labels(trajectory)
    partial_deltas = _extract_partial_deltas(trajectory)
    out: Dict[str, float] = {}
    for inv_id, lab in labels.items():
        tax = float(inv_tax_by_id.get(inv_id, 0.0))
        if lab in ("matched", "only_in_2b"):
            out[inv_id] = tax
        elif lab == "partial":
            out[inv_id] = max(0.0, tax - partial_deltas.get(inv_id, 0.0))
        else:
            out[inv_id] = 0.0
    return out


def _invoice_tax_map(state: Any) -> Dict[str, float]:
    """Map invoice_id → tax_inr using books where available, else 2B."""
    out: Dict[str, float] = {}
    for inv in state.gt_purchase_register:
        out[inv["invoice_id"]] = float(inv.get("tax_inr", 0.0))
    for inv in state.gt_gstr_2b:
        out.setdefault(inv["invoice_id"], float(inv.get("tax_inr", 0.0)))
    return out


def _invoice_supplier_map(state: Any) -> Dict[str, str]:
    """Map invoice_id → supplier gstin using books where available, else 2B."""
    out: Dict[str, str] = {}
    for inv in state.gt_purchase_register:
        out[inv["invoice_id"]] = inv.get("gstin", "")
    for inv in state.gt_gstr_2b:
        out.setdefault(inv["invoice_id"], inv.get("gstin", ""))
    return out


def r1_reconciliation_f1(state: Any, trajectory: List[Any]) -> float:
    """Macro-F1 over 5 labels.

    WHY macro not accuracy: predicting "all matched" when matched is ~70%
    of labels caps macro-F1 around 0.30 (because 4 of 5 per-class F1 scores
    go to 0) while accuracy would be 0.70.
    """
    gt_map = {g.invoice_id: g.true_label for g in state.gt_invoices}
    pred_map = _extract_agent_labels(trajectory)

    per_label_f1: List[float] = []
    for lab in _LABEL_CLASSES:
        tp = fp = fn = 0
        for inv_id, true_lab in gt_map.items():
            pred = pred_map.get(inv_id)
            if pred == lab and true_lab == lab:
                tp += 1
            elif pred == lab and true_lab != lab:
                fp += 1
            elif pred != lab and true_lab == lab:
                fn += 1
        if tp + fp == 0 or tp + fn == 0 or tp == 0:
            per_label_f1.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        per_label_f1.append(f1)
    return sum(per_label_f1) / len(per_label_f1)


def r2_itc_delta_accuracy(state: Any, trajectory: List[Any]) -> float:
    """1 − min(1, |claimed − true| / max(true, 1)).

    WHY absolute delta not relative: overclaim by 2× and under-claim to 0
    both tank to 0 — mirrors real penalty structure where either side of
    misreporting triggers audit.
    """
    inv_tax_by_id = _invoice_tax_map(state)
    per_inv_claim = _claimed_itc_by_invoice(trajectory, inv_tax_by_id)
    claimed = sum(per_inv_claim.values())
    true_itc = sum(g.true_itc_eligible_inr for g in state.gt_invoices)
    denominator = max(true_itc, 1.0)
    return 1.0 - min(1.0, abs(claimed - true_itc) / denominator)


def r3_rule_36_4_compliance(state: Any, trajectory: List[Any]) -> float:
    """0.99 iff per-supplier claimed ≤ 2B-reflected AND ≥1 query action; else 0.01.

    WHY query prerequisite: blocks a confirm-spam strategy that mutates
    labels without ever inspecting data — agent must consult at least one
    query tool to claim compliance.
    """
    query_count = sum(1 for a in trajectory if a.verb in QUERY_VERBS)
    if query_count < 1:
        return 0.01

    inv_tax_by_id = _invoice_tax_map(state)
    supplier_by_id = _invoice_supplier_map(state)
    per_inv_claim = _claimed_itc_by_invoice(trajectory, inv_tax_by_id)

    per_supplier_claim: Dict[str, float] = defaultdict(float)
    for inv_id, amount in per_inv_claim.items():
        supplier = supplier_by_id.get(inv_id, "")
        if supplier:
            per_supplier_claim[supplier] += amount

    per_supplier_2b_cap: Dict[str, float] = defaultdict(float)
    for inv in state.gt_gstr_2b:
        per_supplier_2b_cap[inv["gstin"]] += float(inv.get("tax_inr", 0.0))

    _TOL = 1e-6
    for supplier, claim in per_supplier_claim.items():
        if claim > per_supplier_2b_cap.get(supplier, 0.0) + _TOL:
            return 0.01
    return 0.99


def r4_step_efficiency(state: Any, trajectory: List[Any]) -> float:
    """1 − (steps / max_steps)^2.

    WHY quadratic: discourages exhaustive exploration (penalty spikes near
    budget) without rewarding premature submit (early steps give ≈1.0).
    """
    steps = len(trajectory)
    return 1.0 - (steps / MAX_STEPS) ** 2


def composite_reward(state: Any, trajectory: List[Any]) -> Dict[str, float]:
    """Weighted composite with each component clamped independently."""
    r1 = clamp(r1_reconciliation_f1(state, trajectory))
    r2 = clamp(r2_itc_delta_accuracy(state, trajectory))
    r3 = clamp(r3_rule_36_4_compliance(state, trajectory))
    r4 = clamp(r4_step_efficiency(state, trajectory))
    total = (
        WEIGHTS["R1"] * r1
        + WEIGHTS["R2"] * r2
        + WEIGHTS["R3"] * r3
        + WEIGHTS["R4"] * r4
    )
    return {"R1": r1, "R2": r2, "R3": r3, "R4": r4, "total": total}


def bootstrap_ci(
    values: List[float],
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Returns ``(low, high)`` at confidence level ``1 - alpha``. Used by the
    Section F baseline script to report base-model score error bars.
    """
    if not values:
        return (0.0, 0.0)
    rng = _random_stdlib.Random(seed)
    m = len(values)
    means: List[float] = []
    for _ in range(n):
        sample = [values[rng.randrange(m)] for _ in range(m)]
        means.append(sum(sample) / m)
    means.sort()
    lo_idx = int(math.floor((alpha / 2) * n))
    hi_idx = int(math.ceil((1 - alpha / 2) * n)) - 1
    lo_idx = max(0, min(n - 1, lo_idx))
    hi_idx = max(0, min(n - 1, hi_idx))
    return (means[lo_idx], means[hi_idx])
