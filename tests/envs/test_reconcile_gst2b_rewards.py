# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hand-crafted trajectory tests for the 4-component reward."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from envs.reconcile_gst2b_env.models import (
    InvoiceGroundTruth,
    ReconcileAction,
    ReconcileState,
)
from envs.reconcile_gst2b_env.rewards import (
    bootstrap_ci,
    composite_reward,
    r1_reconciliation_f1,
    r2_itc_delta_accuracy,
    r3_rule_36_4_compliance,
    r4_step_efficiency,
)


def _build_state(rows):
    """rows: list of (invoice_id, gstin, tax_inr, true_label, true_itc, mtype)."""
    pr = []
    twob = []
    gts = []
    for inv_id, gstin, tax, true_label, true_itc, mtype in rows:
        inv = {"invoice_id": inv_id, "gstin": gstin, "tax_inr": float(tax)}
        if true_label != "only_in_2b":
            pr.append(inv)
        if true_label != "only_in_books":
            twob.append(inv)
        gts.append(
            InvoiceGroundTruth(
                invoice_id=inv_id,
                true_gstin_valid=True,
                true_hsn_slab=18,
                true_label=true_label,
                true_itc_eligible_inr=float(true_itc),
                true_in_circular_ring=False,
                true_mismatch_type=mtype,
            )
        )
    return ReconcileState(
        gt_invoices=gts,
        gt_purchase_register=pr,
        gt_gstr_2b=twob,
        invoices=pr,
    )


def _act(verb, **kw):
    return ReconcileAction(verb=verb, payload=kw)


def test_r1_perfect_reconciliation_scores_near_one():
    state = _build_state(
        [
            ("a", "S1", 1800, "matched", 1800, None),
            ("b", "S2", 500, "mismatched", 0, "gstin_typo"),
            ("c", "S3", 800, "only_in_books", 0, "supplier_late_filing"),
            ("d", "S4", 700, "only_in_2b", 700, None),
            ("e", "S5", 1000, "partial", 900, "amendment_after_2b_freeze"),
        ]
    )
    traj = [
        _act("get_schema"),
        _act("mark_matched", invoice_id="a"),
        _act("mark_mismatched", invoice_id="b", reason="gstin"),
        _act("mark_only_in_books", invoice_id="c"),
        _act("mark_only_in_2b", invoice_id="d"),
        _act("mark_partial_match", invoice_id="e", delta_inr=100.0),
    ]
    assert r1_reconciliation_f1(state, traj) == 1.0


def test_r1_all_wrong_scores_zero():
    state = _build_state(
        [
            ("a", "S1", 1800, "matched", 1800, None),
            ("b", "S2", 500, "mismatched", 0, "gstin_typo"),
            ("c", "S3", 800, "only_in_books", 0, "supplier_late_filing"),
            ("d", "S4", 700, "only_in_2b", 700, None),
            ("e", "S5", 1000, "partial", 900, "amendment_after_2b_freeze"),
        ]
    )
    # Every label wrong
    traj = [
        _act("get_schema"),
        _act("mark_mismatched", invoice_id="a", reason="value"),
        _act("mark_only_in_books", invoice_id="b"),
        _act("mark_only_in_2b", invoice_id="c"),
        _act("mark_partial_match", invoice_id="d", delta_inr=0.0),
        _act("mark_matched", invoice_id="e"),
    ]
    score = r1_reconciliation_f1(state, traj)
    assert score < 0.1, f"expected near-zero macro-F1, got {score}"


def test_r2_perfect_claim_scores_one():
    state = _build_state(
        [
            ("a", "S1", 1800, "matched", 1800, None),
            ("b", "S2", 500, "mismatched", 0, "gstin_typo"),
        ]
    )
    traj = [
        _act("get_schema"),
        _act("mark_matched", invoice_id="a"),
        _act("mark_mismatched", invoice_id="b", reason="gstin"),
    ]
    assert r2_itc_delta_accuracy(state, traj) == 1.0


def test_r2_overclaim_2x_scores_near_zero():
    state = _build_state(
        [
            ("a", "S1", 1800, "matched", 1800, None),
            ("b", "S2", 1800, "mismatched", 0, "gstin_typo"),
        ]
    )
    # Claim both as matched → claim=3600, true=1800, delta=1800, ratio=1.0
    traj = [
        _act("get_schema"),
        _act("mark_matched", invoice_id="a"),
        _act("mark_matched", invoice_id="b"),
    ]
    assert r2_itc_delta_accuracy(state, traj) == 0.0


def test_r3_rule_36_4_compliance_violation_scores_low():
    # Books for S1 shows 1800 combined; 2B only reflects 1000.
    # Agent marks both matched → claims 1800 > 2B cap of 1000 → violation.
    state = ReconcileState(
        gt_purchase_register=[
            {"invoice_id": "a", "gstin": "S1", "tax_inr": 1000.0},
            {"invoice_id": "b", "gstin": "S1", "tax_inr": 800.0},
        ],
        gt_gstr_2b=[
            {"invoice_id": "a", "gstin": "S1", "tax_inr": 1000.0},
        ],
        invoices=[
            {"invoice_id": "a", "gstin": "S1", "tax_inr": 1000.0},
            {"invoice_id": "b", "gstin": "S1", "tax_inr": 800.0},
        ],
        gt_invoices=[
            InvoiceGroundTruth(
                invoice_id="a",
                true_gstin_valid=True,
                true_hsn_slab=18,
                true_label="matched",
                true_itc_eligible_inr=1000.0,
                true_in_circular_ring=False,
            ),
            InvoiceGroundTruth(
                invoice_id="b",
                true_gstin_valid=True,
                true_hsn_slab=18,
                true_label="only_in_books",
                true_itc_eligible_inr=0.0,
                true_in_circular_ring=False,
                true_mismatch_type="supplier_late_filing",
            ),
        ],
    )
    traj = [
        _act("get_schema"),
        _act("mark_matched", invoice_id="a"),
        _act("mark_matched", invoice_id="b"),
    ]
    assert r3_rule_36_4_compliance(state, traj) == 0.01


def test_r3_no_query_action_scores_low():
    state = _build_state([("a", "S1", 1000, "matched", 1000, None)])
    traj = [_act("mark_matched", invoice_id="a")]
    assert r3_rule_36_4_compliance(state, traj) == 0.01


def test_r3_compliant_with_query_scores_high():
    state = _build_state(
        [
            ("a", "S1", 1000, "matched", 1000, None),
            ("b", "S2", 800, "matched", 800, None),
        ]
    )
    traj = [
        _act("get_schema"),
        _act("mark_matched", invoice_id="a"),
        _act("mark_matched", invoice_id="b"),
    ]
    assert r3_rule_36_4_compliance(state, traj) == 0.99


def test_r4_step_efficiency_quadratic():
    state = _build_state([("a", "S1", 1000, "matched", 1000, None)])
    traj_short = [_act("get_schema")]  # 1 step
    traj_mid = [_act("get_schema")] * 25  # 25 steps
    traj_full = [_act("get_schema")] * 50  # 50 steps
    assert r4_step_efficiency(state, traj_short) > 0.99
    assert 0.7 < r4_step_efficiency(state, traj_mid) < 0.8  # 1 - 0.25 = 0.75
    assert r4_step_efficiency(state, traj_full) == 0.0


def test_composite_reward_persists_breakdown():
    state = _build_state([("a", "S1", 1000, "matched", 1000, None)])
    traj = [_act("get_schema"), _act("mark_matched", invoice_id="a")]
    result = composite_reward(state, traj)
    assert set(result.keys()) == {"R1", "R2", "R3", "R4", "total"}
    for key in ("R1", "R2", "R3", "R4"):
        assert 0.01 <= result[key] <= 0.99, f"{key} out of clamp range: {result[key]}"


def test_noop_after_one_query_total_in_band():
    """No-op = submit-budget-exhausting queries. Range [0.20, 0.32]."""
    state = _build_state(
        [
            ("a", "S1", 1000, "matched", 1000, None),
            ("b", "S2", 500, "mismatched", 0, "gstin_typo"),
        ]
    )
    # 50 queries (budget-exhaust)
    traj = [_act("get_schema")] * 50
    result = composite_reward(state, traj)
    assert 0.20 <= result["total"] <= 0.32, f"no-op total out of band: {result}"


def test_bootstrap_ci_sanity():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    lo, hi = bootstrap_ci(values, n=500, alpha=0.05, seed=0)
    assert lo <= 0.30 <= hi
    assert lo < hi
