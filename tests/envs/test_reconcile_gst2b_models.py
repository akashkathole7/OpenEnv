# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model tests for reconcile_gst2b_env wire types."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
from envs.reconcile_gst2b_env.models import (
    InvoiceGroundTruth,
    ReconcileAction,
    ReconcileObservation,
    ReconcileState,
)
from pydantic import ValidationError


def test_reconcile_action_accepts_all_16_verbs():
    verbs = [
        "get_schema",
        "list_gstins",
        "fuzzy_search_gstin",
        "get_invoice",
        "list_invoices_by_supplier",
        "get_2b_row",
        "get_hsn_slab",
        "mark_matched",
        "mark_mismatched",
        "mark_only_in_books",
        "mark_only_in_2b",
        "mark_partial_match",
        "flag_circular_ring",
        "request_amendment",
        "confirm_with_user",
        "submit",
    ]
    for v in verbs:
        a = ReconcileAction(verb=v, payload={})
        assert a.verb == v


def test_reconcile_action_rejects_unknown_verb():
    with pytest.raises(ValidationError):
        ReconcileAction(verb="fake_verb", payload={})


def test_reconcile_action_round_trip_json():
    a = ReconcileAction(
        verb="mark_partial_match",
        payload={"invoice_id": "inv_0001", "delta_inr": 250.0},
    )
    data = a.model_dump(mode="json")
    round = ReconcileAction.model_validate(data)
    assert round.verb == a.verb
    assert round.payload == a.payload


def test_observation_has_no_true_or_gt_fields():
    fields = set(ReconcileObservation.model_fields.keys())
    bad = [f for f in fields if f.startswith(("true_", "gt_"))]
    assert not bad, f"observation leaked hidden fields: {bad}"


def test_state_round_trip_preserves_ground_truth():
    s = ReconcileState(
        env_version="gst2b-v1.0",
        env_schema={"max_steps": 50},
        invoices=[{"invoice_id": "inv_0001"}],
        reward_breakdown={"total": 0.5},
        gt_invoices=[
            InvoiceGroundTruth(
                invoice_id="inv_0001",
                true_gstin_valid=True,
                true_hsn_slab=18,
                true_label="matched",
                true_itc_eligible_inr=1800.0,
                true_in_circular_ring=False,
                true_mismatch_type=None,
            )
        ],
        gt_purchase_register=[{"invoice_id": "inv_0001", "tax_inr": 1800.0}],
        gt_gstr_2b=[{"invoice_id": "inv_0001", "tax_inr": 1800.0}],
        gt_company_gstin="27AAPFU0939F1ZV",
        true_itc_claimed_inr=0.0,
        true_rule_36_4_violated=False,
    )
    data = s.model_dump()
    round = ReconcileState.model_validate(data)
    assert round.gt_invoices[0].true_label == "matched"
    assert round.gt_invoices[0].invoice_id == "inv_0001"
    assert round.env_schema == {"max_steps": 50}


def test_invoice_ground_truth_has_exactly_5_true_fields():
    true_fields = [k for k in InvoiceGroundTruth.model_fields if k.startswith("true_")]
    # 5 original + true_mismatch_type (added in Section C)
    assert len(true_fields) == 6
    # Spec constraint: 5 core true_* fields are the required ones; invoice_id
    # is a non-true_* join key.
    required = {
        "true_gstin_valid",
        "true_hsn_slab",
        "true_label",
        "true_itc_eligible_inr",
        "true_in_circular_ring",
    }
    assert required.issubset(set(true_fields))


def test_invoice_ground_truth_mismatch_type_accepts_none_and_5_literals():
    for mt in (
        None,
        "gstin_typo",
        "invoice_number_prefix_drift",
        "tax_slab_off_by_one",
        "supplier_late_filing",
        "amendment_after_2b_freeze",
    ):
        InvoiceGroundTruth(
            invoice_id="x",
            true_gstin_valid=True,
            true_hsn_slab=18,
            true_label="matched",
            true_itc_eligible_inr=0.0,
            true_in_circular_ring=False,
            true_mismatch_type=mt,
        )
    with pytest.raises(ValidationError):
        InvoiceGroundTruth(
            invoice_id="x",
            true_gstin_valid=True,
            true_hsn_slab=18,
            true_label="matched",
            true_itc_eligible_inr=0.0,
            true_in_circular_ring=False,
            true_mismatch_type="made_up_type",
        )
