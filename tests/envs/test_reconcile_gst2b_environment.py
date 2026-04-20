# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke tests for the reconcile_gst2b_env server environment."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
from envs.reconcile_gst2b_env.models import ReconcileAction
from envs.reconcile_gst2b_env.server.reconcile_gst2b_environment import (
    ReconcileGST2BEnvironment,
)


@pytest.mark.parametrize("seed", list(range(10)))
def test_reset_step_submit_smoke(seed):
    env = ReconcileGST2BEnvironment()
    obs = env.reset(seed=seed)
    assert obs.done is False
    assert obs.step_budget == 50
    assert obs.user_request

    q = env.step(ReconcileAction(verb="get_schema", payload={}))
    assert q.done is False
    assert "schema" in q.last_tool_result
    assert q.step_budget == 49

    s = env.step(ReconcileAction(verb="submit", payload={}))
    assert s.done is True
    assert s.reward is not None
    assert env.state.reward_breakdown["total"] == s.reward


def test_submit_before_query_structural_penalty():
    env = ReconcileGST2BEnvironment()
    env.reset(seed=0)
    obs = env.step(ReconcileAction(verb="submit", payload={}))
    assert obs.done is True
    assert obs.reward == -1.0
    assert obs.metadata["termination_reason"] == "submit_before_query"


def test_step_budget_exhaust_terminates():
    env = ReconcileGST2BEnvironment()
    env.reset(seed=0)
    for _ in range(50):
        obs = env.step(ReconcileAction(verb="get_schema", payload={}))
    assert obs.done is True
    assert obs.metadata["termination_reason"] == "step_budget_exhausted"


def test_state_hides_ground_truth_from_observation():
    env = ReconcileGST2BEnvironment()
    obs = env.reset(seed=0)
    obs_data = obs.model_dump()
    assert not any(k.startswith(("true_", "gt_")) for k in obs_data.keys())


def test_supports_concurrent_sessions_flag_is_false():
    assert ReconcileGST2BEnvironment.SUPPORTS_CONCURRENT_SESSIONS is False


def test_schema_reports_all_16_verbs():
    env = ReconcileGST2BEnvironment()
    env.reset(seed=0)
    obs = env.step(ReconcileAction(verb="get_schema", payload={}))
    verbs = obs.last_tool_result["schema"]["verbs"]
    assert len(verbs) == 16


def test_hardened_mode_terminates_on_rule_36_4_violation():
    """In hardened mode, exceeding a supplier's 2B cap terminates mid-episode.

    Strategy: find a supplier whose books-side tax exceeds their 2B-side tax
    (present when the generator planted a supplier_late_filing against that
    supplier — books has the invoice, 2B doesn't). Mark the books-only invoice
    as ``matched`` (an overclaim). Hardened mode must terminate with
    R3 clamped to 0.01 and termination_reason="rule_36_4_violation_hardened".
    """
    env = ReconcileGST2BEnvironment()
    env.reset(seed=3, mode="hardened")

    books_tax_by_supplier: dict[str, float] = {}
    twob_tax_by_supplier: dict[str, float] = {}
    for inv in env.state.gt_purchase_register:
        books_tax_by_supplier[inv["gstin"]] = books_tax_by_supplier.get(
            inv["gstin"], 0.0
        ) + float(inv["tax_inr"])
    for inv in env.state.gt_gstr_2b:
        twob_tax_by_supplier[inv["gstin"]] = twob_tax_by_supplier.get(
            inv["gstin"], 0.0
        ) + float(inv["tax_inr"])

    violating_supplier = next(
        s
        for s, bt in books_tax_by_supplier.items()
        if bt > twob_tax_by_supplier.get(s, 0.0) + 1e-6
    )
    books_invs = [
        inv
        for inv in env.state.gt_purchase_register
        if inv["gstin"] == violating_supplier
    ]

    env.step(ReconcileAction(verb="get_schema", payload={}))
    obs = None
    for inv in books_invs:
        obs = env.step(
            ReconcileAction(
                verb="mark_matched", payload={"invoice_id": inv["invoice_id"]}
            )
        )
        if obs.done:
            break

    assert obs is not None
    assert obs.done is True
    assert obs.metadata["termination_reason"] == "rule_36_4_violation_hardened"
    assert env.state.true_rule_36_4_violated is True
    assert env.state.reward_breakdown["R3"] == 0.01


def test_warmup_mode_does_not_terminate_on_rule_36_4_violation():
    """Warmup mode tolerates per-supplier overclaim mid-episode."""
    env = ReconcileGST2BEnvironment()
    env.reset(seed=3, mode="warmup")
    env.step(ReconcileAction(verb="get_schema", payload={}))
    for inv in env.state.gt_purchase_register[:5]:
        obs = env.step(
            ReconcileAction(
                verb="mark_matched", payload={"invoice_id": inv["invoice_id"]}
            )
        )
        assert obs.metadata["termination_reason"] != "rule_36_4_violation_hardened"
