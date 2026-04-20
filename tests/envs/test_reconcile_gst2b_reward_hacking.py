# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Red-team battery: each attack's composite reward must score < 0.45.

If any attack reaches 0.45, the reward design has a hole — stop and redesign.
"""

from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
from envs.reconcile_gst2b_env.models import ReconcileAction
from envs.reconcile_gst2b_env.server.reconcile_gst2b_environment import (
    ReconcileGST2BEnvironment,
)


ATTACK_CEILING = 0.45
SEED = 0


@pytest.fixture(scope="module")
def attack_trajectories():
    """Shared fixture: builds all 6 attack trajectories once per module."""
    env = ReconcileGST2BEnvironment()
    env.reset(seed=SEED)
    inv_ids = [inv["invoice_id"] for inv in env.state.invoices][:48]
    gstins = list({inv["gstin"] for inv in env.state.gt_purchase_register})[:10]
    triples = list(itertools.combinations(gstins, 3))[:40]

    def _wrap(core):
        return (
            [ReconcileAction(verb="get_schema", payload={})]
            + core
            + [ReconcileAction(verb="submit", payload={})]
        )

    return {
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


def _run_attack(actions):
    env = ReconcileGST2BEnvironment()
    env.reset(seed=SEED)
    final = None
    for a in actions:
        final = env.step(a)
        if final.done:
            break
    return final.reward


def _assert_under_ceiling(name, reward):
    assert reward < ATTACK_CEILING, (
        f"{name} scored {reward:.4f} ≥ {ATTACK_CEILING} — reward design bug"
    )
    print(f"  {name:22s} {reward:.4f}  OK")


def test_attack_submit_all_matched(attack_trajectories):
    _assert_under_ceiling(
        "submit_all_matched", _run_attack(attack_trajectories["submit_all_matched"])
    )


def test_attack_submit_all_mismatched(attack_trajectories):
    _assert_under_ceiling(
        "submit_all_mismatched",
        _run_attack(attack_trajectories["submit_all_mismatched"]),
    )


def test_attack_confirm_spam(attack_trajectories):
    _assert_under_ceiling(
        "confirm_spam", _run_attack(attack_trajectories["confirm_spam"])
    )


def test_attack_zero_itc(attack_trajectories):
    _assert_under_ceiling("zero_itc", _run_attack(attack_trajectories["zero_itc"]))


def test_attack_query_only(attack_trajectories):
    _assert_under_ceiling("query_only", _run_attack(attack_trajectories["query_only"]))


def test_attack_overflag_rings(attack_trajectories):
    _assert_under_ceiling(
        "overflag_rings", _run_attack(attack_trajectories["overflag_rings"])
    )
