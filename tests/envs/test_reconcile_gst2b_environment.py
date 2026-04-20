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
