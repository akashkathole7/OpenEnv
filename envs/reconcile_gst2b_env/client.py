# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HTTP/WebSocket client for the reconcile_gst2b_env environment."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import ReconcileAction, ReconcileObservation, ReconcileState


class ReconcileGST2BEnv(
    EnvClient[ReconcileAction, ReconcileObservation, ReconcileState]
):
    """Client for the GST Input-Tax-Credit reconciliation environment.

    Example:
        >>> with ReconcileGST2BEnv(base_url="http://localhost:8000") as env:
        ...     obs = env.reset(seed=0).observation
        ...     obs = env.step(ReconcileAction(verb="get_schema", payload={})).observation
        ...     obs = env.step(ReconcileAction(verb="submit", payload={})).observation
    """

    def _step_payload(self, action: ReconcileAction) -> Dict[str, Any]:
        return {"verb": action.verb, "payload": action.payload}

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[ReconcileObservation]:
        obs_data = payload.get("observation", {}) or {}
        observation = ReconcileObservation(
            user_request=obs_data.get("user_request", ""),
            last_tool_result=obs_data.get("last_tool_result", {}) or {},
            step_budget=int(obs_data.get("step_budget", 0)),
            invoices_remaining_count=int(obs_data.get("invoices_remaining_count", 0)),
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
            metadata=obs_data.get("metadata", {}) or {},
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ReconcileState:
        return ReconcileState(
            episode_id=payload.get("episode_id"),
            step_count=int(payload.get("step_count", 0)),
            env_version=payload.get("env_version", "gst2b-v1.0"),
            env_schema=payload.get("env_schema", {}) or {},
            invoices=payload.get("invoices", []) or [],
            reward_breakdown=payload.get("reward_breakdown", {}) or {},
            gt_invoices=payload.get("gt_invoices", []) or [],
            gt_purchase_register=payload.get("gt_purchase_register", []) or [],
            gt_gstr_2b=payload.get("gt_gstr_2b", []) or [],
            gt_company_gstin=payload.get("gt_company_gstin", ""),
            true_itc_claimed_inr=float(payload.get("true_itc_claimed_inr", 0.0)),
            true_rule_36_4_violated=bool(payload.get("true_rule_36_4_violated", False)),
        )
