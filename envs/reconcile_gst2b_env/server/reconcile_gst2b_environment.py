# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Server-side implementation of the reconcile_gst2b_env environment.

Orchestrates episode data from ``seed_generator``, dispatches the 16 typed
agent verbs, applies terminal conditions, and computes the 4-component
composite reward.
"""

from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..ground_truth import get_slab, HSN_SLAB_TABLE
    from ..models import (
        InvoiceGroundTruth,
        ReconcileAction,
        ReconcileObservation,
        ReconcileState,
    )
    from ..rewards import composite_reward, MAX_STEPS, QUERY_VERBS
    from ..seed_generator import generate_episode
except ImportError:
    from ground_truth import get_slab, HSN_SLAB_TABLE
    from models import (
        InvoiceGroundTruth,
        ReconcileAction,
        ReconcileObservation,
        ReconcileState,
    )
    from rewards import composite_reward, MAX_STEPS, QUERY_VERBS
    from seed_generator import generate_episode


class ReconcileGST2BEnvironment(Environment):
    """GST Input-Tax-Credit reconciliation environment.

    Single-session per instance (``SUPPORTS_CONCURRENT_SESSIONS=False``).
    The 16 verbs split into 7 query, 7 mutate, 2 meta — query verbs are
    read-only, mutate verbs record per-invoice labels used by the reward
    scorer, meta verbs (``confirm_with_user``, ``submit``) are
    control-flow only.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    VALID_MODES = ("warmup", "hardened")

    def __init__(self) -> None:
        self._state: ReconcileState = ReconcileState(
            episode_id=str(uuid4()), step_count=0
        )
        self._trajectory: List[ReconcileAction] = []
        self._episode: Dict[str, Any] = {}
        self._mode: str = "warmup"

    # ---------- Gym interface ----------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        mode: str = "warmup",
        **kwargs: Any,
    ) -> ReconcileObservation:
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        self._mode = mode
        ep = generate_episode(seed if seed is not None else 0)
        self._episode = ep
        combined_invoices = list(ep["purchase_register"]) + [
            inv for inv in ep["gstr_2b"] if inv["invoice_id"].startswith("inv_2bonly_")
        ]
        self._state = ReconcileState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            env_version="gst2b-v1.0",
            env_schema=self._build_schema(ep),
            invoices=combined_invoices,
            reward_breakdown={},
            gt_invoices=[InvoiceGroundTruth(**g) for g in ep["ground_truth"]],
            gt_purchase_register=list(ep["purchase_register"]),
            gt_gstr_2b=list(ep["gstr_2b"]),
            gt_company_gstin=ep["company_gstin"],
            true_itc_claimed_inr=0.0,
            true_rule_36_4_violated=False,
        )
        self._trajectory = []
        return ReconcileObservation(
            user_request=ep["user_request"],
            last_tool_result={"status": "ready"},
            step_budget=MAX_STEPS,
            invoices_remaining_count=len(combined_invoices),
            done=False,
            reward=0.0,
            metadata={"termination_reason": "reset", "env_version": "gst2b-v1.0"},
        )

    def step(
        self,
        action: ReconcileAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ReconcileObservation:
        self._state.step_count += 1
        self._trajectory.append(action)

        tool_result = self._dispatch(action)

        verb = action.verb
        done = False
        reward: float = 0.0
        termination = "continue"

        if verb == "submit":
            prior_queries = sum(
                1 for a in self._trajectory[:-1] if a.verb in QUERY_VERBS
            )
            if prior_queries < 1:
                done = True
                reward = -1.0
                termination = "submit_before_query"
                self._state.reward_breakdown = {
                    "R1": 0.0,
                    "R2": 0.0,
                    "R3": 0.0,
                    "R4": 0.0,
                    "total": reward,
                }
            else:
                breakdown = composite_reward(self._state, self._trajectory)
                self._state.reward_breakdown = breakdown
                done = True
                reward = breakdown["total"]
                termination = "submitted"
        elif self._state.step_count >= MAX_STEPS:
            breakdown = composite_reward(self._state, self._trajectory)
            self._state.reward_breakdown = breakdown
            done = True
            reward = breakdown["total"]
            termination = "step_budget_exhausted"
        elif self._mode == "hardened" and self._rule_36_4_violated_live():
            breakdown = composite_reward(self._state, self._trajectory)
            self._state.reward_breakdown = breakdown
            self._state.true_rule_36_4_violated = True
            done = True
            reward = breakdown["total"]
            termination = "rule_36_4_violation_hardened"

        obs = ReconcileObservation(
            user_request=self._episode.get("user_request", ""),
            last_tool_result=tool_result,
            step_budget=max(0, MAX_STEPS - self._state.step_count),
            invoices_remaining_count=self._count_unlabeled_invoices(),
            done=done,
            reward=reward,
            metadata={
                "termination_reason": termination,
                "env_version": self._state.env_version,
                "reward_breakdown": self._state.reward_breakdown,
            },
        )
        return obs

    @property
    def state(self) -> ReconcileState:
        return self._state

    # ---------- Dispatch ----------

    def _dispatch(self, action: ReconcileAction) -> Dict[str, Any]:
        verb = action.verb
        payload = action.payload or {}
        if verb == "get_schema":
            return {"schema": self._state.env_schema}
        if verb == "list_gstins":
            gstins = sorted(
                {inv["gstin"] for inv in self._state.gt_purchase_register}
                | {inv["gstin"] for inv in self._state.gt_gstr_2b}
            )
            return {"gstins": gstins}
        if verb == "fuzzy_search_gstin":
            return self._fuzzy_search_gstin(payload.get("query", ""))
        if verb == "get_invoice":
            return self._get_invoice(payload.get("invoice_id", ""))
        if verb == "list_invoices_by_supplier":
            return self._list_invoices_by_supplier(payload.get("gstin", ""))
        if verb == "get_2b_row":
            return self._get_2b_row(payload.get("invoice_id", ""))
        if verb == "get_hsn_slab":
            hsn = payload.get("hsn", "")
            try:
                return {"hsn": hsn, "slab_pct": get_slab(hsn)}
            except KeyError:
                return {"hsn": hsn, "slab_pct": None, "error": "hsn_not_in_table"}
        if verb in (
            "mark_matched",
            "mark_mismatched",
            "mark_only_in_books",
            "mark_only_in_2b",
            "mark_partial_match",
        ):
            return {"acknowledged": True, "invoice_id": payload.get("invoice_id")}
        if verb == "flag_circular_ring":
            return {"flagged_gstins": list(payload.get("gstins", []))}
        if verb == "request_amendment":
            return {
                "acknowledged": True,
                "invoice_id": payload.get("invoice_id"),
                "field": payload.get("field"),
            }
        if verb == "confirm_with_user":
            return {"acknowledged": True, "question": payload.get("question", "")}
        if verb == "submit":
            return {"submitted": True}
        return {"error": f"unknown_verb:{verb}"}

    # ---------- Query helpers ----------

    def _fuzzy_search_gstin(self, query: str) -> Dict[str, Any]:
        universe = sorted(
            {inv["gstin"] for inv in self._state.gt_purchase_register}
            | {inv["gstin"] for inv in self._state.gt_gstr_2b}
        )
        matches = difflib.get_close_matches(query, universe, n=5, cutoff=0.3)
        return {"query": query, "matches": matches}

    def _get_invoice(self, invoice_id: str) -> Dict[str, Any]:
        for inv in self._state.gt_purchase_register:
            if inv["invoice_id"] == invoice_id:
                return {"source": "purchase_register", "invoice": inv}
        for inv in self._state.gt_gstr_2b:
            if inv["invoice_id"] == invoice_id:
                return {"source": "gstr_2b", "invoice": inv}
        return {"error": "invoice_not_found", "invoice_id": invoice_id}

    def _list_invoices_by_supplier(self, gstin: str) -> Dict[str, Any]:
        books = [
            inv for inv in self._state.gt_purchase_register if inv["gstin"] == gstin
        ]
        twob = [inv for inv in self._state.gt_gstr_2b if inv["gstin"] == gstin]
        return {"gstin": gstin, "purchase_register": books, "gstr_2b": twob}

    def _get_2b_row(self, invoice_id: str) -> Dict[str, Any]:
        for inv in self._state.gt_gstr_2b:
            if inv["invoice_id"] == invoice_id:
                return {"invoice": inv}
        return {"error": "not_in_2b", "invoice_id": invoice_id}

    # ---------- Helpers ----------

    def _rule_36_4_violated_live(self) -> bool:
        """Per-supplier claim > 2B cap, computed on trajectory so far."""
        from collections import defaultdict

        per_supplier_claim: Dict[str, float] = defaultdict(float)
        partial_deltas: Dict[str, float] = {}
        labels: Dict[str, str] = {}
        for act in self._trajectory:
            p = act.payload or {}
            if act.verb == "mark_partial_match":
                partial_deltas[p.get("invoice_id", "")] = float(p.get("delta_inr", 0.0))
            if act.verb in (
                "mark_matched",
                "mark_mismatched",
                "mark_only_in_books",
                "mark_only_in_2b",
                "mark_partial_match",
            ):
                iid = p.get("invoice_id")
                if iid is not None:
                    labels[iid] = {
                        "mark_matched": "matched",
                        "mark_mismatched": "mismatched",
                        "mark_only_in_books": "only_in_books",
                        "mark_only_in_2b": "only_in_2b",
                        "mark_partial_match": "partial",
                    }[act.verb]
        inv_tax = {
            inv["invoice_id"]: float(inv.get("tax_inr", 0.0))
            for inv in self._state.gt_purchase_register
        }
        for inv in self._state.gt_gstr_2b:
            inv_tax.setdefault(inv["invoice_id"], float(inv.get("tax_inr", 0.0)))
        supplier = {
            inv["invoice_id"]: inv.get("gstin", "")
            for inv in self._state.gt_purchase_register
        }
        for inv in self._state.gt_gstr_2b:
            supplier.setdefault(inv["invoice_id"], inv.get("gstin", ""))
        for inv_id, lab in labels.items():
            tax = inv_tax.get(inv_id, 0.0)
            if lab in ("matched", "only_in_2b"):
                per_supplier_claim[supplier.get(inv_id, "")] += tax
            elif lab == "partial":
                per_supplier_claim[supplier.get(inv_id, "")] += max(
                    0.0, tax - partial_deltas.get(inv_id, 0.0)
                )
        per_supplier_cap: Dict[str, float] = defaultdict(float)
        for inv in self._state.gt_gstr_2b:
            per_supplier_cap[inv["gstin"]] += float(inv.get("tax_inr", 0.0))
        _TOL = 1e-6
        for s, claim in per_supplier_claim.items():
            if claim > per_supplier_cap.get(s, 0.0) + _TOL:
                return True
        return False

    def _count_unlabeled_invoices(self) -> int:
        labeled = set()
        for a in self._trajectory:
            if a.verb in (
                "mark_matched",
                "mark_mismatched",
                "mark_only_in_books",
                "mark_only_in_2b",
                "mark_partial_match",
            ):
                inv_id = (a.payload or {}).get("invoice_id")
                if inv_id is not None:
                    labeled.add(inv_id)
        total = {i["invoice_id"] for i in self._state.invoices}
        return len(total - labeled)

    def _build_schema(self, ep: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "env_version": "gst2b-v1.0",
            "max_steps": MAX_STEPS,
            "company_gstin": ep.get("company_gstin", ""),
            "n_purchase_register": len(ep.get("purchase_register", [])),
            "n_gstr_2b": len(ep.get("gstr_2b", [])),
            "n_hsn_slab_table_entries": len(HSN_SLAB_TABLE),
            "label_classes": [
                "matched",
                "mismatched",
                "only_in_books",
                "only_in_2b",
                "partial",
            ],
            "verbs": sorted(
                [
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
            ),
        }


__all__ = ["ReconcileGST2BEnvironment"]
