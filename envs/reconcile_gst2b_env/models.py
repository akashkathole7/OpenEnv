# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wire types and hidden ground-truth records for reconcile_gst2b_env.

All types here are Pydantic models (JSON-serializable). Public Observation
fields never carry ``true_*`` or ``gt_*`` data; those prefixes are reserved
for server-only ground truth that must stay hidden from the agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


Verb = Literal[
    # Query (7) — no state mutation
    "get_schema",
    "list_gstins",
    "fuzzy_search_gstin",
    "get_invoice",
    "list_invoices_by_supplier",
    "get_2b_row",
    "get_hsn_slab",
    # Mutate (7) — scored against ground truth
    "mark_matched",
    "mark_mismatched",
    "mark_only_in_books",
    "mark_only_in_2b",
    "mark_partial_match",
    "flag_circular_ring",
    "request_amendment",
    # Meta (2)
    "confirm_with_user",
    "submit",
]


class InvoiceGroundTruth(BaseModel):
    """Hidden ground truth for a single invoice. Server-only."""

    true_gstin_valid: bool
    true_hsn_slab: Literal[0, 0.25, 3, 5, 12, 18, 28, 40]
    true_label: Literal[
        "matched", "mismatched", "only_in_books", "only_in_2b", "partial"
    ]
    true_itc_eligible_inr: float
    true_in_circular_ring: bool


class ReconcileAction(Action):
    """Agent action: discriminator ``verb`` + untyped ``payload`` dict.

    Payload shape varies per verb (e.g. ``invoice_id``, ``reason``, ``gstins``)
    and is validated by the environment at dispatch time, not at parse time.
    """

    verb: Verb
    payload: Dict[str, Any] = Field(default_factory=dict)


class ReconcileObservation(Observation):
    """Public observation. Must not contain any ``true_*`` / ``gt_*`` field."""

    user_request: str = ""
    last_tool_result: Dict[str, Any] = Field(default_factory=dict)
    step_budget: int = 0
    invoices_remaining_count: int = 0


class ReconcileState(State):
    """Server state — public fields plus hidden ground truth.

    ``gt_*`` / ``true_*`` fields hold the simulation's ground truth and must
    never be serialized into ``ReconcileObservation``.
    """

    env_version: str = "gst2b-v1.0"
    env_schema: Dict[str, Any] = Field(default_factory=dict)
    invoices: List[Dict[str, Any]] = Field(default_factory=list)
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)

    gt_invoices: List[InvoiceGroundTruth] = Field(default_factory=list)
    true_itc_claimed_inr: float = 0.0
    true_rule_36_4_violated: bool = False
