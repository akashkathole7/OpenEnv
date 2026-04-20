# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared policy implementations for diversity / baseline / ablation scripts.

Policies are callables ``(obs, rng, context) -> ReconcileAction``. Context is
populated after a ``get_schema`` call so policies that need invoice_ids can
discover them via the env's query surface.

These policies are CPU-only heuristic *proxies* for LLM behavior. Real LLM
runs happen in Colab via ``baseline.py --real`` (scaffolded; requires GPU).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from envs.reconcile_gst2b_env.models import ReconcileAction


ALL_VERBS = (
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
)

_MARK_LABELS = (
    "mark_matched",
    "mark_mismatched",
    "mark_only_in_books",
    "mark_only_in_2b",
    "mark_partial_match",
)


def _pick_invoice_id(context: Dict[str, Any], rng: random.Random) -> str:
    ids = context.get("invoice_ids") or [f"inv_{i:04d}" for i in range(20)]
    return rng.choice(ids)


def _pick_gstins(context: Dict[str, Any], rng: random.Random, k: int = 3) -> List[str]:
    pool = context.get("gstins") or []
    if len(pool) < k:
        return pool[:k]
    return rng.sample(pool, k)


def _build_payload(
    verb: str, context: Dict[str, Any], rng: random.Random
) -> Dict[str, Any]:
    if verb in _MARK_LABELS:
        payload: Dict[str, Any] = {"invoice_id": _pick_invoice_id(context, rng)}
        if verb == "mark_partial_match":
            payload["delta_inr"] = round(rng.uniform(0, 500), 2)
        elif verb == "mark_mismatched":
            payload["reason"] = rng.choice(
                ("value", "tax", "date", "gstin", "hsn", "invoice_no")
            )
        return payload
    if verb == "get_invoice" or verb == "get_2b_row":
        return {"invoice_id": _pick_invoice_id(context, rng)}
    if verb == "list_invoices_by_supplier":
        pool = context.get("gstins") or []
        return {"gstin": rng.choice(pool)} if pool else {"gstin": ""}
    if verb == "fuzzy_search_gstin":
        pool = context.get("gstins") or [""]
        seed_gstin = rng.choice(pool) if pool else ""
        return {"query": seed_gstin[:10]}
    if verb == "get_hsn_slab":
        return {"hsn": rng.choice(("8517", "8471", "2402", "6901"))}
    if verb == "flag_circular_ring":
        return {"gstins": _pick_gstins(context, rng)}
    if verb == "request_amendment":
        return {
            "invoice_id": _pick_invoice_id(context, rng),
            "field": "value_inr",
            "value": rng.uniform(1000, 50000),
        }
    if verb == "confirm_with_user":
        return {"question": "please confirm this invoice"}
    return {}


def random_policy(obs, rng: random.Random, context: Dict[str, Any]) -> ReconcileAction:
    """Uniform-random policy across all 16 verbs."""
    verb = rng.choice(ALL_VERBS)
    return ReconcileAction(verb=verb, payload=_build_payload(verb, context, rng))


def raw_policy(obs, rng: random.Random, context: Dict[str, Any]) -> ReconcileAction:
    """Mock for an unprompted weak LLM.

    Weighted slightly toward ``mark_matched`` to simulate LLM majority-class
    bias, but no schema/invoice querying.
    """
    verbs = (
        "get_schema",
        "mark_matched",
        "mark_mismatched",
        "mark_only_in_books",
        "mark_only_in_2b",
        "mark_partial_match",
        "confirm_with_user",
        "submit",
    )
    weights = (2, 8, 2, 2, 2, 2, 1, 1)
    verb = rng.choices(verbs, weights=weights, k=1)[0]
    return ReconcileAction(verb=verb, payload=_build_payload(verb, context, rng))


def prompted_policy(
    obs, rng: random.Random, context: Dict[str, Any]
) -> ReconcileAction:
    """Mock for a prompted LLM with schema introspection.

    Pattern: 1 schema query, then aggressively mark_matched across first N
    invoices (majority-class bet), submit before budget runs out. Scores
    better than random because it consistently uses the query gate for R3
    and lands R1/R2 credit from the majority-matched labels.
    """
    step = context.get("prompted_step", 0)
    context["prompted_step"] = step + 1
    max_marks = 20

    if step == 0:
        return ReconcileAction(verb="get_schema", payload={})

    ids = context.get("invoice_ids") or []
    labeled = context.setdefault("prompted_labeled", set())
    unlabeled = [i for i in ids if i not in labeled]

    if not unlabeled or len(labeled) >= max_marks or obs.step_budget <= 2:
        return ReconcileAction(verb="submit", payload={})

    target = unlabeled[0]
    labeled.add(target)
    return ReconcileAction(verb="mark_matched", payload={"invoice_id": target})


def placeholder_policy(
    obs, rng: random.Random, context: Dict[str, Any]
) -> ReconcileAction:
    """Random-init marker: pure uniform-random. Same as random_policy."""
    return random_policy(obs, rng, context)


def oracle_heuristic_policy(
    obs, rng: random.Random, context: Dict[str, Any]
) -> ReconcileAction:
    """Ground-truth-aware reference used ONLY for diversity gradient checks.

    Not a real baseline. Uses the hidden ``gt_invoices`` state to mark each
    invoice with its true label, one step at a time. Exists purely to confirm
    the reward signal has gradient (a correctly-labeling policy scores higher
    than random). Do not use for training, evaluation, or pitch demos.

    Takes ``env`` from context["_env"] — set by run_episode when the policy
    is this oracle variant.
    """
    env = context.get("_env")
    step = context.setdefault("oracle_step", 0)
    context["oracle_step"] = step + 1

    if step == 0:
        return ReconcileAction(verb="get_schema", payload={})

    if env is None:
        # Oracle has no ground truth source — fall back to prompted.
        return prompted_policy(obs, rng, context)

    gt_rows = env.state.gt_invoices
    labeled = context.setdefault("oracle_labeled", set())
    pending = [g for g in gt_rows if g.invoice_id not in labeled]
    if not pending or obs.step_budget <= 2:
        return ReconcileAction(verb="submit", payload={})

    target = pending[0]
    labeled.add(target.invoice_id)
    verb_map = {
        "matched": "mark_matched",
        "mismatched": "mark_mismatched",
        "only_in_books": "mark_only_in_books",
        "only_in_2b": "mark_only_in_2b",
        "partial": "mark_partial_match",
    }
    verb = verb_map[target.true_label]
    payload: Dict[str, Any] = {"invoice_id": target.invoice_id}
    if verb == "mark_mismatched":
        payload["reason"] = "value"
    elif verb == "mark_partial_match":
        payload["delta_inr"] = 0.0
    return ReconcileAction(verb=verb, payload=payload)


# Public alias: "heuristic" in diversity_test is the oracle above.
heuristic_policy = oracle_heuristic_policy


def populate_context_from_result(
    context: Dict[str, Any], last_tool_result: Dict[str, Any]
) -> None:
    """Best-effort: extract invoice_ids and gstins from tool result dicts."""
    if "schema" in last_tool_result:
        s = last_tool_result["schema"]
        n_pr = int(s.get("n_purchase_register", 0))
        # Synthesize ids matching the seed_generator convention.
        ids = [f"inv_{i:04d}" for i in range(n_pr)]
        # Also include 2B-only ids if present.
        n_2b_only = max(0, int(s.get("n_gstr_2b", 0)) - n_pr)
        ids.extend(f"inv_2bonly_{j:04d}" for j in range(n_2b_only))
        context["invoice_ids"] = ids
    if "gstins" in last_tool_result:
        context["gstins"] = list(last_tool_result["gstins"])


def run_episode(
    env,
    policy,
    seed: int,
    rng_seed: int,
    mode: str = "warmup",
    max_steps: int = 50,
) -> Dict[str, Any]:
    """Run a single episode and return a log dict."""
    rng = random.Random(rng_seed)
    obs = env.reset(seed=seed, mode=mode)
    # Oracle policy needs env access for ground-truth lookup.
    context: Dict[str, Any] = {"_env": env}
    trajectory_log: List[Dict[str, Any]] = []
    step_count = 0
    while not obs.done and step_count < max_steps + 5:
        action = policy(obs, rng, context)
        obs = env.step(action)
        populate_context_from_result(context, obs.last_tool_result or {})
        trajectory_log.append(
            {
                "step": step_count + 1,
                "verb": action.verb,
                "payload": action.payload,
                "reward": obs.reward,
                "done": obs.done,
            }
        )
        step_count += 1

    breakdown = env.state.reward_breakdown or {
        "R1": 0.0,
        "R2": 0.0,
        "R3": 0.0,
        "R4": 0.0,
        "total": obs.reward if obs.reward is not None else 0.0,
    }
    catastrophic = bool(
        env.state.true_rule_36_4_violated
        or (obs.metadata or {}).get("termination_reason") == "submit_before_query"
    )
    return {
        "seed": seed,
        "mode": mode,
        "component_rewards": {
            k: round(float(v), 4) for k, v in breakdown.items() if k != "total"
        },
        "total": round(float(breakdown.get("total", 0.0)), 4),
        "total_steps": step_count,
        "termination_reason": (obs.metadata or {}).get("termination_reason", "unknown"),
        "done": obs.done,
        "catastrophic_corruption": catastrophic,
        "trajectory": trajectory_log,
    }
