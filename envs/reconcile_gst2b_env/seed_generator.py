# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic episode generator for reconcile_gst2b_env.

``generate_episode(seed)`` returns a dict with ``purchase_register``,
``gstr_2b``, ``ground_truth``, ``user_request``, ``hero_tier``, and
``company_gstin``. Same seed → identical output.

Every invoice carries ``gstin`` (seller / supplier) and
``counterparty_gstin`` (the other side of the invoice — the company itself
for normal purchases; another supplier inside a planted ring). The
``(gstin → counterparty_gstin)`` edges form the supplier graph on which
``networkx.simple_cycles`` runs in the test suite.

Seed ranges (disjoint, asserted at import time):
- TRAIN_SEEDS      = [0, 500)
- DIVERSITY_SEEDS  = [500, 600)
- HELDOUT_SEEDS    = [9000, 9100)
- HERO_SEEDS       = {9500, 9501, 9502}

Hero seeds have planned (not measured) difficulty. Relabel tiers a/b/c to
easy/medium/hard only after Section F measures actual base-model scores.
"""

from __future__ import annotations

import random
import string
from typing import Any, Dict, List, Optional, Tuple

from .ground_truth import HSN_SLAB_TABLE, gstin_checksum, validate_gstin


TRAIN_SEEDS = set(range(0, 500))
HELDOUT_SEEDS = set(range(9000, 9100))
DIVERSITY_SEEDS = set(range(500, 600))
HERO_SEEDS = {9500, 9501, 9502}

assert not (TRAIN_SEEDS & HELDOUT_SEEDS)
assert not (TRAIN_SEEDS & DIVERSITY_SEEDS)
assert not (HELDOUT_SEEDS & HERO_SEEDS)
assert not (DIVERSITY_SEEDS & HERO_SEEDS)

P_RING = 0.3

MISMATCH_TYPES = (
    "gstin_typo",
    "invoice_number_prefix_drift",
    "tax_slab_off_by_one",
    "supplier_late_filing",
    "amendment_after_2b_freeze",
)

# 3 of the 5 types emit the "mismatched" label; weighting balances the
# non-matched label distribution so no single label exceeds 40%.
MISMATCH_WEIGHTS = (1, 1, 1, 3, 3)

_HSN_KEYS = sorted(HSN_SLAB_TABLE.keys())


def _gen_valid_gstin(rng: random.Random, state_code: str) -> str:
    """Generate a synthetic checksum-valid GSTIN at the given state code."""
    pan_letters = "".join(rng.choices(string.ascii_uppercase, k=5))
    pan_digits = "".join(rng.choices(string.digits, k=4))
    pan_last = rng.choice(string.ascii_uppercase)
    entity = rng.choice(string.digits + string.ascii_uppercase)
    payload = f"{state_code}{pan_letters}{pan_digits}{pan_last}{entity}Z"
    return payload + gstin_checksum(payload)


def _perturb_gstin(rng: random.Random, gstin: str) -> str:
    """1-char edit inside the PAN region; recompute checksum so format stays valid."""
    pos = rng.randint(2, 11)
    old = gstin[pos]
    if 2 <= pos <= 6 or pos == 11:
        pool = string.ascii_uppercase.replace(old, "")
    else:
        pool = string.digits.replace(old, "")
    new_char = rng.choice(pool)
    payload = gstin[:pos] + new_char + gstin[pos + 1 : 14]
    return payload + gstin_checksum(payload)


def _gen_invoice(
    rng: random.Random,
    inv_id: str,
    suppliers: List[str],
    seq: int,
    company_gstin: str,
) -> Dict[str, Any]:
    hsn = rng.choice(_HSN_KEYS)
    slab = HSN_SLAB_TABLE[hsn]
    value_inr = round(rng.uniform(1000, 500_000), 2)
    tax_inr = round(value_inr * float(slab) / 100.0, 2)
    month = rng.randint(4, 12)
    day = rng.randint(1, 28)
    return {
        "invoice_id": inv_id,
        "gstin": rng.choice(suppliers),
        "counterparty_gstin": company_gstin,
        "invoice_no": f"INV/24-25/{seq:04d}",
        "date": f"2026-{month:02d}-{day:02d}",
        "hsn": hsn,
        "value_inr": value_inr,
        "tax_inr": tax_inr,
        "slab_pct": slab,
    }


def _apply_mismatch(
    rng: random.Random,
    books_inv: Dict[str, Any],
    twob_inv: Dict[str, Any],
    mtype: str,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], str, float]:
    """Apply one of the 5 mismatch types to the (books, 2B) pair.

    Returns ``(books_inv, twob_inv_or_None, label, true_itc_eligible_inr)``.
    ITC rule (per Section C decision):
      matched              → books tax (== 2B tax when clean)
      partial              → 2B tax
      mismatched           → 0.0
      only_in_books        → 0.0
      only_in_2b           → 2B tax
    """
    if mtype == "gstin_typo":
        books_inv["gstin"] = _perturb_gstin(rng, books_inv["gstin"])
        return books_inv, twob_inv, "mismatched", 0.0

    if mtype == "invoice_number_prefix_drift":
        books_inv["invoice_no"] = (
            books_inv["invoice_no"].replace("/", "-").replace("_", "-")
        )
        return books_inv, twob_inv, "mismatched", 0.0

    if mtype == "tax_slab_off_by_one":
        current = books_inv["slab_pct"]
        candidates = [s for s in (5, 18, 40) if s != current]
        wrong = rng.choice(candidates)
        books_inv["slab_pct"] = wrong
        books_inv["tax_inr"] = round(books_inv["value_inr"] * wrong / 100.0, 2)
        return books_inv, twob_inv, "mismatched", 0.0

    if mtype == "supplier_late_filing":
        return books_inv, None, "only_in_books", 0.0

    if mtype == "amendment_after_2b_freeze":
        new_value = round(books_inv["value_inr"] * rng.uniform(1.05, 1.25), 2)
        new_tax = round(new_value * float(books_inv["slab_pct"]) / 100.0, 2)
        books_inv["value_inr"] = new_value
        books_inv["tax_inr"] = new_tax
        return books_inv, twob_inv, "partial", float(twob_inv["tax_inr"])

    raise ValueError(f"unknown mismatch type: {mtype}")


def _plant_ring(
    rng: random.Random,
    books_list: List[Dict[str, Any]],
    twob_list: List[Dict[str, Any]],
    suppliers: List[str],
    candidate_ids: List[str],
) -> List[str]:
    """Plant a directed 3-cycle A→B→C→A across 3 invoices.

    For each of 3 chosen invoices, sets ``gstin = supplier_i`` and
    ``counterparty_gstin = supplier_{(i+1) mod 3}``. Mirrors the change to
    the 2B side so both books and 2B show the same suspect counterparty.
    The supplier → counterparty edges form a true directed 3-cycle that
    ``networkx.simple_cycles`` detects.

    Returns list of 3 invoice_ids with ring=True.
    """
    if len(candidate_ids) < 3 or len(suppliers) < 3:
        return []
    ring_suppliers = rng.sample(suppliers, 3)
    ring_ids = rng.sample(candidate_ids, 3)
    id_to_books = {inv["invoice_id"]: inv for inv in books_list}
    id_to_twob = {inv["invoice_id"]: inv for inv in twob_list}
    for i, inv_id in enumerate(ring_ids):
        supplier = ring_suppliers[i]
        counterparty = ring_suppliers[(i + 1) % 3]
        books_inv = id_to_books.get(inv_id)
        if books_inv is not None:
            books_inv["gstin"] = supplier
            books_inv["counterparty_gstin"] = counterparty
        twob_inv = id_to_twob.get(inv_id)
        if twob_inv is not None:
            twob_inv["gstin"] = supplier
            twob_inv["counterparty_gstin"] = counterparty
    return ring_ids


def _hero_params(seed: int) -> Optional[Tuple[int, int, bool, str]]:
    """Return (n_invoices, n_mismatches, add_ring, tier) for hero seeds, else None."""
    if seed == 9500:
        return 20, 2, False, "a"
    if seed == 9501:
        return 50, 5, False, "b"
    if seed == 9502:
        return 80, 8, True, "c"
    return None


def generate_episode(seed: int) -> Dict[str, Any]:
    """Generate a full reconciliation episode. Deterministic in ``seed``."""
    rng = random.Random(seed)

    hero = _hero_params(seed)
    if hero is not None:
        n_invoices, n_mismatches, add_ring, hero_tier = hero
        n_extra_2b_only = 0
    else:
        n_invoices = rng.randint(20, 100)
        # Scaled from spec's 3-8 anchor; spec literal was infeasible against
        # the done-gate matched-share (60-80%) + no-label-exceeds-40% joint
        # constraint on 20-100 invoice episodes. See NOTES.md.
        n_mismatches = max(3, round(n_invoices * rng.uniform(0.18, 0.28)))
        add_ring = rng.random() < P_RING
        hero_tier = None
        n_extra_2b_only = max(1, round(n_mismatches * rng.uniform(0.2, 0.4)))

    # Company GSTIN: the buyer whose books we're reconciling.
    company_state = f"{rng.randint(1, 37):02d}"
    company_gstin = _gen_valid_gstin(rng, company_state)

    n_suppliers = rng.randint(10, min(30, max(10, n_invoices)))
    state_codes = [f"{rng.randint(1, 37):02d}" for _ in range(n_suppliers)]
    suppliers = [_gen_valid_gstin(rng, sc) for sc in state_codes]

    books_list: List[Dict[str, Any]] = []
    twob_list: List[Dict[str, Any]] = []
    gt_map: Dict[str, Dict[str, Any]] = {}

    for i in range(n_invoices):
        inv_id = f"inv_{i:04d}"
        base = _gen_invoice(
            rng, inv_id, suppliers, seq=i + 1, company_gstin=company_gstin
        )
        books_list.append(dict(base))
        twob_list.append(dict(base))
        gt_map[inv_id] = {
            "label": "matched",
            "itc_eligible_inr": base["tax_inr"],
            "mtype": None,
            "in_ring": False,
        }

    # Plant mismatches on a sampled subset.
    mismatch_ids = rng.sample([inv["invoice_id"] for inv in books_list], n_mismatches)
    for mid in mismatch_ids:
        mtype = rng.choices(MISMATCH_TYPES, weights=MISMATCH_WEIGHTS, k=1)[0]
        books_inv = next(i for i in books_list if i["invoice_id"] == mid)
        twob_inv = next(i for i in twob_list if i["invoice_id"] == mid)
        books_inv, new_twob, label, itc = _apply_mismatch(
            rng, books_inv, twob_inv, mtype
        )
        if new_twob is None:
            twob_list = [i for i in twob_list if i["invoice_id"] != mid]
        gt_map[mid]["label"] = label
        gt_map[mid]["itc_eligible_inr"] = itc
        gt_map[mid]["mtype"] = mtype

    # 2B-only extras (label = only_in_2b).
    for j in range(n_extra_2b_only):
        inv_id = f"inv_2bonly_{j:04d}"
        extra = _gen_invoice(
            rng, inv_id, suppliers, seq=10_000 + j, company_gstin=company_gstin
        )
        twob_list.append(extra)
        gt_map[inv_id] = {
            "label": "only_in_2b",
            "itc_eligible_inr": extra["tax_inr"],
            "mtype": None,
            "in_ring": False,
        }

    # Plant ring on invoices that are still "matched" — keeps ring detection
    # orthogonal to mismatch labels.
    if add_ring:
        clean_ids = [iid for iid, gt in gt_map.items() if gt["label"] == "matched"]
        ring_ids = _plant_ring(rng, books_list, twob_list, suppliers, clean_ids)
        for rid in ring_ids:
            gt_map[rid]["in_ring"] = True

    # Build ground truth list, stable-ordered by invoice_id.
    ground_truth: List[Dict[str, Any]] = []
    for inv_id in sorted(gt_map.keys()):
        gt = gt_map[inv_id]
        source = next(
            (i for i in books_list if i["invoice_id"] == inv_id),
            None,
        ) or next(i for i in twob_list if i["invoice_id"] == inv_id)
        ground_truth.append(
            {
                "invoice_id": inv_id,
                "true_gstin_valid": validate_gstin(source["gstin"]),
                "true_hsn_slab": source["slab_pct"],
                "true_label": gt["label"],
                "true_itc_eligible_inr": gt["itc_eligible_inr"],
                "true_in_circular_ring": gt["in_ring"],
                "true_mismatch_type": gt["mtype"],
            }
        )

    return {
        "purchase_register": books_list,
        "gstr_2b": twob_list,
        "ground_truth": ground_truth,
        "user_request": (
            "Reconcile the July 2026 purchase register against GSTR-2B, "
            "label each invoice, compute ITC eligibility, and flag any "
            "Rule 36(4) violations or circular-trading rings."
        ),
        "hero_tier": hero_tier,
        "company_gstin": company_gstin,
    }
