# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ground-truth constants, GSTIN validator, slab lookup, and reward-clamp.

Contents:
- ``clamp``: single source of truth for reward-value clamping; rewards.py
  imports this.
- ``gstin_checksum`` / ``validate_gstin``: base-36 checksum and 15-char
  format validator.
- ``PRIMARY_SLABS`` / ``RESIDUAL_SLABS``: GST 2.0 slab structure.
- ``HSN_SLAB_TABLE``: HSN-to-slab reference (synthetic, CBIC-unverified).
- ``get_slab``: HSN lookup; raises ``KeyError`` on miss.
- ``GSTIN_TEST_VECTORS``: 5 checksum-valid GSTINs (1 real anchor + 4 synthetic).
"""

from __future__ import annotations

from typing import Dict, List, Union


def clamp(x: float, eps: float = 0.01) -> float:
    """Clamp ``x`` into ``[eps, 1.0 - eps]``.

    Boundary reward values (0.0 / 1.0) fail many public validators; all reward
    components pass through this before being returned.
    """
    if x < eps:
        return eps
    if x > 1.0 - eps:
        return 1.0 - eps
    return x


# === GSTIN ===

_GSTIN_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_GSTIN_MOD = 36
_VALID_STATE_CODES = {f"{i:02d}" for i in range(1, 38)} | {"97", "99"}


def gstin_checksum(first_14: str) -> str:
    """Base-36 GSTIN checksum over the 14-char payload.

    Luhn-like: iterate right to left with alternating weights 2/1, where each
    weighted value is folded as ``(digit // 36) + (digit % 36)``.
    """
    if len(first_14) != 14:
        raise ValueError(f"gstin_checksum requires 14 chars, got {len(first_14)}")
    factor = 2
    total = 0
    for ch in reversed(first_14):
        code_point = _GSTIN_CHARSET.index(ch)
        digit = code_point * factor
        digit = (digit // _GSTIN_MOD) + (digit % _GSTIN_MOD)
        total += digit
        factor = 1 if factor == 2 else 2
    check = (_GSTIN_MOD - (total % _GSTIN_MOD)) % _GSTIN_MOD
    return _GSTIN_CHARSET[check]


def validate_gstin(g: str) -> bool:
    """Validate GSTIN: length 15, state code, PAN shape, 14th char 'Z', checksum."""
    if not isinstance(g, str) or len(g) != 15:
        return False
    if g[:2] not in _VALID_STATE_CODES:
        return False
    pan = g[2:12]
    if not (pan[0:5].isalpha() and pan[5:9].isdigit() and pan[9:10].isalpha()):
        return False
    if not g[12].isalnum():
        return False
    if g[13] != "Z":
        return False
    try:
        expected = gstin_checksum(g[:14])
    except ValueError:
        return False
    return g[14] == expected


# === GST 2.0 slabs ===

PRIMARY_SLABS: List[int] = [0, 5, 18, 40]

RESIDUAL_SLABS: Dict[Union[int, float], str] = {
    12: "fly_ash_bricks_and_earthen_tiles_only",
    28: "tobacco_pending_cess_discharge_only",
    3: "gold_silver_jewellery",
    0.25: "rough_diamonds",
}

# HSN prefix -> slab.
# NOTE: unverified against CBIC. Synthetic training data only — this
# environment is an RL reconciliation task, not a tax-advice tool.
HSN_SLAB_TABLE: Dict[str, Union[int, float]] = {
    # 0% — exempt
    "0401": 0,
    "0701": 0,
    "1001": 0,
    "3004": 0,
    "4901": 0,
    # 5% — essentials
    "0402": 5,
    "1006": 5,
    "1701": 5,
    "3001": 5,
    "6401": 5,
    "8702": 5,
    "2710": 5,
    # 18% — standard
    "8517": 18,
    "8471": 18,
    "8703": 18,
    "8708": 18,
    "9403": 18,
    "7321": 18,
    "8450": 18,
    "8415": 18,
    "3304": 18,
    "9503": 18,
    "6402": 18,
    "4011": 18,
    # 40% — luxury / sin
    "2402": 40,
    "2202": 40,
    "8901": 40,
    "2403": 40,
    # 12% residual
    "6815": 12,
    "6901": 12,
    # Special
    "7113": 3,
    "7102": 0.25,
}


def get_slab(hsn: str) -> Union[int, float]:
    """Lookup slab for an HSN prefix. Raises ``KeyError`` on miss (explicit, no default)."""
    return HSN_SLAB_TABLE[hsn]


# Synthetic, checksum-valid, not real taxpayers.
# Use only for test fixtures; do not log or publish as real GSTINs.
GSTIN_TEST_VECTORS: List[str] = [
    "27AAPFU0939F1ZV",  # real-format anchor (public test vector, verified checksum)
    "27AAAAA0001A1Z1",  # synthetic, state 27 (MH)
    "29AAAAA0001A1ZX",  # synthetic, state 29 (KA)
    "07AAAAA0001A1Z3",  # synthetic, state 07 (DL)
    "33AAAAA0001A1Z8",  # synthetic, state 33 (TN)
]
