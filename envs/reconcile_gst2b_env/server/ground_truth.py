# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ground-truth constants and GSTIN validator for reconcile_gst2b_env.

Server-only. Client code must not import this module (invariant: client-server
separation).

Contents:
- GST 2.0 slab structure (PRIMARY_SLABS, RESIDUAL_SLABS)
- HSN-to-slab reference table (HSN_SLAB_TABLE; CBIC-unverified, see NOTES.md)
- GSTIN 15-char format and base-36 checksum validator
"""

from __future__ import annotations

from typing import Dict, List, Union


# GST 2.0 primary slabs (effective 22 Sep 2025).
PRIMARY_SLABS: List[int] = [0, 5, 18, 40]

# Residual slabs with narrow applicability.
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


_GSTIN_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_GSTIN_MOD = 36
_VALID_STATE_CODES = {f"{i:02d}" for i in range(1, 38)} | {"97", "99"}


def _gstin_check_digit(payload14: str) -> str:
    """Base-36 GSTIN checksum over the first 14 chars (right-to-left, alternating 2/1)."""
    factor = 2
    total = 0
    for ch in reversed(payload14):
        code_point = _GSTIN_CHARSET.index(ch)
        digit = code_point * factor
        digit = (digit // _GSTIN_MOD) + (digit % _GSTIN_MOD)
        total += digit
        factor = 1 if factor == 2 else 2
    check = (_GSTIN_MOD - (total % _GSTIN_MOD)) % _GSTIN_MOD
    return _GSTIN_CHARSET[check]


# Synthetic, checksum-valid, not real taxpayers.
# Use only for test fixtures; do not log or publish as real GSTINs.
GSTIN_TEST_VECTORS: List[str] = [
    "27AAPFU0939F1ZV",  # real-format anchor (public test vector, verified checksum)
    "27AAAAA0001A1Z1",  # synthetic, state 27 (MH)
    "29AAAAA0001A1ZX",  # synthetic, state 29 (KA)
    "07AAAAA0001A1Z3",  # synthetic, state 07 (DL)
    "33AAAAA0001A1Z8",  # synthetic, state 33 (TN)
]


def is_valid_gstin(gstin: str) -> bool:
    """Validate 15-char GSTIN: state code, PAN shape, 14th char = 'Z', checksum."""
    if not isinstance(gstin, str) or len(gstin) != 15:
        return False
    if gstin[:2] not in _VALID_STATE_CODES:
        return False
    pan = gstin[2:12]
    if not (pan[0:5].isalpha() and pan[5:9].isdigit() and pan[9:10].isalpha()):
        return False
    if not (gstin[12].isalnum()):
        return False
    if gstin[13] != "Z":
        return False
    try:
        expected = _gstin_check_digit(gstin[:14])
    except ValueError:
        return False
    return gstin[14] == expected
