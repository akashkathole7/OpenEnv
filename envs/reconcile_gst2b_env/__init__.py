# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""reconcile_gst2b_env — GST Input-Tax-Credit reconciliation environment."""

# sys.path self-heal: this package's client.py imports `openenv.core`, which
# lives in <repo_root>/src/. When the package is loaded via `python -m
# envs.reconcile_gst2b_env.*` from a bare `python` invocation (no PYTHONPATH
# pointing at src/), that import fails before the module body of the target
# script runs. Adding src/ to sys.path here — at the earliest possible moment
# during package initialization — makes the package robust to missing
# PYTHONPATH config (e.g. Colab notebooks whose shell `!python` invocations
# inherit only the default subprocess env).
import os as _os
import sys as _sys

_here = _os.path.dirname(_os.path.abspath(__file__))
_src_path = _os.path.normpath(_os.path.join(_here, _os.pardir, _os.pardir, "src"))
if _os.path.isdir(_src_path) and _src_path not in _sys.path:
    _sys.path.insert(0, _src_path)
del _os, _sys, _here, _src_path

from .client import ReconcileGST2BEnv
from .models import (
    InvoiceGroundTruth,
    ReconcileAction,
    ReconcileObservation,
    ReconcileState,
)

__all__ = [
    "InvoiceGroundTruth",
    "ReconcileAction",
    "ReconcileObservation",
    "ReconcileState",
    "ReconcileGST2BEnv",
]
