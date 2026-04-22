#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate synthetic SFT expert trajectories via the env's oracle policy.

Empirically motivated by data/training_log_qwen3_*_partial.json: GRPO from
base Qwen3 instruct checkpoints fails to chain label actions at ≤4B scale
across three distinct failure modes. The fix is tool-SFT warm-start before
GRPO — teach the policy the action format and label-commitment preference
first, then let GRPO polish.

This script runs the ground-truth-aware oracle (``oracle_heuristic_policy``
in ``_policies.py``) over a seed range, reconstructs each step as a
chat-template message pair, filters by composite reward threshold, and
emits a JSONL dataset compatible with TRL's SFTTrainer.

Usage (onsite, CPU-only, ~2 h for 500 seeds):
    PYTHONPATH=src:envs uv run python -m \\
        envs.reconcile_gst2b_env.scripts.generate_sft_trajectories \\
        --start-seed 100 --end-seed 600 \\
        --output-path data/sft_trajectories.jsonl \\
        --min-total 0.50

Dry-run (5 seeds, ~30 s — use to verify format):
    PYTHONPATH=src:envs uv run python -m \\
        envs.reconcile_gst2b_env.scripts.generate_sft_trajectories --dry-run

Output format — one JSON object per line, each line a full conversation:
    {
      "seed": <int>,
      "total": <float, final composite reward>,
      "breakdown": {"R1": ..., "R2": ..., "R3": ..., "R4": ..., "total": ...},
      "n_turns": <int, number of assistant turns in the conversation>,
      "messages": [
        {"role": "system", "content": "You are a GST reconciliation agent..."},
        {"role": "user", "content": "Step budget remaining: 50\\n..."},
        {"role": "assistant", "content": "<tool_call>{\\"name\\": \\"get_schema\\", \\"arguments\\": {}}</tool_call>"},
        ... (N user/assistant pairs interleaved) ...
      ]
    }

SFT training side (on-site):
  from trl import SFTConfig, SFTTrainer
  from datasets import load_dataset
  ds = load_dataset("json", data_files="data/sft_trajectories.jsonl", split="train")
  trainer = SFTTrainer(
      model=model,
      args=SFTConfig(..., packing=False),
      train_dataset=ds,
      tokenizer=tokenizer,  # with enable_thinking=False monkeypatch
  )
  trainer.train()
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from envs.reconcile_gst2b_env.scripts._policies import (  # noqa: E402
    oracle_heuristic_policy,
    populate_context_from_result,
)
from envs.reconcile_gst2b_env.scripts.train_grpo_real import (  # noqa: E402
    MAX_STEPS_PER_EPISODE,
    SYSTEM_PROMPT,
)
from envs.reconcile_gst2b_env.server.reconcile_gst2b_environment import (  # noqa: E402
    ReconcileGST2BEnvironment,
)


def _format_user_turn(obs: Any) -> str:
    """Build the user-side message given an observation.

    Mirrors the prompt format in train_grpo_real._eval_episode so SFT
    data distribution matches what the model sees at GRPO-time.
    """
    return (
        f"Step budget remaining: {obs.step_budget}\n"
        f"Invoices remaining: {obs.invoices_remaining_count}\n"
        f"Last tool result: {json.dumps(obs.last_tool_result or {})[:600]}\n"
        'Next action: emit a <tool_call> block with "name" and "arguments".'
    )


def _format_assistant_turn(action: Any) -> str:
    """Format an action as a Qwen3 <tool_call> block.

    This is what TRL's environment_factory GRPO emits during rollouts, so
    SFT examples in this exact format prepare the model for GRPO continuity.
    """
    return (
        f'<tool_call>{{"name": "{action.verb}", '
        f'"arguments": {json.dumps(action.payload)}}}</tool_call>'
    )


def generate_one_trajectory(seed: int) -> Dict[str, Any]:
    """Run the oracle on one seed; return a dict with messages + reward."""
    env = ReconcileGST2BEnvironment()
    obs = env.reset(seed=seed, mode="warmup")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    context: Dict[str, Any] = {"_env": env}
    rng = random.Random(seed)

    n_turns = 0
    while not obs.done and n_turns < MAX_STEPS_PER_EPISODE:
        messages.append({"role": "user", "content": _format_user_turn(obs)})

        action = oracle_heuristic_policy(obs, rng, context)
        messages.append(
            {"role": "assistant", "content": _format_assistant_turn(action)}
        )

        obs = env.step(action)
        populate_context_from_result(context, obs.last_tool_result or {})
        n_turns += 1

    breakdown = env.state.reward_breakdown or {
        "R1": 0.0,
        "R2": 0.0,
        "R3": 0.0,
        "R4": 0.0,
        "total": float(obs.reward) if obs.reward is not None else 0.0,
    }
    total = float(breakdown.get("total", 0.0))

    return {
        "seed": seed,
        "total": round(total, 4),
        "breakdown": {k: round(float(v), 4) for k, v in breakdown.items()},
        "n_turns": n_turns,
        "messages": messages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate SFT expert trajectories via the env's oracle policy."
    )
    parser.add_argument(
        "--start-seed", type=int, default=100, help="Inclusive start seed."
    )
    parser.add_argument(
        "--end-seed",
        type=int,
        default=600,
        help="Exclusive end seed (default 600 → 500 trajectories).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO / "data" / "sft_trajectories.jsonl",
        help="JSONL output path.",
    )
    parser.add_argument(
        "--min-total",
        type=float,
        default=0.50,
        help="Keep only trajectories with composite total ≥ this threshold.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate 5 seeds to verify format, then exit without writing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-seed breakdown during generation.",
    )
    args = parser.parse_args()

    if args.dry_run:
        seeds = list(range(args.start_seed, args.start_seed + 5))
        print(f"[dry-run] generating {len(seeds)} trajectories ...", flush=True)
    else:
        seeds = list(range(args.start_seed, args.end_seed))
        print(
            f"generating {len(seeds)} trajectories from seed {args.start_seed} "
            f"to {args.end_seed} (excl.) ...",
            flush=True,
        )

    kept: List[Dict[str, Any]] = []
    scores: List[float] = []
    turn_counts: List[int] = []
    t0 = time.time()

    for i, seed in enumerate(seeds):
        example = generate_one_trajectory(seed)
        scores.append(example["total"])
        turn_counts.append(example["n_turns"])

        if args.verbose or args.dry_run:
            print(
                f"  seed={seed:4d}  total={example['total']:.3f}  "
                f"turns={example['n_turns']:2d}  "
                f"R1={example['breakdown']['R1']:.2f} "
                f"R2={example['breakdown']['R2']:.2f} "
                f"R3={example['breakdown']['R3']:.2f} "
                f"R4={example['breakdown']['R4']:.2f}",
                flush=True,
            )

        if example["total"] >= args.min_total:
            kept.append(example)

        # Progress indicator every 50 seeds on full runs.
        if not args.verbose and not args.dry_run and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1.0)
            eta_s = (len(seeds) - (i + 1)) / max(rate, 1e-3)
            print(
                f"  [{i + 1:4d}/{len(seeds)}] "
                f"rate={rate:.1f} seeds/s  eta={eta_s:.0f}s  "
                f"kept={len(kept)}/{i + 1}",
                flush=True,
            )

    elapsed = time.time() - t0

    # Summary stats.
    print("\n=== generation complete ===", flush=True)
    print(f"  total seeds:      {len(seeds)}", flush=True)
    print(f"  kept (≥{args.min_total}):      {len(kept)}", flush=True)
    print(f"  wall time:        {elapsed:.1f}s", flush=True)
    if scores:
        print(
            f"  score min/mean/max: {min(scores):.3f} / "
            f"{statistics.mean(scores):.3f} / {max(scores):.3f}",
            flush=True,
        )
    if turn_counts:
        print(
            f"  turns min/mean/max: {min(turn_counts)} / "
            f"{statistics.mean(turn_counts):.1f} / {max(turn_counts)}",
            flush=True,
        )

    if args.dry_run:
        print("\n[dry-run] not writing file. First kept example preview:", flush=True)
        if kept:
            preview = dict(kept[0])
            # Truncate messages list to first 3 turns for preview.
            preview["messages"] = preview["messages"][:6]
            print(json.dumps(preview, indent=2)[:1500], flush=True)
        else:
            print(
                f"  (no trajectories ≥ {args.min_total}; oracle may need debugging)",
                flush=True,
            )
        return 0

    # Write JSONL.
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w") as f:
        for example in kept:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    size_mb = args.output_path.stat().st_size / (1024 * 1024)
    print(
        f"\nwrote {len(kept)} trajectories → {args.output_path} ({size_mb:.2f} MB)",
        flush=True,
    )
    print(
        f"ready for TRL SFTTrainer: "
        f"ds = load_dataset('json', data_files='{args.output_path}', split='train')",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
