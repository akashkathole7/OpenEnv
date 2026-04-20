#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inference-only real baseline: Qwen2.5-3B-Instruct on heldout seeds.

Produces ``data/baseline_metrics_real.json`` with raw + prompted + delta.
Colab T4 budget ~60 min for 30 seeds × 3 samples × 2 conditions = 180 rollouts.
Fallback: ``--n-seeds 20 --n-samples 2`` = 80 rollouts.

Hard stop: if prompted − raw delta comes back <0.05, the script still writes
the JSON (for honest reporting) and exits with code 1 so the notebook surfaces
the failure. The user decides whether to pivot the narrative or shelve.

Action parser: the same ``parse_trajectory_text`` used in
``scripts/training.py``. The model is prompted to emit a JSON array with
exactly one element per turn (the current action); ``parse_trajectory_text``
returns a list and we take the first entry.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from envs.reconcile_gst2b_env.models import ReconcileAction  # noqa: E402
from envs.reconcile_gst2b_env.rewards import bootstrap_ci, composite_reward  # noqa: E402
from envs.reconcile_gst2b_env.scripts.training import (  # noqa: E402
    parse_trajectory_text,
)
from envs.reconcile_gst2b_env.server.reconcile_gst2b_environment import (  # noqa: E402
    ReconcileGST2BEnvironment,
)


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_SEEDS = list(
    range(9000, 9030)
)  # 30 heldout, disjoint from training (9030-9049) and hero (9500-9502)

RAW_SYSTEM = "You are an agent."

PROMPTED_SYSTEM = (
    "You are a GST reconciliation agent. The env has 16 verbs "
    "(7 query: get_schema, list_gstins, fuzzy_search_gstin, get_invoice, "
    "list_invoices_by_supplier, get_2b_row, get_hsn_slab; "
    "7 mutate: mark_matched, mark_mismatched, mark_only_in_books, "
    "mark_only_in_2b, mark_partial_match, flag_circular_ring, request_amendment; "
    "2 meta: confirm_with_user, submit) and 5 labels (matched, mismatched, "
    "only_in_books, only_in_2b, partial). Reward components: macro-F1 on labels, "
    "ITC delta accuracy, Rule 36(4) per-supplier cap (requires ≥1 query action "
    "before submit), step efficiency. Submitting without any prior query "
    "returns -1.0. Respond with exactly one action as a strict JSON array: "
    '[{"verb": "...", "payload": {...}}].'
)


def _truncate(obj: Any, max_chars: int = 800) -> str:
    s = json.dumps(obj, default=str)
    if len(s) > max_chars:
        s = s[:max_chars] + "...[truncated]"
    return s


def _build_turn_prompt(obs: Any, condition: str, tokenizer: Any) -> str:
    system = PROMPTED_SYSTEM if condition == "prompted" else RAW_SYSTEM
    user = (
        f"User request: {obs.user_request}\n"
        f"Step budget remaining: {obs.step_budget}\n"
        f"Invoices remaining: {obs.invoices_remaining_count}\n"
        f"Last tool result: {_truncate(obs.last_tool_result)}\n"
        'Next action (single-element JSON array): [{"verb": "...", "payload": {...}}]'
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _get_next_action(
    obs: Any, condition: str, model: Any, tokenizer: Any, max_new_tokens: int
) -> ReconcileAction:
    prompt = _build_turn_prompt(obs, condition, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    actions = parse_trajectory_text(gen)
    if actions:
        return actions[0]
    # Parser miss → submit. If no prior query this triggers the structural -1.0,
    # which is the correct honest signal for a raw LM that fails to emit JSON.
    return ReconcileAction(verb="submit", payload={})


def _run_episode(
    seed: int,
    sample_idx: int,
    condition: str,
    model: Any,
    tokenizer: Any,
    max_steps: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    env = ReconcileGST2BEnvironment()
    obs = env.reset(seed=seed)
    step_count = 0
    while not obs.done and step_count < max_steps:
        action = _get_next_action(obs, condition, model, tokenizer, max_new_tokens)
        obs = env.step(action)
        step_count += 1

    if not env.state.reward_breakdown:
        env.state.reward_breakdown = composite_reward(env.state, env._trajectory)

    breakdown = env.state.reward_breakdown
    term = (obs.metadata or {}).get("termination_reason", "unknown")
    catastrophic = bool(
        term == "submit_before_query" or env.state.true_rule_36_4_violated
    )
    return {
        "seed": seed,
        "sample_idx": sample_idx,
        "condition": condition,
        "component_rewards": {
            k: round(float(v), 4) for k, v in breakdown.items() if k != "total"
        },
        "total": round(float(breakdown.get("total", 0.0)), 4),
        "total_steps": step_count,
        "termination_reason": term,
        "done": bool(obs.done),
        "catastrophic_corruption": catastrophic,
    }


def _summarize(condition: str, rollouts: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = [r["total"] for r in rollouts]
    lo, hi = bootstrap_ci(totals, n=1000, alpha=0.05, seed=0)
    per_c: Dict[str, Any] = {}
    for c in ("R1", "R2", "R3", "R4"):
        vals = [r["component_rewards"][c] for r in rollouts]
        clo, chi = bootstrap_ci(vals, n=500, alpha=0.05, seed=0)
        per_c[c] = {
            "mean": round(statistics.mean(vals), 4),
            "ci95_low": round(clo, 4),
            "ci95_high": round(chi, 4),
        }
    cat_rate = sum(1 for r in rollouts if r["catastrophic_corruption"]) / max(
        1, len(rollouts)
    )
    return {
        "condition": condition,
        "execution_mode": "real",
        "model": MODEL_NAME,
        "n_rollouts": len(rollouts),
        "total_mean": round(statistics.mean(totals), 4),
        "total_ci95": [round(lo, 4), round(hi, 4)],
        "total_stdev": round(statistics.stdev(totals) if len(totals) > 1 else 0.0, 4),
        "catastrophic_corruption_rate": round(cat_rate, 4),
        "component_summary": per_c,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Real Qwen2.5-3B baseline eval.")
    parser.add_argument(
        "--condition", choices=["raw", "prompted", "both"], default="both"
    )
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=REPO / "data" / "baseline_metrics_real.json"
    )
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    seeds = DEFAULT_SEEDS[: args.n_seeds]

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    print(f"loading {MODEL_NAME} ...", flush=True)
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto"
    )
    model.eval()
    print(f"loaded in {time.time() - t_load:.1f}s on device={model.device}", flush=True)

    conditions = ["raw", "prompted"] if args.condition == "both" else [args.condition]
    all_results: Dict[str, Any] = {}

    for cond in conditions:
        print(f"\n=== condition: {cond} ===", flush=True)
        rollouts: List[Dict[str, Any]] = []
        c_t0 = time.time()
        total = len(seeds) * args.n_samples
        for seed in seeds:
            for k in range(args.n_samples):
                log = _run_episode(
                    seed=seed,
                    sample_idx=k,
                    condition=cond,
                    model=model,
                    tokenizer=tokenizer,
                    max_steps=args.max_steps,
                    max_new_tokens=args.max_new_tokens,
                )
                rollouts.append(log)
                done_n = len(rollouts)
                elapsed = time.time() - c_t0
                print(
                    f"  [{cond}] {done_n:3d}/{total} seed={seed} k={k} "
                    f"total={log['total']:.3f} steps={log['total_steps']} "
                    f"term={log['termination_reason']} elapsed={elapsed:.0f}s",
                    flush=True,
                )
        all_results[cond] = {
            "summary": _summarize(cond, rollouts),
            "rollouts": rollouts,
        }

    # Delta + bootstrap CI over paired differences.
    if "raw" in all_results and "prompted" in all_results:
        r_totals = [r["total"] for r in all_results["raw"]["rollouts"]]
        p_totals = [r["total"] for r in all_results["prompted"]["rollouts"]]
        delta_mean = round(
            all_results["prompted"]["summary"]["total_mean"]
            - all_results["raw"]["summary"]["total_mean"],
            4,
        )
        if len(r_totals) == len(p_totals):
            diffs = [p - r for p, r in zip(p_totals, r_totals)]
            d_lo, d_hi = bootstrap_ci(diffs, n=1000, alpha=0.05, seed=0)
            delta_ci = [round(d_lo, 4), round(d_hi, 4)]
        else:
            delta_ci = [None, None]
        all_results["delta_prompted_minus_raw"] = {
            "mean": delta_mean,
            "ci95": delta_ci,
            "passes_gate_05": bool(delta_mean >= 0.05),
        }
        print(
            f"\n=== delta prompted − raw = {delta_mean:.4f} (CI95 {delta_ci}) ===",
            flush=True,
        )
        if delta_mean < 0.05:
            print(
                "HARD-STOP: delta < 0.05 — storytelling arc at risk. "
                "See NOTES.md for the pivot/shelve decision tree.",
                flush=True,
            )

    all_results["config"] = {
        "n_seeds": len(seeds),
        "n_samples": args.n_samples,
        "seeds": seeds,
        "max_steps": args.max_steps,
        "model": MODEL_NAME,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {args.output}", flush=True)

    if all_results.get("delta_prompted_minus_raw", {}).get("mean", 0) < 0.05:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
