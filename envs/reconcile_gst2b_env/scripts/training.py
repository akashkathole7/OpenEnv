#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GRPO + LoRA training scaffold for reconcile_gst2b_env.

Colab-runnable. Not exercised by local CI (requires GPU + model weights).

Tier map:
  t4    → Qwen2.5-3B-Instruct, 250 steps  (auto-downgrade on OOM to 1.5B + group 4)
  a10   → Qwen2.5-3B-Instruct, 400 steps
  a100  → Qwen2.5-7B-Instruct, 600 steps

Training loop:
- Prompt = initial observation + tool schema.
- Sample ``group_size`` completions per prompt; each completion is a JSON
  array of tool calls parsed into ``ReconcileAction``s.
- Run each trajectory through a fresh env instance; composite_reward["total"]
  is the GRPO reward.
- Eval every ``--eval-every`` steps on seeds 9030–9049 (disjoint from the
  Section E baseline heldout 9000–9029).
- Per-component means logged to ``<output-dir>/training_curves.json``.
- Checkpoint every 100 steps via ``save_pretrained``.

Reward parsing tolerance: malformed JSON → empty trajectory → composite = 0.01
(clamped). No structural -1.0 here; training must tolerate policy exploration.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from envs.reconcile_gst2b_env.models import ReconcileAction  # noqa: E402
from envs.reconcile_gst2b_env.rewards import composite_reward  # noqa: E402
from envs.reconcile_gst2b_env.server.reconcile_gst2b_environment import (  # noqa: E402
    ReconcileGST2BEnvironment,
)


EVAL_SEEDS = list(range(9030, 9050))  # 20 seeds, disjoint from baseline 9000–9029


TIER_CONFIG: Dict[str, Dict[str, Any]] = {
    "t4": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "steps": 250,
        "group_size": 8,
        "fallback_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "fallback_group_size": 4,
    },
    "a10": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "steps": 400,
        "group_size": 8,
        "fallback_model": None,
        "fallback_group_size": None,
    },
    "a100": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "steps": 600,
        "group_size": 8,
        "fallback_model": None,
        "fallback_group_size": None,
    },
}

LORA_RANK = 32
LEARNING_RATE = 5e-6
CONTEXT_LEN = 8192
CHECKPOINT_EVERY = 100
TRAIN_SEEDS_RANGE = (0, 500)


# ---------- trajectory <-> text parsing ----------

_TRAJ_PATTERN = re.compile(r"\[.*\]", re.DOTALL)


def parse_trajectory_text(text: str) -> List[ReconcileAction]:
    """Parse a JSON array of ``{verb, payload}`` dicts from LM output.

    Tolerates surrounding prose; extracts the first JSON array. Invalid
    entries are skipped. Returns ``[]`` on total failure.
    """
    if not text:
        return []
    match = _TRAJ_PATTERN.search(text)
    if match is None:
        return []
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    actions: List[ReconcileAction] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        verb = entry.get("verb")
        payload = entry.get("payload", {}) or {}
        if not isinstance(verb, str) or not isinstance(payload, dict):
            continue
        try:
            actions.append(ReconcileAction(verb=verb, payload=payload))
        except Exception:
            continue
    return actions


def score_trajectory(seed: int, text: str) -> Dict[str, float]:
    """Parse text, execute against a fresh env, return breakdown dict."""
    env = ReconcileGST2BEnvironment()
    env.reset(seed=seed)
    actions = parse_trajectory_text(text)
    for a in actions:
        obs = env.step(a)
        if obs.done:
            break
    # If trajectory didn't terminate the env, compute composite on what we have.
    if not env.state.reward_breakdown:
        breakdown = composite_reward(env.state, env._trajectory)
        env.state.reward_breakdown = breakdown
    return env.state.reward_breakdown


# ---------- eval ----------


def evaluate_checkpoint(policy_fn) -> Dict[str, float]:
    """Run ``policy_fn`` on each EVAL_SEEDS episode, return per-component means.

    ``policy_fn(seed) -> List[ReconcileAction]`` — stateless, produces a full
    trajectory for one seed. Used for periodic eval during training.
    """
    totals: List[Dict[str, float]] = []
    for seed in EVAL_SEEDS:
        env = ReconcileGST2BEnvironment()
        env.reset(seed=seed)
        actions = policy_fn(seed)
        for a in actions:
            obs = env.step(a)
            if obs.done:
                break
        if not env.state.reward_breakdown:
            env.state.reward_breakdown = composite_reward(env.state, env._trajectory)
        totals.append(env.state.reward_breakdown)
    mean = {}
    for key in ("R1", "R2", "R3", "R4", "total"):
        vals = [t.get(key, 0.0) for t in totals]
        mean[key] = sum(vals) / max(1, len(vals))
    return mean


# ---------- stub eval policy (replaced by real LM at Colab runtime) ----------


def _stub_eval_policy_factory(model, tokenizer):
    """Build a policy_fn that samples tool-call JSON from the LM.

    ``model`` + ``tokenizer`` are HuggingFace artifacts. The factory produces
    a function ``(seed) -> List[ReconcileAction]`` that prompts the model and
    parses the output.
    """
    sys_prompt = _build_system_prompt()

    def policy_fn(seed: int) -> List[ReconcileAction]:
        env = ReconcileGST2BEnvironment()
        obs = env.reset(seed=seed)
        user = (
            f"User request: {obs.user_request}\n"
            f"Step budget: {obs.step_budget}\n"
            "Return ONLY a JSON array of tool calls, no prose.\n"
            'Example: [{"verb":"get_schema","payload":{}}, '
            '{"verb":"submit","payload":{}}]'
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        gen = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return parse_trajectory_text(gen)

    return policy_fn


def _build_system_prompt() -> str:
    return (
        "You are a GST reconciliation agent. You have 16 verbs: 7 query "
        "(get_schema, list_gstins, fuzzy_search_gstin, get_invoice, "
        "list_invoices_by_supplier, get_2b_row, get_hsn_slab), 7 mutate "
        "(mark_matched, mark_mismatched, mark_only_in_books, mark_only_in_2b, "
        "mark_partial_match, flag_circular_ring, request_amendment), and 2 "
        "meta (confirm_with_user, submit). You MUST call at least one query "
        "verb before submit or receive -1.0. Respond with a JSON array of "
        "tool calls only."
    )


# ---------- main training loop ----------


def train_with_tier(
    tier: str,
    steps_override: Optional[int],
    output_dir: Path,
    use_unsloth: bool,
    resume: Optional[str],
    eval_every: int,
) -> int:
    """Train with the given tier config. OOM → auto-downgrade for T4 only."""
    try:
        return _train_inner(
            tier,
            steps_override,
            output_dir,
            use_unsloth,
            resume,
            eval_every,
            use_fallback=False,
        )
    except RuntimeError as exc:
        msg = str(exc).lower()
        if tier != "t4" or "out of memory" not in msg:
            raise
        print(
            f"\n[OOM] {exc}\n[OOM] Auto-downgrading: "
            f"{TIER_CONFIG['t4']['fallback_model']}, "
            f"group_size={TIER_CONFIG['t4']['fallback_group_size']}\n",
            flush=True,
        )
        return _train_inner(
            tier,
            steps_override,
            output_dir,
            use_unsloth,
            resume,
            eval_every,
            use_fallback=True,
        )


def _train_inner(
    tier: str,
    steps_override: Optional[int],
    output_dir: Path,
    use_unsloth: bool,
    resume: Optional[str],
    eval_every: int,
    use_fallback: bool,
) -> int:
    cfg = TIER_CONFIG[tier]
    model_name = cfg["fallback_model"] if use_fallback else cfg["model"]
    group_size = cfg["fallback_group_size"] if use_fallback else cfg["group_size"]
    total_steps = steps_override if steps_override is not None else cfg["steps"]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    # Load model + tokenizer. Unsloth when available, else transformers.
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel  # type: ignore

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=CONTEXT_LEN,
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model, r=LORA_RANK, lora_alpha=LORA_RANK, lora_dropout=0.0
            )
        except ImportError:
            print(
                "[warn] Unsloth unavailable; falling back to transformers.", flush=True
            )
            use_unsloth = False

    if not use_unsloth:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    if resume:
        from peft import PeftModel  # type: ignore

        model = PeftModel.from_pretrained(model, resume)

    curves: Dict[str, List[Any]] = {
        "steps": [],
        "R1": [],
        "R2": [],
        "R3": [],
        "R4": [],
        "total": [],
        "config": {
            "tier": tier,
            "model": model_name,
            "group_size": group_size,
            "lora_rank": LORA_RANK,
            "lr": LEARNING_RATE,
            "context_len": CONTEXT_LEN,
            "total_steps": total_steps,
            "eval_every": eval_every,
            "eval_seeds": EVAL_SEEDS,
            "fallback": use_fallback,
        },
    }

    def _record_eval(step: int) -> None:
        policy = _stub_eval_policy_factory(model, tokenizer)
        means = evaluate_checkpoint(policy)
        curves["steps"].append(step)
        for k in ("R1", "R2", "R3", "R4", "total"):
            curves[k].append(round(means[k], 4))
        (output_dir / "training_curves.json").write_text(json.dumps(curves, indent=2))
        print(
            f"[eval @ step {step:4d}] "
            f"R1={means['R1']:.3f} R2={means['R2']:.3f} "
            f"R3={means['R3']:.3f} R4={means['R4']:.3f} total={means['total']:.3f}",
            flush=True,
        )

    # Pre-training eval (step 0)
    print(f"\n=== training: tier={tier} model={model_name} steps={total_steps} ===")
    _record_eval(0)

    # GRPO training loop (real loop plugged in at Colab time via trl.GRPOTrainer).
    # For the dryrun scaffold we step through N iterations, eval periodically,
    # and save checkpoints. Gradient-carrying code lives on the Colab path.
    t0 = time.time()
    for step in range(1, total_steps + 1):
        # Real loop: sample group_size completions × len(train_batch) prompts,
        # parse trajectories, score via composite_reward, compute GRPO
        # advantages, backprop. Implemented in Colab-side Unsloth code.
        # Dryrun path: no-op step so scaffold validates end-to-end.
        if step % eval_every == 0:
            _record_eval(step)
        if step % CHECKPOINT_EVERY == 0 or step == total_steps:
            ckpt_dir = output_dir / "checkpoints" / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            print(f"[ckpt] saved {ckpt_dir}", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== training complete in {elapsed:.1f}s ===")
    print(f"curves: {output_dir / 'training_curves.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GRPO + LoRA training for reconcile_gst2b_env."
    )
    parser.add_argument("--tier", choices=sorted(TIER_CONFIG.keys()), required=True)
    parser.add_argument("--steps", type=int, default=None, help="Override step count")
    parser.add_argument("--output-dir", type=Path, default=Path("data/training"))
    parser.add_argument(
        "--use-unsloth",
        type=lambda v: str(v).lower() in ("true", "1", "yes"),
        default=True,
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Steps between evals (default 50; 5 recommended for dryruns).",
    )
    args = parser.parse_args()

    return train_with_tier(
        tier=args.tier,
        steps_override=args.steps,
        output_dir=args.output_dir,
        use_unsloth=args.use_unsloth,
        resume=args.resume,
        eval_every=args.eval_every,
    )


if __name__ == "__main__":
    sys.exit(main())
