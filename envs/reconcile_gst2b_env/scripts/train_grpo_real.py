#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real GRPO training for reconcile_gst2b_env via TRL's environment_factory.

Kaggle T4 budget: target 2–3 h for 150 steps on Qwen/Qwen3-0.6B with
LoRA rank 16.

Model swap: Qwen2.5-1.5B → Qwen3-0.6B because TRL main's
``add_response_schema`` only supports the Qwen3 family (see TRL issue
#5460). GRPOTrainer re-runs the schema inference internally, so a manual
``tokenizer.response_schema`` fallback gets clobbered — only Qwen3 works
end-to-end. The size reduction is acceptable: for a demo run, training
signal visibility matters more than absolute model scale.

Pattern: TRL's ``environment_factory`` receives a class (not an instance).
TRL creates one instance per generation, auto-discovers public methods as
tools, and drives the multi-turn function-calling loop. Our class wraps
a local ``ReconcileGST2BEnvironment`` (no HTTP) and exposes the 16 verbs
as typed tool methods with docstrings. Reward is stored on ``self.reward``
and read by the reward_func — we do NOT recompute reward in the trainer.

Explicit failure-mode defenses:
- OOM on model load → caught, stack trace printed with guidance to lower
  num_generations or max_completion_length (no smaller Qwen3 exists).
- TRL < 0.21 → hard halt with install guidance.
- Env import failure → top-level import, fail fast.
- Flat curves after 150 steps → still shipped as an honest finding (env
  may be too hard for 0.6B at this scale).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Fail fast on env package import.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from envs.reconcile_gst2b_env.models import ReconcileAction  # noqa: E402
from envs.reconcile_gst2b_env.rewards import composite_reward  # noqa: E402
from envs.reconcile_gst2b_env.scripts.training import parse_trajectory_text  # noqa: E402
from envs.reconcile_gst2b_env.server.reconcile_gst2b_environment import (  # noqa: E402
    ReconcileGST2BEnvironment,
)


MODEL_NAME = "Qwen/Qwen3-0.6B"
LORA_RANK = 16
# q/v only covers ~half of attention's learnable surface. Adding k/o lets
# the adapter shift key projections (affects what the model attends to)
# and output projections (affects how attended info flows forward),
# roughly doubling the representational surface at small VRAM cost.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
# 5e-6 × 150 steps barely perturbs a rank-16 LoRA on Qwen3-0.6B; 1e-5 is a
# conservative 2× bump paired with the early-stop safety net (if totals
# collapse > 0.10 below step-0, training halts with exit code 2).
LEARNING_RATE = 1e-5
TOTAL_STEPS = 150
EVAL_EVERY = 25
CHECKPOINT_EVERY = 50
NUM_GENERATIONS = 4
MAX_COMPLETION_LENGTH = 512
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 4

TRAIN_SEEDS = list(range(0, 150))  # 150 prompt-slots, one per training step
EVAL_SEEDS = list(
    range(9030, 9040)
)  # 10 heldout, disjoint from baseline (9000–9029) and hero (9500–9502)
MAX_STEPS_PER_EPISODE = 50

# System prompt — same shape as real_baseline.py PROMPTED_SYSTEM (~106 tokens),
# rewritten for function-calling (tool methods) rather than strict JSON array.
SYSTEM_PROMPT = (
    "You are a GST reconciliation agent. Each episode is one company's "
    "monthly purchase register vs the GSTR-2B return. Call the available "
    "tool methods to inspect invoices, query the schema, and mark each "
    "invoice with one of five labels (matched, mismatched, only_in_books, "
    "only_in_2b, partial). You MUST call at least one query tool before "
    "`submit` — submitting without a prior query returns −1.0. Reward is "
    "weighted macro-F1 on labels + ITC delta accuracy + per-supplier "
    "Rule 36(4) compliance + step efficiency. Call `submit` when done."
)


# ---------- TRL tool-method class (environment_factory target) ----------


class ReconcileToolEnv:
    """TRL environment_factory target. One instance per generation.

    Wraps a local ``ReconcileGST2BEnvironment`` — no HTTP. Each public
    method (not starting with ``_``) is auto-exposed by TRL as a tool
    with its typed args and docstring.

    ``self.reward`` holds the env's running composite total. The
    reward_func reads this at the end of each episode.
    """

    def __init__(self) -> None:
        self._env = ReconcileGST2BEnvironment()
        self.reward: float = 0.0
        self._done: bool = False

    # ---- lifecycle ----

    def reset(self, **kwargs: Any) -> Optional[str]:
        """Start a new episode. ``kwargs`` carries dataset columns — we
        read ``seed`` if present, else default to 0."""
        seed = int(kwargs.get("seed", 0))
        obs = self._env.reset(seed=seed)
        self.reward = 0.0
        self._done = False
        return (
            f"New episode. {obs.invoices_remaining_count} invoices to reconcile. "
            f"Step budget {obs.step_budget}. Call tools to inspect and label, then `submit`."
        )

    # ---- helpers (private, not tools) ----

    def _step_verb(self, verb: str, payload: Dict[str, Any]) -> str:
        if self._done:
            raise ValueError("Episode has ended — no further tool calls accepted.")
        action = ReconcileAction(verb=verb, payload=payload)
        obs = self._env.step(action)
        breakdown = self._env.state.reward_breakdown or {}
        self.reward = float(breakdown.get("total", 0.0))
        if obs.done:
            self._done = True
        return json.dumps(obs.last_tool_result or {})

    def _finalize_reward(self) -> None:
        """If an episode ends without the env populating reward_breakdown,
        backfill via composite_reward."""
        if not self._env.state.reward_breakdown:
            self._env.state.reward_breakdown = composite_reward(
                self._env.state, self._env._trajectory
            )
        self.reward = float(self._env.state.reward_breakdown.get("total", 0.0))

    # ---- query tools (7) ----

    def get_schema(self) -> str:
        """Return the env schema: verbs, labels, max_steps, company_gstin, invoice counts.

        Returns:
            JSON string with keys ``schema`` (dict) or ``error`` on miss.
        """
        return self._step_verb("get_schema", {})

    def list_gstins(self) -> str:
        """List all supplier GSTINs appearing in the purchase_register or gstr_2b.

        Returns:
            JSON string with key ``gstins`` (sorted list of GSTIN strings).
        """
        return self._step_verb("list_gstins", {})

    def fuzzy_search_gstin(self, query: str) -> str:
        """Fuzzy match a GSTIN substring against known suppliers.

        Args:
            query: Partial GSTIN or pattern to match.

        Returns:
            JSON string with key ``matches`` (up to 5 closest GSTINs).
        """
        return self._step_verb("fuzzy_search_gstin", {"query": query})

    def get_invoice(self, invoice_id: str) -> str:
        """Fetch a single invoice from purchase_register (preferred) or gstr_2b.

        Args:
            invoice_id: e.g. ``inv_0005`` or ``inv_2bonly_0001``.

        Returns:
            JSON string with keys ``source`` and ``invoice`` on hit, ``error`` on miss.
        """
        return self._step_verb("get_invoice", {"invoice_id": invoice_id})

    def list_invoices_by_supplier(self, gstin: str) -> str:
        """List all invoices for a specific supplier across books and 2B.

        Args:
            gstin: 15-char GSTIN.

        Returns:
            JSON string with keys ``purchase_register`` and ``gstr_2b`` (lists).
        """
        return self._step_verb("list_invoices_by_supplier", {"gstin": gstin})

    def get_2b_row(self, invoice_id: str) -> str:
        """Fetch an invoice's GSTR-2B-side record.

        Args:
            invoice_id: e.g. ``inv_0005``.

        Returns:
            JSON string with key ``invoice`` on hit, ``error`` if not in 2B.
        """
        return self._step_verb("get_2b_row", {"invoice_id": invoice_id})

    def get_hsn_slab(self, hsn: str) -> str:
        """Look up the GST slab (percent) for an HSN code.

        Args:
            hsn: 4-character HSN prefix, e.g. ``8517``.

        Returns:
            JSON string with keys ``hsn`` and ``slab_pct`` on hit, ``error`` on miss.
        """
        return self._step_verb("get_hsn_slab", {"hsn": hsn})

    # ---- mutate tools (7) ----

    def mark_matched(self, invoice_id: str) -> str:
        """Label an invoice as ``matched`` (books and 2B agree).

        Args:
            invoice_id: Target invoice id.

        Returns:
            Acknowledgement JSON.
        """
        return self._step_verb("mark_matched", {"invoice_id": invoice_id})

    def mark_mismatched(self, invoice_id: str, reason: str) -> str:
        """Label an invoice as ``mismatched`` with a reason code.

        Args:
            invoice_id: Target invoice id.
            reason: One of ``value`` | ``tax`` | ``date`` | ``gstin`` | ``hsn`` | ``invoice_no``.

        Returns:
            Acknowledgement JSON.
        """
        return self._step_verb(
            "mark_mismatched", {"invoice_id": invoice_id, "reason": reason}
        )

    def mark_only_in_books(self, invoice_id: str) -> str:
        """Label an invoice as present only in the purchase_register (not in 2B).

        Args:
            invoice_id: Target invoice id.

        Returns:
            Acknowledgement JSON.
        """
        return self._step_verb("mark_only_in_books", {"invoice_id": invoice_id})

    def mark_only_in_2b(self, invoice_id: str) -> str:
        """Label an invoice as present only in GSTR-2B (not in the books).

        Args:
            invoice_id: Target invoice id.

        Returns:
            Acknowledgement JSON.
        """
        return self._step_verb("mark_only_in_2b", {"invoice_id": invoice_id})

    def mark_partial_match(self, invoice_id: str, delta_inr: float) -> str:
        """Label an invoice as a partial match; claim books tax minus delta.

        Args:
            invoice_id: Target invoice id.
            delta_inr: Amount (INR) by which books overstates the 2B tax. 0 for no delta.

        Returns:
            Acknowledgement JSON.
        """
        return self._step_verb(
            "mark_partial_match",
            {"invoice_id": invoice_id, "delta_inr": float(delta_inr)},
        )

    def flag_circular_ring(self, gstins: List[str]) -> str:
        """Flag a list of supplier GSTINs as a suspected circular-trading ring.

        Args:
            gstins: List of GSTINs suspected to form a cycle (typically 3).

        Returns:
            Acknowledgement JSON listing the flagged GSTINs.
        """
        return self._step_verb("flag_circular_ring", {"gstins": list(gstins)})

    def request_amendment(self, invoice_id: str, field: str, value: str) -> str:
        """Request an amendment to a specific field on an invoice.

        Args:
            invoice_id: Target invoice id.
            field: Field name, e.g. ``value_inr`` or ``gstin``.
            value: New value (stringified; numeric fields are parsed downstream).

        Returns:
            Acknowledgement JSON.
        """
        return self._step_verb(
            "request_amendment",
            {"invoice_id": invoice_id, "field": field, "value": value},
        )

    # ---- meta tools (2) ----

    def confirm_with_user(self, question: str) -> str:
        """Ask a clarifying question (no state effect; useful for reasoning traces).

        Args:
            question: The clarifying question.

        Returns:
            Acknowledgement JSON echoing the question.
        """
        return self._step_verb("confirm_with_user", {"question": question})

    def submit(self) -> str:
        """Signal the episode is complete. Raises to end the tool-calling loop.

        Returns:
            (Never returns normally — raises ValueError with final reward.)
        """
        result = self._step_verb("submit", {})
        self._finalize_reward()
        raise ValueError(f"Episode complete. Final reward={self.reward:.3f}. {result}")


# ---------- reward function (read-only; no reward recomputation) ----------


def reward_func(environments: List[Any], **kwargs: Any) -> List[float]:
    """Per TRL contract: read ``env.reward`` from each environment instance.

    CRITICAL: we force ``_finalize_reward()`` first. If the model stops
    calling tools without invoking ``submit`` (common — it hits
    ``max_completion_length`` or decides it's done mid-episode), the
    underlying env never populates ``reward_breakdown`` and ``self.reward``
    stays at 0. That caused ``frac_reward_zero_std=1`` on the first real
    Kaggle run. Finalizing here backfills composite_reward over whatever
    trajectory was collected, so every rollout produces a real score.
    Idempotent when submit was called.
    """
    rewards: List[float] = []
    for env in environments:
        try:
            env._finalize_reward()
        except Exception:
            # Never let reward collection crash a training step.
            pass
        rewards.append(float(getattr(env, "reward", 0.0)))
    return rewards


# ---------- eval callback ----------


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_SINGLE_OBJECT_RE = re.compile(
    r'\{[^{}]*"verb"\s*:\s*"[^"]+"(?:[^{}]|\{[^{}]*\})*\}', re.DOTALL
)


def _parse_action_robust(gen: str) -> Optional[ReconcileAction]:
    """Accept any of the three formats the trained LM might emit.

    After GRPO training with TRL's function-calling schema, the model may
    shift away from the JSON-array prompt format toward Qwen3's native
    ``<tool_call>`` tags. Eval must not penalize that shift by treating
    it as a parse failure.
    """
    # 1. JSON-array: [{"verb": "...", "payload": {...}}, ...]
    actions = parse_trajectory_text(gen)
    if actions:
        return actions[0]
    # 2. Qwen3 <tool_call> tags: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    m = _TOOL_CALL_RE.search(gen)
    if m:
        try:
            raw = json.loads(m.group(1))
            name = raw.get("name")
            args_ = raw.get("arguments", {}) or {}
            if isinstance(name, str) and isinstance(args_, dict):
                return ReconcileAction(verb=name, payload=args_)
        except Exception:
            pass
    # 3. Bare single JSON object via regex: {"verb": "...", "payload": {...}}
    m = _SINGLE_OBJECT_RE.search(gen)
    if m:
        try:
            raw = json.loads(m.group(0))
            if isinstance(raw, dict) and isinstance(raw.get("verb"), str):
                return ReconcileAction(
                    verb=raw["verb"], payload=raw.get("payload", {}) or {}
                )
        except Exception:
            pass
    # 4. Balanced-brace scan — strictly looser than fallback 3. Walks every
    # ``{`` in the generation, tracks brace depth, and JSON-parses each
    # balanced object whose literal text contains ``"verb"``. Catches cases
    # where the regex fails (nested objects with quoted braces, payloads
    # longer than one level, escape sequences), at worst-case O(n²) on
    # short generations — fine for ≤256-token eval outputs.
    for i, ch in enumerate(gen):
        if ch != "{":
            continue
        depth = 0
        for j in range(i, len(gen)):
            if gen[j] == "{":
                depth += 1
            elif gen[j] == "}":
                depth -= 1
                if depth == 0:
                    candidate = gen[i : j + 1]
                    if '"verb"' in candidate:
                        try:
                            raw = json.loads(candidate)
                            if isinstance(raw, dict) and isinstance(
                                raw.get("verb"), str
                            ):
                                return ReconcileAction(
                                    verb=raw["verb"],
                                    payload=raw.get("payload", {}) or {},
                                )
                        except Exception:
                            pass
                    break
    return None


def _eval_episode(
    seed: int, model: Any, tokenizer: Any, max_steps: int = MAX_STEPS_PER_EPISODE
) -> Dict[str, float]:
    """One eval episode. Forces ``get_schema`` as turn 1 so the structural
    −1.0 submit-before-query penalty can never dominate regardless of what
    the LM emits. Subsequent turns are LM-driven with multi-format parsing
    (JSON-array, ``<tool_call>``, bare object). Parse failures break the
    loop rather than falling back to ``submit``.
    """
    import torch

    env = ReconcileGST2BEnvironment()
    env.reset(seed=seed)
    # Forced turn 1: guarantees R3 is not clamped to 0.01 and defuses the
    # submit-before-query structural floor.
    obs = env.step(ReconcileAction(verb="get_schema", payload={}))
    step_count = 1
    while not obs.done and step_count < max_steps:
        user_msg = (
            f"Step budget remaining: {obs.step_budget}\n"
            f"Invoices remaining: {obs.invoices_remaining_count}\n"
            f"Last tool result: {json.dumps(obs.last_tool_result or {})[:600]}\n"
            'Next action (single-element JSON array): [{"verb": "...", "payload": {...}}]'
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        action = _parse_action_robust(gen)
        if action is None:
            # Don't submit-fallback — that would cost an episode that was
            # merely mid-format-drift. End the loop and score the partial
            # trajectory.
            break
        obs = env.step(action)
        step_count += 1

    if not env.state.reward_breakdown:
        env.state.reward_breakdown = composite_reward(env.state, env._trajectory)
    breakdown = env.state.reward_breakdown
    return {
        "R1": float(breakdown.get("R1", 0.0)),
        "R2": float(breakdown.get("R2", 0.0)),
        "R3": float(breakdown.get("R3", 0.0)),
        "R4": float(breakdown.get("R4", 0.0)),
        "total": float(breakdown.get("total", 0.0)),
    }


def _build_eval_callback(
    tokenizer: Any, output_dir: Path, curves: Dict[str, List[Any]]
):
    from transformers import TrainerCallback

    class ComponentEvalCallback(TrainerCallback):
        def __init__(self) -> None:
            self._did_step_zero = False

        def _run_and_log(self, step: int, model: Any) -> None:
            # Unwrap DDP if present (single-GPU T4 never wraps, but cheap
            # insurance for multi-GPU reruns on L4/A100).
            eval_model = model.module if hasattr(model, "module") else model
            # Adapter sanity check — if the eval curves are flat while
            # training reward is non-zero, the first thing to rule out is
            # "eval is hitting the base model without the LoRA adapter
            # applied". This print makes that visible in Kaggle logs.
            try:
                from peft import PeftModel  # type: ignore

                is_peft = isinstance(eval_model, PeftModel)
            except Exception:
                is_peft = False
            active = getattr(eval_model, "active_adapter", None)
            print(
                f"[eval @ step {step:4d}] adapter_check: "
                f"PeftModel={is_peft} active_adapter={active}",
                flush=True,
            )
            rows = [_eval_episode(seed, eval_model, tokenizer) for seed in EVAL_SEEDS]
            means = {
                k: statistics.mean(r[k] for r in rows)
                for k in ("R1", "R2", "R3", "R4", "total")
            }
            curves["steps"].append(step)
            for k in ("R1", "R2", "R3", "R4", "total"):
                curves[k].append(round(means[k], 4))
            (output_dir / "curves.json").write_text(json.dumps(curves, indent=2))
            print(
                f"[eval @ step {step:4d}] "
                f"R1={means['R1']:.3f} R2={means['R2']:.3f} "
                f"R3={means['R3']:.3f} R4={means['R4']:.3f} total={means['total']:.3f}",
                flush=True,
            )
            # Collapse safety net — halt if eval regresses > 0.10 below the
            # step-0 baseline. Onsite demo failure has no retry; better to
            # exit(2) with the 30-step backup intact than to push through and
            # ship a collapsed curve. Step-0 is always index 0 of curves.
            if len(curves["total"]) >= 2:
                step0 = curves["total"][0]
                latest = curves["total"][-1]
                if latest < step0 - 0.10:
                    print(
                        f"\n[FATAL] training collapsed: eval total {latest:.3f} < "
                        f"step-0 {step0:.3f} − 0.10. "
                        f"30-step backup preserved.",
                        flush=True,
                    )
                    sys.exit(2)

        def on_train_begin(self, args, state, control, **kwargs):
            if self._did_step_zero:
                return
            self._did_step_zero = True
            model = kwargs.get("model")
            if model is not None:
                self._run_and_log(0, model)

        def on_step_end(self, args, state, control, **kwargs):
            # Step-1 gradient-flow sanity: if grad_norm is 0 at the first
            # logged step, gradients aren't reaching the LoRA params at all
            # (wrong target_modules, frozen adapter, etc.). Halt immediately
            # — 150 steps with zero gradient is a wasted Kaggle slot.
            # logging_steps=1 guarantees step-1 is in state.log_history.
            if state.global_step == 1:
                grad_entries = [e for e in state.log_history if "grad_norm" in e]
                if grad_entries:
                    gn = float(grad_entries[-1]["grad_norm"])
                    print(f"[grad sanity @ step 1] grad_norm={gn:.6f}", flush=True)
                    if gn == 0.0:
                        print(
                            "\n[FATAL] grad_norm=0 at step 1 — gradients not "
                            "flowing into LoRA params. Check peft_config and "
                            "target_modules match the model's attention module "
                            "names.",
                            flush=True,
                        )
                        sys.exit(4)
            if state.global_step == 0 or state.global_step % EVAL_EVERY != 0:
                return
            model = kwargs.get("model")
            if model is None:
                return
            self._run_and_log(state.global_step, model)

    return ComponentEvalCallback()


# ---------- main ----------


def _assert_trl_version() -> None:
    try:
        import trl  # type: ignore
    except ImportError:
        print(
            "ERROR: trl not installed. Install from main:\n"
            "  pip install 'trl @ git+https://github.com/huggingface/trl.git@main'",
            flush=True,
        )
        sys.exit(2)
    ver = getattr(trl, "__version__", "0.0.0")
    major, minor = (int(x) for x in ver.split(".")[:2])
    print(f"trl version: {ver}", flush=True)
    if (major, minor) < (0, 21):
        print(
            f"ERROR: trl {ver} is too old. environment_factory needs >= 0.21.\n"
            "Install from main:\n"
            "  pip install 'trl @ git+https://github.com/huggingface/trl.git@main'",
            flush=True,
        )
        sys.exit(2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GRPO + LoRA via TRL environment_factory."
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--output-dir", type=Path, default=REPO / "data" / "real_training"
    )
    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    args = parser.parse_args()

    _assert_trl_version()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # If a prior 30-step run produced curves.json, preserve it under a
    # descriptive name before the new run overwrites. Onsite this is what
    # we fall back to if the 150-step run collapses or OOMs mid-training.
    existing_curves = args.output_dir / "curves.json"
    if existing_curves.exists():
        backup = args.output_dir / "curves_30step_backup.json"
        existing_curves.rename(backup)
        print(f"backed up prior curves.json → {backup}", flush=True)

    # Write an empty curves.json stub immediately so downstream plotting never
    # hits FileNotFoundError if the process dies mid-training.
    curves: Dict[str, List[Any]] = {
        "steps": [],
        "R1": [],
        "R2": [],
        "R3": [],
        "R4": [],
        "total": [],
        "config": {
            "model": args.model,
            "lora_rank": LORA_RANK,
            "lora_targets": LORA_TARGETS,
            "lr": LEARNING_RATE,
            "total_steps": args.total_steps,
            "eval_every": args.eval_every,
            "checkpoint_every": args.checkpoint_every,
            "num_generations": NUM_GENERATIONS,
            "max_completion_length": MAX_COMPLETION_LENGTH,
            "per_device_batch": PER_DEVICE_BATCH,
            "grad_accum": GRAD_ACCUM,
            "eval_seeds": EVAL_SEEDS,
            "train_seeds_range": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]],
        },
    }
    (args.output_dir / "curves.json").write_text(json.dumps(curves, indent=2))

    # Load model + tokenizer with OOM guard.
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from peft import LoraConfig  # type: ignore

        print(f"loading {args.model} ...", flush=True)
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # Response schema: TRL's add_response_schema supports the Qwen3 family
        # natively (GRPOTrainer re-runs schema inference internally, so a manual
        # override is clobbered — only Qwen3 works end-to-end).
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype="auto", device_map="auto"
        )
        print(f"model loaded in {time.time() - t0:.1f}s on {model.device}", flush=True)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:  # type: ignore[name-defined]
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda" in msg:
            print("\n[FATAL] CUDA OOM on model load.", flush=True)
            traceback.print_exc()
            print(
                "\nQwen3-0.6B is the smallest TRL-supported Qwen3 for "
                "add_response_schema. Recovery options:\n"
                "  1) Drop num_generations from 4 to 2 in the script constants.\n"
                "  2) Drop max_completion_length from 512 to 256.\n"
                "  3) Switch GPU to a larger instance (Kaggle P100 16GB → "
                "L4 24GB if available, or Colab A100 40GB).\n",
                flush=True,
            )
            sys.exit(3)
        raise

    # LoRA config.
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        lora_dropout=0.0,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Dataset — 150 prompt slots. Each row: prompt + seed (passed through to reset()).
    from datasets import Dataset  # type: ignore

    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": SYSTEM_PROMPT}]] * len(TRAIN_SEEDS),
            "seed": TRAIN_SEEDS,
        }
    )

    # GRPOConfig.
    from trl import GRPOConfig, GRPOTrainer  # type: ignore

    grpo_cfg = GRPOConfig(
        output_dir=str(args.output_dir / "grpo_out"),
        max_steps=args.total_steps,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        learning_rate=LEARNING_RATE,
        use_vllm=False,  # Kaggle vLLM story unreliable; HF generation
        logging_steps=1,
        save_steps=args.checkpoint_every,
        save_total_limit=3,
        report_to="none",
        bf16=True,
        remove_unused_columns=False,  # keep ``seed`` column for reset(**kwargs)
    )

    eval_callback = _build_eval_callback(tokenizer, args.output_dir, curves)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=dataset,
        reward_funcs=reward_func,
        environment_factory=ReconcileToolEnv,
        peft_config=lora_cfg,
        callbacks=[eval_callback],
    )

    # Trainable-param visibility — confirms the LoRA expansion (q/k/v/o) is
    # actually attached. A zero or suspiciously-small count means target
    # module names didn't match the model's actual attention layers, which
    # the step-1 grad_norm check would catch downstream but we want to see
    # earlier in the log so the operator can bail before wasting wall-time.
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    pct = 100.0 * n_trainable / n_total if n_total else 0.0
    print(
        f"trainable params: {n_trainable:,} / {n_total:,} ({pct:.3f}%) "
        f"[LoRA rank {LORA_RANK}, targets {LORA_TARGETS}, lr {LEARNING_RATE}]",
        flush=True,
    )

    print(f"\n=== training: {args.total_steps} steps ===", flush=True)
    trainer.train()

    # Final checkpoint.
    final_ckpt = args.output_dir / "checkpoints" / f"step_{args.total_steps}"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_ckpt))
    tokenizer.save_pretrained(str(final_ckpt))
    print(f"\nfinal checkpoint: {final_ckpt}", flush=True)
    print(f"curves: {args.output_dir / 'curves.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
