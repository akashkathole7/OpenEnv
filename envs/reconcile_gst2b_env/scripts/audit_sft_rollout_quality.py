#!/usr/bin/env python3
"""5-metric rollout-quality audit for the SFT-trained Qwen3-4B checkpoint.

Runs greedy rollouts on held-out seeds 9030-9034 against the merged checkpoint
and reports the 5 trajectory-quality metrics from ONSITE_DAY1_PROMPT.md
Action 5: query_to_mark, mean_inspection_depth, repeated_query_rate,
tool_diversity, premature_marking_rate.

Uses the same Qwen <tool_call>{name, arguments}</tool_call> format the SFT
data was generated in (see scripts/generate_sft_trajectories.py).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from envs.reconcile_gst2b_env.models import ReconcileAction  # noqa: E402
from envs.reconcile_gst2b_env.rewards import composite_reward  # noqa: E402
from envs.reconcile_gst2b_env.server.reconcile_gst2b_environment import (  # noqa: E402
    ReconcileGST2BEnvironment,
)


SYSTEM_PROMPT = (
    "You are a GST reconciliation agent. Each episode is one company's monthly "
    "purchase register vs the GSTR-2B return. Call the available tool methods "
    "to inspect invoices, query the schema, and mark each invoice with one of "
    "five labels (matched, mismatched, only_in_books, only_in_2b, partial). "
    "You MUST call at least one query tool before `submit` — submitting "
    "without a prior query returns −1.0. Reward is weighted macro-F1 on "
    "labels + ITC delta accuracy + per-supplier Rule 36(4) compliance + step "
    "efficiency. Call `submit` when done."
)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


def _str_p(desc):
    return {"type": "string", "description": desc}


# 16-verb tool schema (mirrors ReconcileToolEnv in train_grpo_real.py).
TOOLS_DEF = [
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": "Return the env schema: verbs, labels, max_steps, company_gstin, invoice counts.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_gstins",
            "description": "List all supplier GSTINs in purchase_register or gstr_2b.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fuzzy_search_gstin",
            "description": "Fuzzy match a GSTIN substring against known suppliers.",
            "parameters": {
                "type": "object",
                "properties": {"query": _str_p("Partial GSTIN or pattern.")},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_invoice",
            "description": "Fetch a single invoice from purchase_register or gstr_2b.",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": _str_p("e.g. inv_0005 or inv_2bonly_0001.")
                },
                "required": ["invoice_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_invoices_by_supplier",
            "description": "List all invoices for a specific supplier across books and 2B.",
            "parameters": {
                "type": "object",
                "properties": {"gstin": _str_p("15-char supplier GSTIN.")},
                "required": ["gstin"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_2b_row",
            "description": "Fetch an invoice's GSTR-2B-side record.",
            "parameters": {
                "type": "object",
                "properties": {"invoice_id": _str_p("e.g. inv_0005.")},
                "required": ["invoice_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hsn_slab",
            "description": "Look up the GST slab (percent) for an HSN code.",
            "parameters": {
                "type": "object",
                "properties": {"hsn": _str_p("4-character HSN prefix, e.g. 8517.")},
                "required": ["hsn"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_matched",
            "description": "Label an invoice as matched (books and 2B agree).",
            "parameters": {
                "type": "object",
                "properties": {"invoice_id": _str_p("Target invoice id.")},
                "required": ["invoice_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_mismatched",
            "description": "Label an invoice as mismatched with a reason code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": _str_p("Target invoice id."),
                    "reason": _str_p(
                        "One of: value | tax | date | gstin | hsn | invoice_no."
                    ),
                },
                "required": ["invoice_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_only_in_books",
            "description": "Label an invoice as present only in purchase_register (not in 2B).",
            "parameters": {
                "type": "object",
                "properties": {"invoice_id": _str_p("Target invoice id.")},
                "required": ["invoice_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_only_in_2b",
            "description": "Label an invoice as present only in GSTR-2B (not in books).",
            "parameters": {
                "type": "object",
                "properties": {"invoice_id": _str_p("Target invoice id.")},
                "required": ["invoice_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_partial_match",
            "description": "Label an invoice as a partial match; claim books tax minus delta.",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": _str_p("Target invoice id."),
                    "delta_inr": {
                        "type": "number",
                        "description": "Amount (INR) by which books overstates 2B tax.",
                    },
                },
                "required": ["invoice_id", "delta_inr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_circular_ring",
            "description": "Flag a list of supplier GSTINs as a suspected circular-trading ring.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gstins": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of GSTINs suspected to form a cycle (typically 3).",
                    }
                },
                "required": ["gstins"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_amendment",
            "description": "Request an amendment to a specific field on an invoice.",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": _str_p("Target invoice id."),
                    "field": _str_p("Field name, e.g. value_inr or gstin."),
                    "value": _str_p("New value (stringified)."),
                },
                "required": ["invoice_id", "field", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_with_user",
            "description": "Ask a clarifying question (no state effect).",
            "parameters": {
                "type": "object",
                "properties": {"question": _str_p("The clarifying question.")},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Signal the episode is complete.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

QUERY_VERBS = {
    "get_schema",
    "list_gstins",
    "fuzzy_search_gstin",
    "get_invoice",
    "list_invoices_by_supplier",
    "get_2b_row",
    "get_hsn_slab",
}
MARK_VERBS = {
    "mark_matched",
    "mark_mismatched",
    "mark_only_in_books",
    "mark_only_in_2b",
    "mark_partial_match",
}
INSPECT_VERBS = {"get_invoice", "get_2b_row", "list_invoices_by_supplier"}


def _build_user_prompt(obs):
    """Mirror generate_sft_trajectories._format_user_turn exactly."""
    return (
        f"Step budget remaining: {obs.step_budget}\n"
        f"Invoices remaining: {obs.invoices_remaining_count}\n"
        f"Last tool result: {json.dumps(obs.last_tool_result or {})[:600]}\n"
        'Next action: emit a <tool_call> block with "name" and "arguments".'
    )


def _parse_tool_call(text):
    """Extract (verb, payload) from a Qwen3 tool_call emission.

    Three observed formats:
      1. <tool_call>\\n"name": ..., "arguments": ...\\n</tool_call>  (braceless inner)
      2. <tool_call>{"name": ..., "arguments": ...}</tool_call>      (braced inner, closed)
      3. <tool_call>{"name": ..., "arguments": ...}                  (braced inner, no close)
    """
    idx = text.find(TOOL_CALL_OPEN)
    if idx < 0:
        return None
    rest = text[idx + len(TOOL_CALL_OPEN) :]
    stripped = rest.lstrip()
    body = None
    if stripped.startswith("{"):
        # Brace-counted JSON object (handles missing close tag).
        depth = 0
        end = 0
        for i, c in enumerate(stripped):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == 0:
            return None
        body = stripped[:end]
    else:
        # Braceless inner — bounded by close tag or end-of-string.
        close = rest.find(TOOL_CALL_CLOSE)
        inner = (rest[:close] if close >= 0 else rest).strip()
        body = "{" + inner + "}"
    try:
        d = json.loads(body)
    except json.JSONDecodeError:
        return None
    verb = d.get("name")
    payload = d.get("arguments", {}) or {}
    if not verb:
        return None
    return verb, payload


def _generate_from_messages(
    model,
    tokenizer,
    messages,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    tools=None,
):
    import torch

    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if tools is not None:
        template_kwargs["tools"] = tools
    try:
        prompt = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **template_kwargs
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        if top_k > 0:
            gen_kwargs["top_k"] = top_k
    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)
    gen = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    return gen


def _run_episode(
    seed,
    model,
    tokenizer,
    max_steps,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    tools=None,
):
    """Run one episode with accumulating multi-turn conversation context.

    Matches generate_sft_trajectories.py: ONE messages list across all turns,
    appending {role:user, content:user_prompt(obs)} and {role:assistant,
    content:gen} each turn. Without accumulation the model is fed a "turn 1"
    shape every step, which is OOD relative to SFT training and produces the
    spurious over-query collapse signature.
    """
    from pydantic import ValidationError

    env = ReconcileGST2BEnvironment()
    obs = env.reset(seed=seed)
    trajectory = []
    raw_completions = []
    parser_misses = 0
    off_vocab_misses = 0
    off_vocab_attempted = []
    step_count = 0
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    while not obs.done and step_count < max_steps:
        messages.append({"role": "user", "content": _build_user_prompt(obs)})
        gen = _generate_from_messages(
            model,
            tokenizer,
            messages,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            tools=tools,
        )
        raw_completions.append(gen)
        messages.append({"role": "assistant", "content": gen})
        parsed = _parse_tool_call(gen)
        if parsed is None:
            parser_misses += 1
            action = ReconcileAction(verb="submit", payload={})
        else:
            verb, payload = parsed
            try:
                action = ReconcileAction(verb=verb, payload=payload)
            except ValidationError:
                off_vocab_misses += 1
                off_vocab_attempted.append(verb)
                action = ReconcileAction(verb="submit", payload={})
        trajectory.append({"verb": action.verb, "payload": dict(action.payload)})
        obs = env.step(action)
        step_count += 1

    if not env.state.reward_breakdown:
        env.state.reward_breakdown = composite_reward(env.state, env._trajectory)
    breakdown = env.state.reward_breakdown
    return {
        "seed": seed,
        "trajectory": trajectory,
        "raw_first_completion": raw_completions[0] if raw_completions else "",
        "raw_last_completion": raw_completions[-1] if raw_completions else "",
        "parser_misses": parser_misses,
        "off_vocab_misses": off_vocab_misses,
        "off_vocab_attempted": off_vocab_attempted,
        "total_steps": step_count,
        "termination": (obs.metadata or {}).get("termination_reason", "unknown"),
        "reward_breakdown": {k: round(float(v), 4) for k, v in breakdown.items()},
    }


def audit(trajectories):
    per_seed = []
    for traj in trajectories:
        verbs = [a["verb"] for a in traj]
        n_q = sum(1 for v in verbs if v in QUERY_VERBS)
        n_m = sum(1 for v in verbs if v in MARK_VERBS)
        marked_invs = [
            a["payload"].get("invoice_id") for a in traj if a["verb"] in MARK_VERBS
        ]
        inspections_per_inv = collections.defaultdict(set)
        for a in traj:
            if a["verb"] in INSPECT_VERBS:
                inv = a["payload"].get("invoice_id") or a["payload"].get("gstin")
                if inv:
                    inspections_per_inv[inv].add(a["verb"])
        depths = [len(inspections_per_inv.get(i, set())) for i in marked_invs]
        seen = set()
        repeated = 0
        for a in traj:
            if a["verb"] in QUERY_VERBS:
                key = (a["verb"], tuple(sorted((a["payload"] or {}).items())))
                if key in seen:
                    repeated += 1
                seen.add(key)
        inspected = collections.defaultdict(set)
        premature = 0
        for a in traj:
            if a["verb"] in {"get_invoice", "get_2b_row"}:
                inv = a["payload"].get("invoice_id")
                if inv:
                    inspected[inv].add(a["verb"])
            elif a["verb"] in MARK_VERBS:
                inv = a["payload"].get("invoice_id")
                if inv and not inspected[inv]:
                    premature += 1
        per_seed.append(
            {
                "query_to_mark": n_q / max(n_m, 1),
                "mean_inspection_depth": sum(depths) / max(len(depths), 1),
                "repeated_query_rate": repeated / max(n_q, 1),
                "tool_diversity": len(set(verbs)) / 16,
                "premature_marking_rate": premature / max(n_m, 1),
            }
        )
    keys = list(per_seed[0].keys())
    avg = {k: sum(r[k] for r in per_seed) / len(per_seed) for k in keys}
    return avg, per_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, help="path to merged checkpoint dir"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[9030, 9031, 9032, 9033, 9034]
    )
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 = greedy. >0 enables do_sample. Match GRPO with 0.7.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="nucleus sampling. Match GRPO with 0.95.",
    )
    parser.add_argument(
        "--top-k", type=int, default=0, help="0 = no top-k. Match GRPO with 20."
    )
    parser.add_argument(
        "--use-tools",
        action="store_true",
        help="Pass 16-verb tool schemas to apply_chat_template "
        "(triggers Qwen3 native tool-call mode).",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    print(f"loading merged checkpoint from {args.checkpoint} ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype="auto", device_map="auto"
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s on {model.device}", flush=True)
    sampling_mode = (
        "greedy"
        if args.temperature == 0
        else (f"sample T={args.temperature} top_p={args.top_p} top_k={args.top_k}")
    )
    print(f"  sampling: {sampling_mode}", flush=True)
    print(f"  use_tools: {args.use_tools}", flush=True)

    tools = TOOLS_DEF if args.use_tools else None
    rollouts = []
    for seed in args.seeds:
        t1 = time.time()
        r = _run_episode(
            seed,
            model,
            tokenizer,
            args.max_steps,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            tools=tools,
        )
        elapsed = time.time() - t1
        verbs_list = [a["verb"] for a in r["trajectory"]]
        verb_counts = collections.Counter(verbs_list)
        verb_summary = ", ".join(f"{v}={n}" for v, n in verb_counts.most_common(5))
        ov = f" off_vocab={r['off_vocab_misses']}" if r["off_vocab_misses"] else ""
        ov_attempts = (
            f"  attempted={r['off_vocab_attempted']}"
            if r["off_vocab_attempted"]
            else ""
        )
        print(
            f"seed={seed}: {r['total_steps']} steps  misses={r['parser_misses']}{ov}  "
            f"term={r['termination']}  R_total={r['reward_breakdown'].get('total', 'n/a')}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        print(f"  top-5 verbs: {verb_summary}{ov_attempts}", flush=True)
        rollouts.append(r)

    metrics_avg, per_seed = audit([r["trajectory"] for r in rollouts])

    print("\n=== AUDIT METRICS (averaged across seeds) ===")
    thresholds = {
        "query_to_mark": ("≥ 0.5", lambda v: v >= 0.5),
        "mean_inspection_depth": ("≥ 1.5", lambda v: v >= 1.5),
        "repeated_query_rate": ("≤ 0.3", lambda v: v <= 0.3),
        "tool_diversity": ("≥ 0.30", lambda v: v >= 0.30),
        "premature_marking_rate": ("≤ 0.3", lambda v: v <= 0.3),
    }
    fail_count = 0
    for k, v in metrics_avg.items():
        thr_str, thr_fn = thresholds[k]
        verdict = "PASS" if thr_fn(v) else "FAIL"
        if verdict == "FAIL":
            fail_count += 1
        print(f"  {k:<28}: {v:.4f}   threshold {thr_str:<10} -> {verdict}")
    print(f"\nFAIL count: {fail_count}/5")

    print("\n=== PER-SEED METRICS ===")
    for seed, m in zip(args.seeds, per_seed):
        print(f"  seed={seed}: " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))

    print(
        "\n=== SAMPLE TRAJECTORIES (first 2 seeds, all verbs + invoice_id payload) ==="
    )
    for r in rollouts[:2]:
        print(
            f"\n--- seed {r['seed']}  steps={r['total_steps']}  "
            f"R_total={r['reward_breakdown'].get('total', 'n/a')}  "
            f"R={r['reward_breakdown']} ---"
        )
        for i, a in enumerate(r["trajectory"]):
            inv = a["payload"].get("invoice_id") or a["payload"].get("gstin") or ""
            print(f"  [{i:2d}] {a['verb']:<28} inv={inv}")
        print(f"  raw first completion: {r['raw_first_completion'][:200]!r}")

    out = {
        "checkpoint": args.checkpoint,
        "seeds": args.seeds,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "do_sample": args.temperature > 0,
            "use_tools": args.use_tools,
        },
        "metrics_avg": metrics_avg,
        "per_seed_metrics": per_seed,
        "fail_count": fail_count,
        "rollouts": [
            {
                "seed": r["seed"],
                "total_steps": r["total_steps"],
                "termination": r["termination"],
                "parser_misses": r["parser_misses"],
                "off_vocab_misses": r["off_vocab_misses"],
                "off_vocab_attempted": r["off_vocab_attempted"],
                "reward_breakdown": r["reward_breakdown"],
                "trajectory": r["trajectory"],
                "raw_first_completion": r["raw_first_completion"],
                "raw_last_completion": r["raw_last_completion"],
            }
            for r in rollouts
        ],
    }
    if args.output is None:
        ckpt_tag = Path(args.checkpoint).name
        out_path = (
            REPO / "envs" / "reconcile_gst2b_env" / "data" / f"audit_{ckpt_tag}.json"
        )
    else:
        out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
