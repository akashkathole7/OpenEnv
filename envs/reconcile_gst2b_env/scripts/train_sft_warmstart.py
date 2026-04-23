#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Phase 2 of the on-site training pipeline: tool-SFT warm-start.

Pipeline overview (3 phases):
  1. ``generate_sft_trajectories.py``  → CPU,  ~2 h, oracle distillation
  2. ``train_sft_warmstart.py`` (THIS) → A100, ~4 h, supervised fine-tune
  3. ``train_grpo_real.py``           → A100, ~4 h, GRPO polish from SFT

Empirical motivation: data/training_log_qwen3_*_partial.json shows pure
GRPO from base Qwen3 instruct fails to chain mark_* actions across
0.6B/1.7B/4B (three distinct failure modes). The SFT warm-start teaches
the model the action format and label-commitment preference; GRPO then
polishes the R3 (Rule 36(4) compliance) and R4 (step efficiency)
optimization on top of an already-competent policy.

Reads the JSONL produced by ``generate_sft_trajectories.py`` (OpenAI-style
``messages`` field per row), applies the same Qwen3 ``enable_thinking=False``
chat template as the GRPO phase will use, fine-tunes a LoRA adapter on
the same target modules (q/k/v/o, rank 16), and saves a checkpoint
that ``train_grpo_real.py`` can load by setting ``MODEL_NAME`` to the
SFT output dir.

On-site usage (after Phase 1 completes; assumes A100 40GB):
    PYTHONPATH=src:envs uv run python -m \\
        envs.reconcile_gst2b_env.scripts.train_sft_warmstart \\
        --input-jsonl data/sft_trajectories.jsonl \\
        --output-dir  data/sft_checkpoint \\
        --model       Qwen/Qwen3-4B \\
        --epochs      1 \\
        --batch-size  1 \\
        --grad-accum  8 \\
        --learning-rate 2e-5

Smoke test (CPU, 5 examples, ~30 s — verifies code path):
    PYTHONPATH=src:envs uv run python -m \\
        envs.reconcile_gst2b_env.scripts.train_sft_warmstart \\
        --input-jsonl data/sft_trajectories.jsonl \\
        --dry-run

After this completes, point train_grpo_real.py at the checkpoint:
    # In train_grpo_real.py, change:
    MODEL_NAME = "data/sft_checkpoint"  # was "Qwen/Qwen3-4B"
    # then run train_grpo_real.py as usual.

Failure-mode defenses:
* CUDA OOM on model load → caught with recovery guidance (drop batch,
  switch to 4B from 8B, etc.). Mirrors train_grpo_real's OOM handler.
* TRL < 0.21 → halt with install guidance (reuses _assert_trl_version).
* Empty input JSONL → halt with explicit message before tokenizer load.
* enable_thinking monkeypatch failure → diagnostic print at training
  start so the operator can spot a config mismatch before burning hours.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

# Reuse the GRPO script's TRL version check so SFT and GRPO phases
# enforce the same TRL >= 0.21 invariant.
from envs.reconcile_gst2b_env.scripts.train_grpo_real import (  # noqa: E402
    LORA_ALPHA,
    LORA_RANK,
    LORA_TARGETS,
    _assert_trl_version,
)


def _patch_tokenizer_no_thinking(tokenizer: Any) -> None:
    """Force ``enable_thinking=False`` on every chat-template render.

    Same monkeypatch as in train_grpo_real.py — keeps SFT data
    distribution aligned with GRPO-time generation. Without this,
    Qwen3's default thinking-mode preamble would balloon every
    training example by 100-300 tokens of <think>...</think> noise.
    """
    _orig = tokenizer.apply_chat_template

    def _no_think(*a: Any, **kw: Any) -> Any:
        kw.setdefault("enable_thinking", False)
        return _orig(*a, **kw)

    tokenizer.apply_chat_template = _no_think  # type: ignore[method-assign]
    print("tokenizer patched: enable_thinking forced to False", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2: tool-SFT warm-start LoRA fine-tune for GRPO bootstrapping."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="JSONL produced by generate_sft_trajectories.py (one conversation per row).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "data" / "sft_checkpoint",
        help="LoRA adapter output directory (consumable by train_grpo_real.py).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B",
        help="Base instruct model. 4B is the smallest with documented function-calling alignment.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Per-example max length. Multi-turn conversations can run 3-6K tokens; "
        "truncating at 2048 keeps per-step wall time tractable on T4. "
        "Larger values are viable on A100 (see ONSITE_BRIEFING.md Phase 2).",
    )
    parser.add_argument(
        "--precision",
        choices=("fp16", "bf16"),
        default="fp16",
        help="Mixed-precision mode. fp16 is correct for T4 (Turing tensor cores are "
        "fp16-only, bf16 falls back to slow CUDA-core compute). Use bf16 on Ampere+.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Learning-rate warmup fraction.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=5,
        help="Per-step train-loss logging frequency.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
        help="Checkpoint save frequency (in optimizer steps).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load 5 examples, instantiate trainer, exit before train(). CPU-runnable.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Cap the dataset to the first N rows. Use for smoke cycles "
        "(e.g. --limit-rows 300) to verify the training loop reaches the "
        "first loss log before committing to a full run.",
    )
    args = parser.parse_args()

    if not args.input_jsonl.exists():
        print(
            f"ERROR: input JSONL not found at {args.input_jsonl}.\n"
            f"  Run generate_sft_trajectories.py first to produce it.",
            flush=True,
        )
        return 1

    # Sanity: ensure JSONL is non-empty before paying tokenizer/model load cost.
    n_lines = sum(1 for _ in args.input_jsonl.open())
    if n_lines == 0:
        print(
            f"ERROR: {args.input_jsonl} is empty. Re-run trajectory generation.",
            flush=True,
        )
        return 1
    print(f"input JSONL: {args.input_jsonl} ({n_lines} trajectories)", flush=True)

    _assert_trl_version()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Defer heavy imports until after CLI / file-existence checks pass.
    try:
        import torch  # type: ignore
        from datasets import load_dataset  # type: ignore
        from peft import LoraConfig  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from trl import SFTConfig, SFTTrainer  # type: ignore
    except ImportError as exc:
        print(
            f"ERROR: missing training deps: {exc}\n"
            "  Install with:\n"
            "    pip install 'trl @ git+https://github.com/huggingface/trl.git@main'\n"
            "    pip install 'transformers @ git+https://github.com/huggingface/transformers.git@main'\n"
            "    pip install peft accelerate bitsandbytes datasets",
            flush=True,
        )
        return 2

    print(f"loading {args.model} ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    _patch_tokenizer_no_thinking(tokenizer)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype="auto", device_map="auto"
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:  # type: ignore[name-defined]
        msg = str(exc).lower()
        if "out of memory" in msg or "cuda" in msg:
            print("\n[FATAL] CUDA OOM on model load.", flush=True)
            traceback.print_exc()
            print(
                "\nRecovery options for SFT phase:\n"
                "  1) Drop --max-seq-length from 4096 to 2048\n"
                "  2) Drop --batch-size to 1 (already default)\n"
                "  3) Increase --grad-accum to keep effective batch size constant\n"
                "  4) Switch to a smaller base model (Qwen3-1.7B, but tool-call\n"
                "     alignment is weaker — see data/training_log_qwen3_1_7b_partial.json)\n",
                flush=True,
            )
            return 3
        raise
    print(f"model loaded in {time.time() - t0:.1f}s on {model.device}", flush=True)

    # Load dataset; for dry-run, slice down to 5 examples for fast smoke check.
    ds = load_dataset("json", data_files=str(args.input_jsonl), split="train")
    if args.dry_run:
        ds = ds.select(range(min(5, len(ds))))
        print(f"[dry-run] using {len(ds)} examples", flush=True)
    elif args.limit_rows is not None:
        ds = ds.select(range(min(args.limit_rows, len(ds))))
        print(
            f"[--limit-rows {args.limit_rows}] using {len(ds)} training examples "
            f"(smoke-cycle subset)",
            flush=True,
        )
    else:
        print(f"loaded {len(ds)} training examples", flush=True)

    # LoRA config — keep ranks/targets identical to train_grpo_real.py so the
    # SFT checkpoint slots cleanly into the GRPO phase without re-init.
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,  # higher than GRPO's 0.0; SFT has more data, regularize more
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )

    use_bf16 = args.precision == "bf16"
    use_fp16 = args.precision == "fp16"
    # Do not pass dataset_text_field. In TRL >=1.0 the default ("text") plus
    # the collator's own messages-column auto-detect is what actually routes
    # chat-format data; passing None in 1.x was observed to stall silently
    # between __init__ and the first step (no progress bar, ~1.5h silence
    # on Kaggle T4 with Qwen3-1.7B). See RCA 2026-04-23.
    sft_cfg = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        max_length=args.max_seq_length,
        packing=False,  # multi-turn convos: don't concat across examples
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        peft_config=lora_cfg,
        processing_class=tokenizer,
    )

    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    pct = 100.0 * n_trainable / n_total if n_total else 0.0
    print(
        f"trainable params: {n_trainable:,} / {n_total:,} ({pct:.3f}%) "
        f"[LoRA rank {LORA_RANK}, targets {LORA_TARGETS}, lr {args.learning_rate}]",
        flush=True,
    )

    if args.dry_run:
        print(
            "\n[dry-run] trainer instantiated successfully. Skipping train() call.\n"
            "  Verifies: input JSONL parses, model loads, tokenizer patches, "
            "  LoRA attaches, SFTTrainer accepts the config.\n"
            "  To actually train, omit --dry-run.",
            flush=True,
        )
        return 0

    print(
        f"\n=== SFT training: {args.epochs} epoch(s) over {len(ds)} examples ===",
        flush=True,
    )
    train_result = trainer.train()

    # Save the LoRA adapter + tokenizer for downstream GRPO consumption.
    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print(f"\nfinal LoRA adapter: {final_dir}", flush=True)
    print(
        f"\nNext: edit train_grpo_real.py to set\n"
        f'  MODEL_NAME = "{final_dir}"\n'
        f"and run the GRPO polish phase as usual.",
        flush=True,
    )

    # Summary stats for the run record.
    metrics = train_result.metrics if hasattr(train_result, "metrics") else {}
    summary = {
        "model": args.model,
        "input_jsonl": str(args.input_jsonl),
        "n_examples": len(ds),
        "epochs": args.epochs,
        "trainable_params": n_trainable,
        "trainable_pct": round(pct, 4),
        "lora_rank": LORA_RANK,
        "lora_targets": LORA_TARGETS,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "metrics": {
            k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
        },
        "final_checkpoint": str(final_dir),
    }
    summary_path = args.output_dir / "sft_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
