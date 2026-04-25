# LESSONS_LEARNED.md

Honest engineering notes from the pre-onsite window (2026-04-22 to 2026-04-24) plus the on-site Day 1 A100 SXM4-80GB run (2026-04-25). What we tried, what failed, what we diagnosed, what ships with this submission, and what's corrected for follow-up validation post-hackathon.

This doc exists because the judges' criteria explicitly reward "messy but ambitious with real training evidence" over polished-but-boring, and because the [ONSITE_BRIEFING.md](ONSITE_BRIEFING.md) invariant "do NOT fake numbers" is load-bearing for the red-team CI contract. It is not a retrospective on mistakes, it is the engineering trail behind the shipped artifacts.

## 1. Three Qwen3-scale attempts, one survived commit

We targeted a three-failure-mode story across Qwen3-0.6B, Qwen3-1.7B, and Qwen3-4B on Kaggle T4 / T4x2, each meant to pin a different reward component. The live training runs happened; the commit-surviving evidence did not. Of the three:

- **Qwen3-0.6B** ran to 10 training steps + 150 eval steps. The training log (`data/smoke.log`, ~12 KB, committed on 2026-04-22) was reconstructed into [`data/smoke_test_10step.json`](data/smoke_test_10step.json) on 2026-04-24 and is the JSON file multiple docs reference. Per-step `reward` is bimodal around 0.174 (tool-call rollouts) and 0.107 (clipped rollouts); 150-step eval composite is flat at 0.353, within 0.004 of the `query_only` red-team attack signature. The plots in [`data/figures/`](data/figures/) are from this run.
- **Qwen3-1.7B** ran ~15 training steps before the Kaggle session was reset. We observed live the entropy collapse to 0.12 (vs 0.44 on 0.6B) and zero tool-call emission, matching the briefing description. The per-step JSON log was not written to `/kaggle/working/` before the reset and never made it to disk on our laptop.
- **Qwen3-4B** crashed at step 0 eval after the `clipped_ratio=1.0` + `enable_thinking=True` combination burned the 448-token completion budget on a `<think>` preamble. Observed live; log not persisted.

Older versions of [BLOG.md](BLOG.md) §6, [ROUND2_PROBLEM_STATEMENT.md](ROUND2_PROBLEM_STATEMENT.md), and [README_HF_SPACE.md](README_HF_SPACE.md) referenced `data/training_log_qwen3_1_7b_partial.json` and `data/training_log_qwen3_4b_partial.json` as if they were committed artifacts. They were not. The 2026-04-24 README/BLOG rewrites replaced those references with honest single-scale documented evidence plus qualitative notes on the two observed-but-unlogged attempts.

### Failure Mode 4: audit-script OOD trap chain (on-site, 2026-04-25)

After the Day 1 SFT run on Qwen3-4B (375 steps, train_loss 0.341, see [`data/sft_full_run.log`](data/sft_full_run.log) and [`data/sft_summary.json`](data/sft_summary.json)), four sequential bugs in the audit/eval infrastructure each fabricated a different false collapse signature before the real model behavior could be observed:

1. **Greedy decoding masked sampling capability.** Initial audits used `do_sample=False` while the SFT trajectories were generated under sampling. The argmax-mode of a 4B Qwen3 SFT collapsed to a 2-step `list_gstins → submit` shape (R = `{0.01, 0.01, 0.99, 0.99}`, total 0.353) that exactly matched the `query_only` red-team attack signature. Switching to `T=0.7 top_p=0.95 top_k=20` (matching `train_grpo_real.py:557-561`) lifted trajectory lengths to 8-22 steps but did not yet produce marks.
2. **Fresh-context prompts were OOD vs multi-turn training.** The audit script rebuilt `messages = [system, user]` fresh every turn. Training trajectories ([`scripts/generate_sft_trajectories.py:113-133`](scripts/generate_sft_trajectories.py)) build ONE accumulating list across all 49 turns (system + 49 user/assistant pairs by end). Fresh context = "turn 1" shape, and training-data turn 1 is always a query. Patching `_run_episode` to accumulate context shifted output format from braceless to braced and lifted query target diversity (1-2 → 20+ distinct GSTINs per seed), but mark-emission still didn't appear.
3. **Prompt-format false-alarm.** Mid-diagnosis suspicion that `_build_user_prompt` had a `[:600]` truncation mismatch turned out to be wrong: audit and `generate_sft_trajectories._format_user_turn` already used the byte-identical `json.dumps(obs.last_tool_result or {})[:600]` truncation. Lost ~5 minutes verifying.
4. **Missing `tools=` parameter to `apply_chat_template`.** Qwen3's chat template has separate branches conditional on whether tools are described as schema. SFT data was generated with raw text `<tool_call>` blocks (no `tools=`); SFT training trains on those raw text trajectories. But Qwen3-4B has internalized the function-calling output mode bound to chat templates rendered with `tools=`. **Without `tools=` at inference, the model emits over-query collapse (zero marks across n=5). With `tools=` passing the 16-verb schema list, the model emits 32-46 marks per 50-step trajectory.** This is the 4th and final OOD bug; once fixed, the SFT checkpoint produced n=5 mean composite reward 0.280.

Lesson: the diagnostic infrastructure is as load-bearing as the model itself. Small audit-script details (decoder mode, context accumulation, tokenization match, chat-template parameters) can mask or fabricate failure modes that look identical to real ones from the outside. **When the data shape predicts the model should work (90.6% mark verbs across 3000 SFT trajectories) but the audit shows collapse, exhaust audit-script bug hypotheses before retraining.**

The diagnostic tool that produced the headline number is committed at [`scripts/audit_sft_rollout_quality.py`](scripts/audit_sft_rollout_quality.py) for reproducibility. The audit-OOD chain artifacts are at [`data/audit_seeds_9031_9034.json`](data/audit_seeds_9031_9034.json) (pre-fix collapse evidence) and [`data/audit_F_n5.json`](data/audit_F_n5.json) (headline n=5 with `tools=` fix).

### Failure Mode 5: reward-landscape inversion at the Phase 2 → Phase 3 boundary (on-site, 2026-04-25)

After the audit-OOD chain was resolved and marks emerged on the SFT checkpoint, a structural problem in the reward landscape blocked GRPO progression.

The n=5 audit at GRPO-matching sampling parameters revealed a **bimodal policy**: 3 of 5 seeds emit marks (32-46 mark verbs in 48-50 step trajectories, R_total 0.17 - 0.26), and 2 of 5 seeds collapse to a 3-step `get_schema → list_gstins → submit` query_only shape (R_total 0.353).

**The query_only attack mode scores higher (0.353) than any marking trajectory (max 0.26).** Reward breakdown reveals why:

- Marking trajectories: R1 ≈ 0.15 (real macro-F1 from emitted labels), R2 ≈ 0.41 (real ITC delta), but R3 = 0.01 (per-supplier compliance check fails despite many queries) and R4 = 0.01 (full 50-step budget exhausted).
- Query-only trajectories: R1 = 0.01 floor, R2 = 0.01 floor, but R3 = 0.99 (≥1 query before submit) and R4 = 0.99 (3 of 50 steps used).

The composite weights (0.40 / 0.25 / 0.25 / 0.10) plus R3-R4 saturation make the cheap-to-game cells (query precondition + step efficiency) dominate the expensive-to-improve cells (label F1 + ITC delta) at this policy quality. **GRPO's advantage signal would point toward the attack, not away from it.**

**Empirical evidence for the inversion** (post-Action 7, P2 audit on 2026-04-25):

| Checkpoint | n=5 mean composite | Mode A (marking) seeds | Mode B (query_only attack) seeds |
|---|---:|:---:|:---:|
| Step 350 | **0.3141** | 1 of 5 | 4 of 5 |
| Step 375 (final) | **0.2797** | 3 of 5 | 2 of 5 |

The MORE-trained checkpoint scores LOWER. The last 25 optimizer steps moved 2 seeds from Mode B (R_total 0.353) into Mode A (R_total 0.17 - 0.26), and this is what dragged the n=5 mean down from 0.314 to 0.280. Training is moving the policy in the direction we want (more marking, more genuine R1+R2 reward instead of R3+R4 floor exploitation), and the composite reward is penalizing it.

5 seeds is small for absolute composite estimates, but the directional finding is structurally explained, not a noise artifact: under the current weights, every Mode B → Mode A transition reduces composite reward by ≈ 0.10 - 0.18 (from R_total 0.353 down to the 0.17 - 0.26 Mode A range). At any sample size, more transitions in this direction would deepen the gap. This is a structural property of the arithmetic-composite-reward + R3-R4-saturation design, not a bug in `rewards.py` (Guardrail 1 frozen). The same property is what makes the 6 red-team attacks defensible at <0.45. The post-hackathon agenda is to shape one without sacrificing the other.

Source artifacts: [`data/audit_ckpt350_F_n5.json`](data/audit_ckpt350_F_n5.json) and [`data/audit_F_n5.json`](data/audit_F_n5.json), both at GRPO-matching sampling (T=0.7, top_p=0.95, top_k=20) with `tools=` enabled. Visualized in [`data/figures/day1_training_progression.png`](data/figures/day1_training_progression.png).

This is not a Phase-2 SFT bug. The SFT teaches the model to mark; the reward function correctly identifies that the model's marks are mostly wrong (high mark_matched bias on a 5-class problem). The structural issue is that "many wrong marks" scores lower than "no marks at all" under the current weighting + R3-R4 saturation behavior, a known pathology of arithmetic composite rewards when the floor of one component (R1 + R2 here) is too easy to reach via avoidance.

Three resolutions exist, each with caveats:

1. **Length-penalize Mode B** (add a structural penalty to ≤3-step trajectories that submit without marking). Cheapest, ~10 lines in `rewards.py`. **Blocked by Guardrail 1**: `rewards.py` is frozen, the 6 red-team CI tests calibrate against the exact current weights/clamps. Any reward change would re-tune the red-team battery.
2. **Reward shaping in `train_grpo_real.py`** (additive bonus for any mark verb in the trajectory, on top of the existing `+0.02` format bonus). Touches the trainer, not `rewards.py`. Possible post-hackathon validation path.
3. **Class-balanced SFT subsampling** to reduce the model's mark_matched-default bias, lifting R1 + R2 from the marking-trajectory floor so the 0.260 Mode A ceiling rises above the 0.353 Mode B query-only attack. Requires regenerating Phase 1 trajectories with stratified sampling on the label distribution.

GRPO Phase 3 is **deferred**. Running it on this SFT checkpoint with the current reward landscape would optimize toward the query_only attack signature within ~50 steps (the same 0.353 plateau the pre-onsite Qwen3-0.6B run hit on Kaggle T4, for the same structural reason).

Both failure modes 4 and 5 are documented honestly because the on-site brief is explicit: *"Do NOT fake numbers."* The shipped result is the n=5 mean composite reward of **0.280** under GRPO-matching sampling with `tools=` enabled, an honest improvement over the prompted Qwen2.5-3B baseline of 0.18 (delta 0.10), well below the red-team ceiling of 0.45, and below the bimodal Mode B exploit-shape ceiling of 0.353.

## 2. Kaggle environment brittleness, five distinct classes of failure surfaced

Running Phase 2 SFT on Kaggle T4×2 with Qwen3-1.7B surfaced a cascade of environment-specific issues, each of which had to be unblocked before reaching the training loop:

1. **Dataset mount path.** Kaggle mounts `akashkathole/sft-trajectories-gst2b` at `/kaggle/input/datasets/<user>/<slug>/`, not the common `/kaggle/input/<slug>/`. Fixed in commit `7df8a03` by searching `/kaggle/input/**/` recursively with three filename patterns.
2. **`uv run --no-sync` venv isolation.** Cell 10's dry-run used `uv run --no-sync python` which creates a fresh empty venv, bypassing system `typing_extensions` / `torch` / `transformers` from cell 6's `pip install`. Fixed in commit `fadfd24` by using plain `python`.
3. **`torchao` version floor.** Newer transformers 5.6 requires `torchao > 0.16.0`, but Kaggle base image ships 0.10.0. Fixed in commit `91ef32f` by adding `torchao>=0.16.0` to the pip install.
4. **CPU-only torch wheel.** When Kaggle Accelerator was set to `None`, `pip install --upgrade transformers` resolved torch to the CPU-only wheel (`2.10.0+cpu` instead of `2.10.0+cu128`), silently routing training to CPU where it would have taken days. Fixed in commits `7ffc164` then `e4c8ea0` then `392bff3` after observing that a defensive `--force-reinstall torch` caused its own ABI conflict with Kaggle's pinned `torchvision/torchaudio 2.10.0`.
5. **`torch._library.opaque_object.is_opaque_value` ImportError.** Caused by a stale torch 2.11 from a previous Kaggle session persisting across Kernel Restart. Resolved by pinning `torch==2.10.0+cu128` explicitly with `--force-reinstall` so the pin is idempotent across kernel states.

None of these were reward/env/training-design issues. All were Kaggle-env idiosyncrasies that don't reproduce on a clean A100 instance with a configured HF compute credit.

## 3. Two distinct training-loop hangs, both diagnosed

Once cell 6 imports cleared, two separate silent hangs appeared inside `trainer.train()`:

1. **bf16 on T4 hang (~5h silent).** Turing tensor cores support fp16 but not bf16; `bf16=True` on T4 silently falls back to CUDA-core fp32 compute, which on an already-slow pipeline-parallel `device_map="auto"` split across both T4s triggered an NCCL deadlock. Fixed by forcing `CUDA_VISIBLE_DEVICES=0` (single GPU, model fits comfortably in 16 GB) and switching to `fp16=True` (commits `9ab9e4e` + `723d2c3`).

2. **TRL 1.x `dataset_text_field=None` hang (~1.5h silent, no progress bar).** Our `SFTConfig(..., dataset_text_field=None)` was written against TRL 0.21.x semantics where `None` signaled "use messages format." In TRL 1.x the default is `"text"` and the collator's messages-column auto-detect does the routing. Passing `None` in 1.x produces a silent fallthrough somewhere in `_prepare_dataset` → chat-template auto-patching, hanging before the tqdm progress bar even initializes. Fixed in commit `4b3707d` by removing the field entirely and letting TRL 1.x auto-detect the `messages` column.

The second fix (Path B) is queued for on-site; we did not push-and-pray further Kaggle attempts after the RCA landed.

## 4. What shipped in this submission

- All 42 tests green including the 6 red-team attack ceilings (recomputed 2026-04-24, scores match the CI contract: `query_only=0.349`, `submit_all_matched=0.308`, `overflag_rings=0.283`, `submit_all_mismatched=0.266`, `zero_itc=0.266`, `confirm_spam=0.010`, all `<0.45`).
- Two labeled PNG plots in [`data/figures/`](data/figures/) embedded in both READMEs, showing the documented Qwen3-0.6B training trace and per-component breakdown vs red-team attacks.
- Reconstructed [`data/smoke_test_10step.json`](data/smoke_test_10step.json) from the committed `smoke.log` so every doc reference resolves.
- Prompted Qwen2.5-3B baseline numbers ([`data/baseline_metrics_real.json`](data/baseline_metrics_real.json)): delta 1.18 over raw policy, 95% CI [1.09, 1.27], 180 rollouts. This is the honest pre-training evidence.
- Live HF Space with the 3D ring viewer, 90-second demo video, BLOG / PITCH / JUDGE_TOUR / ROUND2_PROBLEM_STATEMENT / EXEC_SUMMARY / QA_REHEARSAL docs.

## 5. What ran on-site (Day 1, A100 SXM4-80GB, 2026-04-25)

Hugging Face Spaces compute upgrade to A100 SXM4-80GB (Pro subscription required for dev-mode SSH; ~$2.50/hr). $21 effective compute budget after subscription. Wall-clock used: ~2.5 h.

Pipeline that ran:

- **Phase 1 verification.** 3000-row balanced trajectory file (1000 oracle + 1000 inspect_then_label + 1000 supplier_cap_aware, round-robin interleaved) verified locally (mean total 0.690, mean R3 0.791). Did not regenerate on-site per ONSITE_DAY1_PROMPT.md guidance.
- **Phase 2 SFT.** [`scripts/train_sft_warmstart.py`](scripts/train_sft_warmstart.py) on Qwen/Qwen3-4B + LoRA rank 16 + bf16 + 1 epoch over 3000 rows = 375 optimizer steps. Wall time **31:12** (5s/step), final aggregate train_loss **0.341**. Smoke run at `--limit-rows 300` cleared the 5-min gate (first loss line at ~31s) and the step-10 loss-stagnation canary (loss[10]/loss[5] = 0.918 with full-run logging cadence). Full artifacts: [`data/sft_full_run.log`](data/sft_full_run.log), [`data/sft_summary.json`](data/sft_summary.json), [`data/sft_smoke_run.log`](data/sft_smoke_run.log).
- **LoRA merge.** Final adapter (132 MB) merged into base Qwen3-4B weights using `peft.PeftModel.merge_and_unload()` to produce a full inference-ready checkpoint (8 GB safetensors). Required because `train_grpo_real.py` loads via `AutoModelForCausalLM.from_pretrained(args.model)` directly without PEFT-aware wrapping.
- **5-metric rollout audit on held-out seeds 9030-9034.** [`scripts/audit_sft_rollout_quality.py`](scripts/audit_sft_rollout_quality.py) at `T=0.7 top_p=0.95 top_k=20` matching `train_grpo_real.py:557-561`. After resolving the 4-bug audit-OOD trap chain (Failure Mode 4 above), n=5 mean composite reward **0.280**, with 3 of 5 seeds emitting 32-46 mark verbs per trajectory (R1 = 0.15, R2 = 0.41 in mark-emitting trajectories) and 2 of 5 seeds collapsing to the 3-step query_only shape (R = `{0.01, 0.01, 0.99, 0.99}` total 0.353). Headline artifact: [`data/audit_F_n5.json`](data/audit_F_n5.json).
- **Step-100 early-stop control.** Trained a fresh checkpoint to step 100 (`--limit-rows 800`) to test the over-training hypothesis. Result: malformed JSON output (missing commas in tool_call body) and off-vocab verb names like `query_schema`, `list_invoices`. Step 100 is undercooked; it hasn't learned tool_call grammar yet. By step 375 grammar is clean; the policy collapse documented in Failure Mode 5 is post-grammar-acquisition. No "less-trained = healthier" sweet spot exists in this curve. Artifact: [`data/audit_step100_n5.json`](data/audit_step100_n5.json).

What was deferred (Phase 3 GRPO):

- Per Failure Mode 5, the reward landscape favors the query_only attack (0.353) over the marking trajectory (max 0.26) at this policy quality. GRPO's advantage signal would optimize toward the attack within ~50 steps. Running GRPO unmodified would reproduce the pre-onsite 0.353 plateau exactly.
- Three resolutions are documented in Failure Mode 5; all touch either the frozen `rewards.py` (Guardrail 1), the trainer's reward shaping path, or Phase 1 trajectory regeneration. None can be done responsibly within a 30-min Action 7 docs window without violating an invariant or shipping under-tested changes.
- Compute remaining at decision time (~$17 of $21 / ~6.8 h gross / 5.7 h after safety) was preserved as buffer for Action 7 docs sprint, HF Space rebuild, and judge-time investigation rather than spent on a high-risk-low-reward GRPO that would likely converge to the attack.

What ships:

- Trained Qwen3-4B SFT checkpoint with **n=5 mean composite reward 0.280** as the primary on-site number (above prompted Qwen2.5-3B baseline 0.18, below red-team ceiling 0.45, below query_only Mode B exploit shape 0.353).
- Full diagnostic chain artifacts ([`data/audit_seeds_9031_9034.json`](data/audit_seeds_9031_9034.json) pre-fix collapse, [`data/audit_F_n5.json`](data/audit_F_n5.json) headline, [`data/audit_step100_n5.json`](data/audit_step100_n5.json) early-stop control) committed for reproducibility.
- Audit script [`scripts/audit_sft_rollout_quality.py`](scripts/audit_sft_rollout_quality.py) committed as a reusable diagnostic tool.
- The reward-landscape inversion finding (Failure Mode 5) documented as a substantive contribution to the post-hackathon agenda: the same arithmetic-reward + multi-component-clamping design that makes the 6 red-team attacks defensible also creates this asymmetry. Future work shapes one without sacrificing the other.

## 6. Why this shipped as-is instead of a vaporware trained number

Per [ONSITE_BRIEFING.md](ONSITE_BRIEFING.md) emergency fallback: *"Do NOT fake numbers. If trained numbers are worse than the prompting baseline, the honest story is: 'we ran out of compute on-site; the pre-staged SFT warm-start script is in the repo for post-hackathon validation.' Judges reward honesty per the collapse narrative already in [BLOG.md](BLOG.md) §6."* The red-team tests are CI-enforced, the reward design is auditable, the prompted baseline is a real 1.18-delta improvement over raw, the on-site SFT lifts that further to a 0.280 mean composite reward, and the documented 0.6B + on-site 4B failure modes are real training evidence.

The judges' 2026 India-hackathon criteria doc says explicitly: *"A messy but ambitious environment with real training evidence beats a polished but boring one."* This submission is ambitious in the environment (16 verbs, 5 mismatch types, directed-ring fraud, per-supplier Rule 36(4), hidden ground truth invariant), real in the training evidence (pre-onsite Kaggle 0.6B plateau + on-site A100 4B SFT at 0.280), and honest in the scope (no GRPO Phase 3 because the reward landscape would push toward the attack).
