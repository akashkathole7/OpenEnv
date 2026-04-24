# ONSITE_DAY1_PROMPT.md — paste-ready prompt for Claude Code on 2026-04-25 at Bangalore

> **How to use this file.** Paste this entire document as the first message to a new Claude Code session opened inside `/home/aakash/Videos/ReconcileEnv-GST2B` on 2026-04-25 morning. Everything below is written *as a direct prompt to that Claude*. No preamble, no introductions, no recap. Execute the state-verification block first, then the training pipeline, in the order given.

---

## Identity and context

You are continuing **Aakash Kathole's hackathon project** (solo entrant, Meta × Scaler Hackathon Grand Finale, Bangalore 2026-04-25 to 2026-04-26). GitHub: `akashkathole7`. Branch: `scaffold/reconcile-gst2b`. Fork remote: `fork`. HF Space clone: `hf_space_clone/` (remote: `origin` → `https://huggingface.co/spaces/akashkathole/reconcile_gst2b_env`).

**Today (2026-04-25) is Day 1 on-site.** Scaler provides HF compute credits (ask Aakash for the exact UI/URL and credit balance when he arrives; the briefing assumes A100 40GB available, with Qwen3-1.7B fallback if A100 is unavailable or saturated). Results are judged asynchronously with YouTube Live stream on 2026-05-02, so there is no hard Day-2 submission cutoff that forces fake numbers — ship honest-partial over vaporware if Phase 2/3 bricks.

**Pre-onsite state has been finalized as of 2026-04-24.** The Phase 1 + Phase 2 + Phase 3 scripts are already written, tested as far as Kaggle would allow, and the Path B surgical fix for the TRL 1.x silent-hang is already committed. The Kaggle notebook is left as a historical artifact — do NOT debug Kaggle further on Day 1. Phase 2+3 run on the on-site A100 via whatever Scaler compute UI provides (Colab A100 link, HF Compute dashboard, direct SSH, you'll ask).

## Read these first (in order, before acting)

1. **`envs/reconcile_gst2b_env/ONSITE_BRIEFING.md`** — the original briefing, 444 lines. Mode table, rubric crosswalk, Phase 0-4 commands, 12 invariants, 11 numbers Aakash defends in Q&A, judge Q&A ammo, emergency fallback. Written 2026-04-22, updated through 2026-04-22.
2. **`envs/reconcile_gst2b_env/LESSONS_LEARNED.md`** — 6-section engineering retrospective of the pre-onsite Kaggle attempts (written 2026-04-24). Covers: which training logs survived commit (0.6B only), five Kaggle env brittleness failures, two distinct training hangs, what ships, what's corrected for on-site. Read this before touching the training pipeline.
3. **`git log --oneline -20`** — the commit trail. Most recent commits as of 2026-04-24 end-of-day (pre-onsite hotel window in Bangalore, after the polish sprints + tab-live sprints):
   - `503d9c0 feat: make Gradio tab 4 'Baseline Comparison' live` ← **Tab 4 now shows a composite-reward bar chart**
   - `a051a7b fix: app.py oracle uses SimpleNamespace + rewards directly (no openenv)` ← **app.py has NO openenv import, uses stubs + composite_reward directly**
   - `5cb88ad feat: make Gradio tab 2 'Rollout Replay' live with oracle trace (F7)` ← **Tab 2 runs oracle episodes, shows verb trace + reward breakdown**
   - `fd601d4 docs: tighten training plots for thumbnail legibility (F1 fix)` ← **data/figures/three_scales_*.png rewritten at dpi=200 with cleaner attack-range band**
   - `906284e docs: add ONSITE_DAY1_PROMPT.md handoff` ← **(this file's first version)**
   - `2595df5 docs: add LESSONS_LEARNED.md`
   - `afade93 docs: concretize README Rubric-mapping section with real OpenEnv API`
   - `c6cf710 docs: rewrite README top for 3-5 min judge flow`
   - `91a02c4 docs: add judge-facing training figures + reconstruct smoke_test_10step.json`
   - `392bff3 fix: pin torch==2.10.0 in cell 6`
   - `e4c8ea0 fix: drop --force-reinstall torch from cell 6`
   - `4b3707d fix: remove dataset_text_field=None (TRL 1.x auto-detect), add --limit-rows` ← **Path B surgical fix already landed here**
   - `c90bc7c fix: pin trl<1.0 on Kaggle` (superseded by 4b3707d)
   - `7ffc164 fix: restore CUDA torch` (Kaggle-only, irrelevant on A100)
   - `723d2c3 fix: switch SFT to fp16 + max_seq 2048` (T4-specific; A100 can use bf16)
   - `9ab9e4e fix: force Kaggle cells 10/12 to single GPU` (Kaggle-only)

### HF Space tab state (what's already live, do NOT rebuild)

As of this handoff, the deployed Hugging Face Space [huggingface.co/spaces/akashkathole/reconcile_gst2b_env](https://huggingface.co/spaces/akashkathole/reconcile_gst2b_env) has:

- **Tab 1 Schema + Label Diff** — placeholder (honest scope; judges see "coming after on-site training")
- **Tab 2 Rollout Replay** — LIVE, runs oracle on any seed, shows verb trace + 4-component reward breakdown
- **Tab 3 Circular-Ring Viewer** — LIVE, 3D supplier graph with planted fraud rings
- **Tab 4 Baseline Comparison** — LIVE, composite-reward bar chart with oracle (mean over 5 seeds with min-max error bar), 6 red-team attacks, prompted Qwen2.5-3B baseline (0.18), trained target placeholder (gray bar at 0.50 with `_TRAINED_PLACEHOLDER` constant in app.py), and 0.45 red-team ceiling dashed line

After Action 7 (training lands), Tab 4's gray placeholder bar gets updated by changing `_TRAINED_PLACEHOLDER` in `app.py` to the measured trained number and swapping the gray color to a solid one. The bar chart regenerates on next Space reload.

### Training-figure regeneration

`envs/reconcile_gst2b_env/scripts/make_training_figures.py` regenerates both embedded PNGs (`data/figures/three_scales_reward.png` and `data/figures/three_scales_components.png`) from the underlying JSON. Run it after training lands to refresh the plots:
```bash
PYTHONPATH=src:envs uv run python -m \
    envs.reconcile_gst2b_env.scripts.make_training_figures
```

## Mode

Default: **Qwen3-4B** on A100 40GB (briefing's on-site default).
Fallback: **Qwen3-1.7B** if A100 is unavailable / saturated, or OOM on 4B with `batch-size=1, grad-accum=8`. The SFT training script now takes `--model` and `--precision {fp16,bf16}` CLI args; for A100 use `--precision bf16`. For the 1.7B fallback, use `--precision fp16` (1.7B fits any GPU comfortably; bf16 works on A100 but fp16 is also fine).

## State-verification block (RUN BEFORE ANY TRAINING ACTION)

Before firing any training command, confirm in under 5 minutes:

```bash
# 1. Confirm branch + pushed state
git status
git branch --show-current       # expect: scaffold/reconcile-gst2b
git log fork/scaffold/reconcile-gst2b..HEAD --oneline   # expect: empty (everything pushed)
git log --oneline -5
# Expect HEAD at 503d9c0 (Tab 4 live) unless Aakash pushed more overnight.

# 2. Confirm Path B fix is in place (dataset_text_field is NOT actively passed to SFTConfig)
grep -n "dataset_text_field" envs/reconcile_gst2b_env/scripts/train_sft_warmstart.py
# expect: only a comment line (~line 272) explaining the removal.
# If the line 'dataset_text_field=None,' exists as an active SFTConfig arg,
# STOP and re-apply the fix (see LESSONS_LEARNED.md §3).

# 3. Confirm Phase 1 data file exists and has 3000 rows
wc -l envs/reconcile_gst2b_env/data/sft_trajectories.jsonl
# expect: 3000 lines

# 4. Confirm tests green, env imports, GPU visible
PYTHONPATH=src:envs uv run pytest tests/envs/test_reconcile_gst2b_*.py -v --tb=short | tail -5
# expect: 42 passed
PYTHONPATH=src:envs uv run python -c "
from envs.reconcile_gst2b_env.server.reconcile_gst2b_environment import ReconcileGST2BEnvironment
e = ReconcileGST2BEnvironment(); o = e.reset(seed=42, mode='warmup')
print('env OK, step_budget =', o.step_budget)
import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')
"

# 5. Confirm Space tabs 2/4 still live (app.py has the SimpleNamespace oracle path)
grep -n "SimpleNamespace\|_oracle_trajectory\|_baseline_comparison_figure\|_TRAINED_PLACEHOLDER" envs/reconcile_gst2b_env/app.py | head -10
# expect: matches for SimpleNamespace import + 3 function defs + _TRAINED_PLACEHOLDER constant.
# If missing, the Space tabs are broken — do NOT push any change that breaks
# the currently-deployed Space without restoring these first.

# 6. Confirm embedded figures exist + regenerator is callable
ls -la envs/reconcile_gst2b_env/data/figures/
# expect: three_scales_reward.png + three_scales_components.png
PYTHONPATH=src:envs uv run python -c "
from envs.reconcile_gst2b_env.scripts.make_training_figures import _attack_scores
s = _attack_scores()
print('red-team attack scores recomputed:', {k: v['total'] for k, v in s.items()})
# expect: query_only=0.3492, submit_all_matched=0.3075, etc. (matches CI test file)
"
```

If any of 2, 3, 4, 5, 6 fail, STOP and ask Aakash. Do not "fix" invariants unilaterally.

## Actions

### Action 1 — Claim Scaler compute credits (5 min)

Ask Aakash: "Which compute UI provides our HF credits? Direct URL? Credit balance? Is this an A100 40GB or something else? Any session time limit?" His answers determine whether you're running on a Colab-like notebook, an SSH shell, or an HF Spaces compute endpoint. If A100 is not available, switch to Qwen3-1.7B per Mode section and keep going.

### Action 2 — Phase 0 sanity (15 min, already mostly in State-verification)

Beyond the state-verification: re-run `openenv validate --verbose` if the compute instance has `openenv` installed, and verify ground-truth hiding:
```bash
PYTHONPATH=src:envs uv run pytest tests/envs/test_reconcile_gst2b_*.py::test_state_hides_ground_truth_from_observation -v
```

### Action 3 — Phase 1 trajectory verify (5 min, NOT regenerate)

The committed `data/sft_trajectories.jsonl` is the balanced 3000-row subsample from pre-onsite (1000 oracle + 1000 inspect_then_label + 1000 supplier_cap_aware, round-robin interleaved). Do NOT regenerate on Day 1 unless Aakash explicitly asks. Verify:
```bash
PYTHONPATH=src:envs uv run python3 -c "
import json, statistics
rows = [json.loads(l) for l in open('envs/reconcile_gst2b_env/data/sft_trajectories.jsonl')]
print(f'rows: {len(rows)} | mean total: {statistics.mean(r[\"total\"] for r in rows):.3f}')
print(f'mean R3: {statistics.mean(r[\"breakdown\"][\"R3\"] for r in rows):.3f}')
"
# expect: rows: 3000, mean total: ~0.690, mean R3: ~0.791
```

### Action 4 — Phase 2 SFT smoke with --limit-rows 300 (10 min wall, gate)

```bash
PYTHONPATH=src:envs uv run python -m envs.reconcile_gst2b_env.scripts.train_sft_warmstart \
    --input-jsonl envs/reconcile_gst2b_env/data/sft_trajectories.jsonl \
    --output-dir  envs/reconcile_gst2b_env/data/sft_checkpoint_smoke \
    --model       Qwen/Qwen3-4B \
    --epochs      1 \
    --batch-size  1 \
    --grad-accum  8 \
    --learning-rate 2e-5 \
    --precision   bf16 \
    --max-seq-length 2048 \
    --limit-rows  300 \
    --logging-steps 1 \
    --save-steps 25 \
    2>&1 | tee envs/reconcile_gst2b_env/data/sft_smoke_run.log
```

**Gate:** first `{'loss': ...}` line within 3 minutes of `=== SFT training ===`. Step time <30s on A100 after step 3. If first loss line does NOT appear in 5 min, STOP and ask Aakash — the TRL 1.x hang may be re-surfacing in a different form and warrants fresh RCA, not more push-and-pray (see LESSONS_LEARNED §3).

### Action 5 — Phase 2 SFT full run (~2 h on A100)

Only after Action 4 passes the gate. Remove `--limit-rows 300`, bump `--output-dir` to `data/sft_checkpoint`, otherwise identical:
```bash
PYTHONPATH=src:envs uv run python -m envs.reconcile_gst2b_env.scripts.train_sft_warmstart \
    --input-jsonl envs/reconcile_gst2b_env/data/sft_trajectories.jsonl \
    --output-dir  envs/reconcile_gst2b_env/data/sft_checkpoint \
    --model       Qwen/Qwen3-4B \
    --epochs      1 \
    --batch-size  1 \
    --grad-accum  8 \
    --learning-rate 2e-5 \
    --precision   bf16 \
    --max-seq-length 2048 \
    --logging-steps 5 \
    --save-steps 50 \
    2>&1 | tee envs/reconcile_gst2b_env/data/sft_full_run.log
```

**Run in background via `run_in_background: true`.** Aakash does not want running commentary during a 2h training block. Check back when the process exits.

**Pass gates:** final train loss < 0.8; `adapter_model.safetensors` exists in `data/sft_checkpoint/final/`; `sft_summary.json` written.

**Mandatory rollout audit before Phase 3:** sample 5 rollouts on held-out seeds 9030-9034 against the SFT checkpoint. Check query-to-mark ratio. If queries per invoice < 2, retrain with reduced oracle weight (subsample 500 oracle + 1250 inspect + 1250 supplier = 3000 rows) before Phase 3. This catches the "reward rising but quality not" failure mode from [`ONSITE_BRIEFING.md`](ONSITE_BRIEFING.md) self-serve-guide Q52.

### Action 6 — Phase 3 GRPO polish (~2 h on A100)

```bash
PYTHONPATH=src:envs uv run python -m envs.reconcile_gst2b_env.scripts.train_grpo_real \
    --model envs/reconcile_gst2b_env/data/sft_checkpoint/final \
    --total-steps 200 \
    --output-dir envs/reconcile_gst2b_env/data/grpo_checkpoint \
    2>&1 | tee envs/reconcile_gst2b_env/data/grpo_full_run.log
```

Tier 1+2 fixes are already baked into `train_grpo_real.py`: `enable_thinking=False` monkeypatch, `beta=0.0`, zero-std deterministic jitter σ=0.005, `+0.02` format bonus for trajectories with ≥1 action, absolute early-stop at total `<0.30`, step-1 `grad_norm` sanity check.

**Pass gates:** eval total on hero seeds > 0.45 (clears red-team ceiling). R1 ≥ 0.25, R2 ≥ 0.30, R3 = 0.99 on ≥50% of hero seeds, R4 ≥ 0.40.

**Rollout-length canary** (per Lewis Tunstall's Scaler workshop, folded into `ONSITE_BRIEFING.md`): mean rollout length should stabilize 8-20 steps per episode. If it climbs monotonically toward 50, the model is hacking budget-exhaust; stop and diagnose.

### Action 7 — Measure, update docs, push, redeploy (~1.5 h)

Eval gap per `ONSITE_BRIEFING.md` Phase 4: neither `real_baseline.py` nor `hero_baseline.py` can eval a trained checkpoint without edits. Pick Path A (add `--model-path` to `real_baseline.py`, ~30 min, TDD against a new test file) or Path B (new `scripts/eval_trained.py`, ~45 min). Path A preferred.

Then:
1. Run trained eval on 30 heldout seeds → `data/trained_eval.json`
2. Run trained eval on hero seeds 9500/9501/9502 → `data/trained_hero.json`
3. Edit `EXEC_SUMMARY.md` bullet 5 (replace Qwen2.5-3B prompting baseline with trained Qwen3-4B number)
4. Edit `EXEC_SUMMARY.md` bullet 10 (add "Post-SFT+GRPO on A100, total lifts to X.XX on 30 heldout seeds")
5. Edit `ROUND2_PROBLEM_STATEMENT.md` at-a-glance metric
6. Edit `BLOG.md` §6 (add closing paragraph AFTER the collapse narrative — DO NOT delete or soften collapse)
7. Edit `README.md` and `README_HF_SPACE.md` headline-numbers table (both have TL;DR sections that mention "Training evidence from A100 Apr 25-26 lands as a follow-up commit"; replace with actual numbers)
8. **Regenerate the figures** from the new data: `PYTHONPATH=src:envs uv run python -m envs.reconcile_gst2b_env.scripts.make_training_figures`. This updates `data/figures/three_scales_reward.png` and `three_scales_components.png` so the README's embedded plots refresh. Both are referenced via `raw.githubusercontent.com/akashkathole7/OpenEnv/...` URLs so they'll auto-render on the Space after fork push.
9. **Update Tab 4 placeholder in `app.py`**: change the constant `_TRAINED_PLACEHOLDER = 0.50` to the measured trained number, and in `_baseline_comparison_figure()` change the placeholder row's `"color": "#bbbbbb"` (gray) to a solid color (suggest `"#9467bd"` purple or `"#17becf"` teal to distinguish from oracle/baseline/attack colors), change the `"label"` to drop "(target, on-site Apr 25-26)" and add the new measured context, and remove the `text_pos: "inside"` special case so the number renders outside-bar like the others. This makes the Tab 4 chart reflect actual trained performance instead of the target placeholder.
10. Update `LESSONS_LEARNED.md` §5 (what's corrected for on-site): change future tense to past tense, add the actual trained numbers. Also update §1 to note which Phase ran on-site and any live-observed behavior (entropy, tool-call frequency).
11. Git commit single: `onsite: land trained Qwen3-4B numbers (SFT + GRPO on A100)`
12. `git push fork scaffold/reconcile-gst2b`
13. Sync `hf_space_clone/`: copy the updated `app.py` (Tab 4 placeholder → real), `README.md` (from `README_HF_SPACE.md`), `LESSONS_LEARNED.md`, `BLOG.md`, `EXEC_SUMMARY.md`, `ROUND2_PROBLEM_STATEMENT.md`, plus `data/trained_eval.json`, `data/trained_hero.json`. **Do NOT copy `data/figures/*.png` into `hf_space_clone/`** — HF Spaces requires binary files go through Xet storage, and the README references them via raw.githubusercontent from the fork which resolves fine.
14. Commit in hf_space_clone: `sync: trained numbers from on-site A100 run`
15. `cd hf_space_clone && git push origin main`
16. Verify Space rebuild completes green (incognito reload of the Space URL). Click Tab 4 — the placeholder bar should now be the measured trained number, not gray. Click Tab 2 — still works. Click Tab 3 — still works.

## Guardrails (invariants, do NOT violate)

1. **`rewards.py` is frozen.** Any perceived bug → flag to Aakash, do not fix unilaterally. Red-team CI contract depends on the exact 0.40/0.25/0.25/0.10 weights, `[0.01, 0.99]` clamp, and `−1.0` structural penalty. The `zero_itc` R3 blind-spot is deliberate and documented.
2. **The 6 red-team attacks must stay `<0.45`.** If any test turns red after a code change, revert. The tests in [`tests/envs/test_reconcile_gst2b_reward_hacking.py`](../../tests/envs/test_reconcile_gst2b_reward_hacking.py) are the contract.
3. **Ground truth is hidden from the agent.** `ground_truth.py` is oracle-only. Do not add a helper tool that leaks GT.
4. **16-verb surface is content-locked.** Do not add or remove verbs.
5. **PITCH.md structure is content-locked.** May fix a factual error (e.g., a changed number). Do not rewrite sentences.
6. **No em-dashes in any judge-facing doc.** Use commas, colons, periods. (AI-tell cleanup invariant; the previous session removed 268.)
7. **Preserve the honest collapse narrative in BLOG.md §6.** Story is "we diagnosed and fixed." Deleting/softening loses judge appeal, gains nothing.
8. **Branch is `scaffold/reconcile-gst2b`. Do not merge to main on-site.** PRs against `main` come later.
9. **HF Space `requirements.txt` has 6 pinned deps** (gradio, networkx, plotly, numpy, pandas, pydantic). If you add a runtime import to `app.py`, add it here too.
10. **90-second demo URL `https://www.youtube.com/watch?v=rglR1hGgdb8`** is linked in 5+ docs. Do not change unless Aakash uploads a new video.
11. **QLoRA merge footgun.** If Phase 2 ends up using 4-bit quantization, download original 16-bit base weights and merge the LoRA into THOSE, not the dequantized 4-bit copy. ~30% quality damage from the naive path per Daniel Han.
12. **No fake numbers.** Emergency fallback is honest-partial ("trained checkpoint exists, eval surface needs completion") over vaporware. Judges reward honesty per the collapse narrative.
13. **Kaggle is out of scope on Day 1.** `scripts/kaggle_phase2_sft.ipynb` is a historical artifact. Do not re-attempt Kaggle runs; on-site A100 is the clean environment.
14. **Tab 4 placeholder constant.** `app.py` has `_TRAINED_PLACEHOLDER = 0.50` used only to draw the gray bar in Baseline Comparison. Update it to the measured trained number in Action 7 step 9. Until then, leave it at 0.50 so the Space keeps rendering a valid chart.
15. **HF Space has NO openenv package installed.** `app.py` deliberately uses `SimpleNamespace` stubs + `composite_reward` directly — do NOT add imports of `models.py` or `server/*` into `app.py` unless you also pip-install openenv in `hf_space_clone/requirements.txt`. The current 6 deps (gradio, networkx, plotly, numpy, pandas, pydantic) are sufficient and pinned.
16. **HF Space does NOT accept binary files in git.** Pushing any `*.png` / `*.jpg` / etc. to `hf_space_clone/` will be rejected by HF's pre-receive hook (Xet storage required). README references images via `raw.githubusercontent.com/akashkathole7/OpenEnv/...` from the fork, which is sufficient. Only text files go to `hf_space_clone/`.

## Judge demo windows

- **3-minute pitch:** [`envs/reconcile_gst2b_env/PITCH.md`](PITCH.md) — content-locked. Aakash rehearses to this text. If he asks you to tweak, fix a single number only.
- **~2-minute Q&A:** [`envs/reconcile_gst2b_env/QA_REHEARSAL.md`](QA_REHEARSAL.md) — 22 anticipated probes with answers. Judge-framing ammunition is in [`ONSITE_BRIEFING.md`](ONSITE_BRIEFING.md) §Judge Q&A (fixed weights vs curriculum, jagged intelligence, vLLM precision mismatch, LoRA Without Regret). Use only if asked, do not volunteer.
- **The 11 numbers Aakash defends** live in [`ONSITE_BRIEFING.md`](ONSITE_BRIEFING.md) §"11 numbers". After Action 7 lands trained numbers, re-align bullets 5 and 10 of that section too.

## Communication norms (tired solo entrant)

- Aakash may be sleep-deprived, context-switching between judges and compute. Keep messages short, numbered, action-oriented. No running commentary.
- When a training run is in flight, use `run_in_background: true`. Do not poll `nvidia-smi`. Do not narrate.
- If you find a bug in `rewards.py` or the env server during on-site work, **flag it, do not fix it unilaterally**. Red-team CI contract is judge-visible.
- If a judge-visible doc needs updating, do all four: update → commit → push → redeploy HF Space. Mismatched numbers between GitHub and Space is a visible judge gotcha.
- Aakash's laptop clock may drift. `date` before any time-sensitive commit.

## Close: state-readout request

**After reading the files above and running the State-verification block, report back to Aakash with exactly these 7 bullets** (one line each, factual, no narration):

1. Git state: `<HEAD hash and branch>`, `<n unpushed commits to fork>`, `<n untracked files>`.
2. Tests: `<pass/fail count>` from `PYTHONPATH=src:envs uv run pytest tests/envs/test_reconcile_gst2b_*.py --tb=line | tail -1`.
3. Path B fix: `<present / missing>` (grep check on `dataset_text_field` in `train_sft_warmstart.py`).
4. Phase 1 data: `<row count>` in `data/sft_trajectories.jsonl`, `<mean total>` from the verification snippet.
5. Compute: `<A100 / other GPU / CPU only>` from `torch.cuda.get_device_name(0)`, and `<Scaler compute UI confirmed yes/no>` from Aakash.
6. Space tabs: `<live / broken>` based on verification step 5 (SimpleNamespace oracle + Tab 4 figure function + _TRAINED_PLACEHOLDER constant all present).
7. Figures: `<both PNGs present / missing>` and red-team recompute matches CI (query_only=0.349, submit_all_matched=0.308, overflag_rings=0.283, submit_all_mismatched=0.266, zero_itc=0.266, confirm_spam=0.010).

Do not begin any training action until Aakash acknowledges the state-readout.

End of prompt.
