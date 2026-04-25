# Pitch rehearsal doc, Meta × Scaler Finale, April 25–26 2026

Local-only (gitignore explicitly, do not commit). 3-minute pitch + 2-minute Q&A.
Target pace: 150 words per minute. 3 min = 450 words max.

---

## P1, 3-minute pitch script

Two versions depending on whether 3B + A100 training happens by April 24.
Pick ONE on demo day. Do NOT try to switch versions mid-pitch.

### Version A, ship with real training curve (if A100 run lands)

> 440 words. Time this at 2:55 in your 3rd dry run.

**[0:00 – 0:30, problem, 75 words]**

India has 14 million GST-registered businesses. Every month, each one matches
its purchase register, invoices it claims it bought, against GSTR-2B, a
regulator-generated document of what its suppliers claimed they sold to it.
Where they disagree, the business either over-claims tax credit and gets an
audit notice, or under-claims and forfeits money. Today, this is done by hand,
by CA-firm juniors, across millions of invoices. It's rule-heavy, error-
sensitive, and boring. An excellent target for RL.

**[0:30 – 1:30, environment, 150 words]**

`reconcile_gst2b_env` is an OpenEnv environment for this exact workflow. Five
planted mismatch types, GSTIN typos, invoice-number drift, tax-slab off-by-
one, supplier late filings, post-freeze amendments, plus thirty-percent-
probability directed three-cycle circular-trading rings, detectable by
networkx simple-cycles. The agent works through sixteen typed tool verbs,
seven query and seven mutate, over fifty-step episodes. Reward is four
arithmetic components, each clamped to zero-point-zero-one to zero-point-
nine-nine: macro-F1 on labels, ITC delta accuracy, Rule 36(4) per-supplier
compliance, and quadratic step efficiency. No LLM judge anywhere in the
reward path, boring on purpose, tamper-resistant by design. Six red-team
attacks CI-enforced under 0.45; the hardest, `query_only`, scores 0.349.
Hidden ground truth is a first-class invariant, unit-tested to never leak
into any observation.

**[1:30 – 2:30, training evidence, 150 words]**

On training: prompted Qwen2.5-3B on Kaggle T4 lifts total reward from minus-one
to 0.18, delta 1.18, ninety-five percent bootstrap CI from 1.09 to 1.27.
That's prompt engineering. The real GRPO run, four hours of A100 on Qwen2.5-
3B, lifts it further to __[insert final total]__, here's the curve.
**[point at slide 2]** R1 moves from 0.01 to __[value]__, R2 from 0.01 to
__[value]__, R3 holds at 0.99, R4 at 0.99. Red-team battery still holds -
all six attacks unchanged, all under 0.45. The LoRA adapter is thirty
megabytes, drops into any OpenEnv-compatible inference stack. Commit
3ca024d documents the Tier 1+2 training fixes that unblocked this -
enable-thinking off, KL-tether zero, zero-std group jitter, format bonus.
`rewards.py` untouched throughout.

**[2:30 – 3:00, differentiation + compute ask, 65 words]**

What's distinct from Gaia2 or τ-bench: this isn't open-ended web tool use.
The regulator publishes the rules. Correctness is crisp. Over-claim has a
measurable penalty. This is adversarial domain reasoning against hidden
ground truth with a CI-enforced red-team ceiling. Environment ships, training
ships, the LoRA ships. Next ask: a seven-B model on eighty-gigabyte A100 to
cover the RCM and ISD scope fences.

---

### Version B, ship without real training curve (current state, April 22)

> 445 words. This is the honest engineering-diagnosis version.
> Use this if 3B + A100 training doesn't land by April 24 noon.

**[0:00 – 0:30, problem, 75 words]**

India has 14 million GST-registered businesses. Every month, each one matches
its purchase register, invoices it claims it bought, against GSTR-2B, a
regulator-generated document of what its suppliers claimed they sold. Where
they disagree, the business either over-claims tax credit and gets an audit
notice, or under-claims and forfeits money. Today this is done by hand, by
CA-firm juniors, across millions of invoices. It's rule-heavy, error-
sensitive, and boring. An excellent target for RL.

**[0:30 – 1:30, environment, ~120 words]**

`reconcile_gst2b_env` is an OpenEnv environment for this workflow. Five
planted mismatch types, GSTIN typos, invoice-number drift, tax-slab off-
by-one, supplier late filings, post-freeze amendments, plus thirty-percent-
probability directed three-cycle circular-trading rings, detected via
`networkx.simple_cycles`. The agent works through sixteen typed tool verbs
over fifty-step episodes. Reward is four arithmetic components, each clamped:
macro-F1 on labels, ITC delta accuracy, Rule 36(4) per-supplier compliance,
step efficiency. No LLM judge anywhere in the reward path, boring on purpose,
tamper-resistant by design. Six red-team attacks CI-enforced under 0.45; max
`query_only` at 0.349. Hidden ground truth is a first-class invariant, unit-
tested to never leak into any observation.

**[1:30 – 2:30, training evidence, ~150 words]**

Prompted Qwen2.5-3B baseline scores zero-point-one-eight versus raw minus-one
- delta 1.18, CI 1.09 to 1.27, excludes zero. Reward surface has gradient.
Then three GRPO training attempts across Qwen3 sizes, three distinct failure
modes, and that's the actual finding. **0.6B**: bimodal 0.25 tool-call rate,
can't reliably chain five mark actions, plateaus at zero-point-three-five-three
- the `query_only` red-team signature. **1.7B**: tool-call rate collapses to
zero across fifteen steps, entropy down to zero-point-one-two from 0.6B's
zero-point-four-four, deterministic mode collapse. **4B**: opposite problem -
emits parseable tool-call JSON every turn, but never commits to a label,
over-queries to step-budget exhaustion, crashes R4 to floor at zero-point-
two-five-five. Three scales, three different reward components pinned to
floor. Not a recipe bug, not a bigger-model fix, the env is surfacing
three structural walls, each caught by a different reward component.

**[2:30 – 3:00, differentiation + compute ask, 70 words]**

What's distinct from Gaia2 or τ-bench: not open-ended web tool use, the
regulator publishes the rules, correctness is crisp, over-claim has a
measurable penalty. Adversarial domain reasoning against hidden ground truth.
Environment ships, diagnosis ships, smoke evidence ships. The coin-flip-
raised-to-five tool-call chaining problem at point-six-B scale is a compute
bound, not a recipe bound. Next ask: Qwen2.5-3B on A100 forty-gig for four
hours.

---

## P2, 60-second Gradio demo walkthrough

> Live Space: https://huggingface.co/spaces/akashkathole/reconcile_gst2b_env
> **Only Tab 3 (Circular-Ring Viewer) is functional**, Tabs 1/2/4 are
> placeholders per `app.py` lines 10–13. Do NOT attempt a step-by-step
> rollout demo live; there is no UI for tool-verb invocation. The Ring
> Viewer is visually striking and one-click, which is the right shape
> for a 60-second slot anyway.

Four interactions, 60 seconds. Practice once on your laptop's wifi to
confirm the 3D plot renders under 3 seconds.

| T+ | Action | Narrate |
|----|--------|---------|
| 0:00 | Open Space, click Tab 3 "Circular-Ring Viewer" | "Live on HuggingFace Spaces. This tab visualizes the circular-trading ring detector." |
| 0:05 | Seed dropdown → pick 9500 | "Hero seed 9500, one of three with fixed planted rings for demo reproducibility." |
| 0:15 | Click "Render" | "This is the supplier graph for one episode. Every edge is an invoice relationship. Red nodes are GSTINs the env's ring detector flagged via `networkx.simple_cycles`, a directed 3-cycle A→B→C→A, planted with 30% probability per episode." |
| 0:30 | Rotate the 3D plot with mouse | "The agent sees the same invoices a human would, but has to detect this ring **without** the highlight. Ground truth is hidden from every observation by a unit-tested invariant, the agent never sees a `true_in_circular_ring` field." |
| 0:45 | Seed dropdown → switch to 9501, click Render | "Different seed, different supplier graph, different ring. Deterministic in the seed, you can re-run and get bit-identical episodes, verified by `make reproduce`." |
| 0:58 | Close plot, pivot to slides | "Reward breakdown and attack scores are on slide 1, let's go there." |

**Failure modes to pre-check on April 24:**
- Cold-start the Space 5 min before pitch (HF Spaces sleep after ~15 min idle; first request takes 30–60 s to warm up)
- **Fallback if Space is down:** record a 45-second screencast of this exact 4-step flow on April 24 evening. Save to `~/Desktop/ring_viewer_demo.mp4`. If the live Space shows a loading screen at T+10s during your pitch, play the local video instead, narration is identical.
- Confirm hero seeds 9500/9501 appear in the dropdown on fresh reload.

---

## P3, Slide content (2 slides max)

> Build in whatever you're fastest at, Google Slides, Keynote, Marp, Slidev.
> Don't overfit to design. Judges look at content for 30 seconds each.

### Slide 1, Environment + reward design (show during minute 1 of pitch)

```
Title:  reconcile_gst2b_env
Sub:    India GST Input-Tax-Credit reconciliation, as an OpenEnv RL env

[ LEFT HALF ]                          [ RIGHT HALF ]
═══════════════                        ═══════════════

4-component reward                     Red-team CI ceiling  <0.45
(weights sum to 1.0)
                                       submit_all_matched    0.308
  R1  macro-F1           0.40           submit_all_mismatched 0.266
  R2  ITC Δ accuracy     0.25           confirm_spam          0.010
  R3  Rule 36(4)         0.25           zero_itc              0.266
  R4  step efficiency    0.10           query_only            0.349  ← max
                                       overflag_rings        0.283
Structural penalty:
  submit before query → −1.0           All six CI-enforced.
  (not clamped)                        Fails the suite if any ≥ 0.45.
```

**Bottom caption:** "No LLM in the reward path, addresses ARE §B.3.1 judge-hack concerns."

Optional decoration: reuse `envs/reconcile_gst2b_env/data/ring_viewer.png` as a
small watermark, it's the only visually interesting artifact in the repo.

---

### Slide 2, Training evidence (show during minute 2 of pitch)

**If P0 (3B + A100) landed, Version A:**

```
Title:  Training curve, Qwen2.5-3B on A100 × 4h

[ PLOT, 3/4 of slide ]
  X: training step, 0 → 400
  Y: eval total, 4 lines for R1/R2/R3/R4 + black line for total
  Overlay: dashed line at 0.353 labeled "query_only attack signature"
  Overlay: dashed line at 0.179 labeled "prompted baseline"

[ BOTTOM CAPTION ]
Step 0:  R1=0.01  R2=0.01  R3=0.99  R4=0.99  total=0.353
Step N:  R1=[X]   R2=[X]   R3=0.99  R4=0.99  total=[X]
Red-team battery unchanged (max query_only 0.349).
```

**If P0 didn't land, Version B (current state):**

```
Title:  Training run, honestly

[ TIMELINE, left→right ]

(1) 150-step collapse → (2) diagnosis → (3) Tier 1+2 fix → (4) 10-step smoke

(1)  flat 0.353 across 7 evals
     = query_only attack signature (0.349)

(2)  three angles:
     • Qwen3 enable_thinking=True ate the 512-token budget
     • TRL default beta=0.04 KL-tethered LoRA to base
     • zero tool calls → zero-std groups → zero gradient

(3)  commit 3ca024d, rewards.py untouched
     gen config + zero-std jitter + +0.02 format bonus

(4)  10-step smoke:
     reward_std  ∈ [0.002, 0.133]   positive all 10 steps
     grad_norm   ∈ [0.485, 0.550]   positive all 10 steps
     tool-call frequency 0.25, coin-flip^5 chaining problem at 0.6B

Next compute ask: Qwen2.5-3B on A100 40GB × 4h
```

---

## P4, 3 hostile Q&A probes, memorize verbatim

> 15 seconds per rebuttal. Memorize. If you freeze, say "let me pull that up"
> and flip to QA_REHEARSAL.md on your laptop, that's rehearsal, not failure.

### Probe A, "Your training curve is flat. Did training actually work?"

> **Likelihood: ~90% if Version B is used. ~30% if Version A.**

**Verbatim rebuttal (40 words, ~14 seconds):**

> "The initial 150-step run collapsed to 0.353, the query_only red-team
> signature. I diagnosed it three ways and shipped a fix in commit 3ca024d.
> The 10-step smoke in `data/smoke_test_10step.json` confirms it works
> mechanically. The remaining gap is compute scale, 0.6B on T4 can't chain
> five tool calls reliably. Qwen2.5-3B on A100 is the stated next ask."

**If pressed further:** open `data/smoke_test_10step.json` on-screen, scroll to
the `summary` block, point at `gates.pass=true`.

---

### Probe B, "Isn't the 1.18 delta inflated by raw hitting the −1.0 structural floor?"

> **Likelihood: ~60%, it's the obvious pushback on the baseline number.**

**Verbatim rebuttal (38 words, ~13 seconds):**

> "Partly, raw hits minus-one because every rollout submits without querying
> and takes the structural penalty. But the component breakdown tells the
> real story. Prompted R3 is 0.87, R4 is 0.79, those are genuine Rule 36(4)
> compliance and step efficiency, not artifacts of the floor. R1 and R2 at
> 0.01 are the actual training target."

**If pressed further:** cite `data/baseline_metrics_real.json` lines with the
per-component means.

---

### Probe C, "This is too niche. GST is an Indian tax. Why does a global hackathon care?"

> **Likelihood: ~40%, some judges will be non-Indian.**

**Verbatim rebuttal (42 words, ~14 seconds):**

> "Task shape generalizes. Two-sided document reconciliation against a
> regulator-published schema, that's invoice matching in any jurisdiction,
> supply-chain cross-filing, any audit trail. The Indian GST frame is the
> concrete instance. The reward shape is the transferable piece: arithmetic,
> clamped, with red-team ceilings tested in CI."

**Backup line if they still push:** "Fourteen million Indian businesses do this
monthly. The market is real. But the RL framework is the contribution."

---

## Rehearsal checklist (do each twice before April 25)

- [ ] Dry-run Version B pitch with a timer. Target under 3:00 with breathing.
- [ ] Dry-run Version A pitch. Same target.
- [ ] Gradio 7-click sequence, on your actual laptop, on wifi.
- [ ] Cold-start the HF Space at least once 24 hours before demo.
- [ ] Say Probe A rebuttal out loud three times without looking.
- [ ] Say Probe B rebuttal out loud three times without looking.
- [ ] Say Probe C rebuttal out loud three times without looking.
- [ ] Test the Version A pitch with `[insert final total]` slots filled in
      *if and only if* the A100 run has landed.
- [ ] Decide at April 24 noon: Version A or Version B? Do not switch after.

---

## Meta notes

- This file is gitignored-compatible. Add `envs/reconcile_gst2b_env/PITCH.md`
  to `.gitignore` if it's not already covered, or just keep it local.
- Commit count is frozen per the last "build frozen" directive. This doc is
  rehearsal material, not shippable.
- If something breaks onsite, the `JUDGE_TOUR.md` path is your fallback -
  it's a 10-minute guided read that hits the same beats as this 3-minute
  pitch, at a slower pace, and is already in-repo.
