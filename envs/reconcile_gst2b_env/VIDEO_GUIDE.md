# 90-second demo video, complete production guide

Local-only doc, not committed. Read once top-to-bottom before starting.
Total time budget: **2–3 hours** including retakes.

---

## Step 0, Pick your approach (decide before doing anything)

Two viable paths. **Pick one before opening any software.** Switching mid-way doubles your time.

| | **Approach A, Silent screen capture** | **Approach B, Voice-over walkthrough** |
|---|---|---|
| What happens on-screen | Pure screen recording, no voice, on-screen text overlays for context | Screen recording + your voice narrating the 197-word script |
| Judge experience | Scans the visuals; decides in 5s | Listens through; higher engagement |
| Total time to produce | **45–60 min** | **90–150 min** |
| Polish required | Low | Medium (audio quality matters) |
| Risk of embarrassing retakes | Low | Medium (tongue-twisters, ums) |
| Best for | Time-constrained, non-native English nerves | Higher score potential, confident voice |

**Recommendation tree:**
- If you have **< 2 hours total** OR are nervous about English voiceover → **Approach A**
- If you have **≥ 2 hours** AND feel confident reading the script → **Approach B**
- If unsure → start **Approach A**. It's always shippable. Upgrade to B only if time allows.

The rest of this document covers both paths. Skip the sections for the approach you didn't pick.

---

## Step 1, Setup (5 minutes, common to both approaches)

### 1a. Install screen-recording software

Ubuntu options, in order of preference:

```bash
# Option 1: OBS Studio (most control, learning curve ~10 min)
sudo apt install obs-studio

# Option 2: GNOME Screen Recorder (zero setup, Ctrl+Alt+Shift+R)
# Already installed on Ubuntu 22.04+

# Option 3: SimpleScreenRecorder (lightweight, easy)
sudo apt install simplescreenrecorder

# Option 4: Loom browser extension (https://www.loom.com/)
# Only if the other three fail; has cloud dependency
```

**My pick for you: GNOME Screen Recorder** (Ctrl+Alt+Shift+R). Zero setup, records at 1080p, outputs to `~/Videos/Screencasts/`. You lose fine control but gain 20 minutes not fighting OBS configuration.

### 1b. Silence everything

```bash
# Slack, Discord, Mail, Telegram, WhatsApp Web, quit entirely
# Phone: airplane mode or silent on another room's table
# Browser: disable all notifications
```

Chrome notification kill:
```
chrome://settings/content/notifications → Don't allow sites to send notifications
```

### 1c. Prepare the three browser tabs, in this exact order

```bash
cd /home/aakash/Videos/ReconcileEnv-GST2B
# Open audit HTML in default browser
xdg-open data/audit.html

# Start local Gradio (background, port 7860)
PYTHONPATH=src:envs uv run python envs/reconcile_gst2b_env/app.py &
# Wait ~5 seconds for startup
sleep 5

# Open HF Space
xdg-open https://huggingface.co/spaces/akashkathole/reconcile_gst2b_env

# Open local Gradio
xdg-open http://localhost:7860
```

Order your tabs:
- **Tab 1:** `data/audit.html`, scroll to **seed 3** ring table
- **Tab 2:** HF Space (remote), Space must be **warmed up**. If it shows "Building" or "App starting", wait. Idle Spaces sleep after ~30 min.
- **Tab 3:** `localhost:7860`, click into **Tab 3 "Circular-Ring Viewer"**. Select **seed 9502** from the dropdown. Click **Render**. Wait for the 3D plot. Rotate once so the triangle is visible. **This is your hero shot.**

### 1d. Pre-warm the HF Space

```bash
curl -s https://huggingface.co/spaces/akashkathole/reconcile_gst2b_env > /dev/null
```

Wait 30 seconds. Refresh the tab. If it now shows a working UI, you're good. If it still says "Space is building" or "App is starting", wait another 60 seconds.

**Multi-perspective risk:** HF Spaces sleep after idle. If you start recording with a cold Space, 30 seconds of your 90-second video will be a spinner. **Unacceptable.** Pre-warm is mandatory.

### 1e. Set your screen resolution to **1920×1080**

```bash
xrandr  # list available resolutions
xrandr --output eDP-1 --mode 1920x1080  # adjust output name to match yours
```

Higher resolution = harder to read on judges' phones. Lower resolution = looks unprofessional. 1080p is the goldilocks.

### 1f. Set browser zoom to 110-125% on all three tabs

Ctrl++ (or Cmd++ on Mac) two clicks. This makes text large enough to read at 1.5x playback on a phone.

---

## Step 2, Recording

### Approach A, Silent screen capture (45 min total)

**Pre-recording mental rehearsal (5 min):**
Walk through the 8-beat timeline without recording. Time yourself with a stopwatch. Hit every beat at the right second. Do this twice.

**Hit record. Timeline:**

> ⚠️ **IMPORTANT, only UI that actually exists is referenced below.** Tabs 1 / 2 / 4 on the live HF Space are static placeholders (per Option A). The only interactive tab is **Tab 3 (Circular-Ring Viewer)** with a seed dropdown and **"render supplier graph"** button. Do NOT click any "Reset" or "Step" button, they do not exist in this deployment.

| T+ | Action | What the judge sees |
|---|---|---|
| 0:00–0:04 | **Local Gradio (localhost:7860) → Tab 3 Ring Viewer.** Seed 9502 already rendered. Rotate the plot once with mouse drag. | Motion, red nodes, 3D plot. **This is the hook.** |
| 0:04–0:08 | Hover over one of the red ring nodes, the GSTIN tooltip appears. Hold. | Detail: "these are real GSTINs". |
| 0:08–0:18 | Switch to `audit.html` tab. Scroll slowly to seed 3's "planted circular ring" table. Pause 3 seconds. | 3 GSTIN rows showing ground-truth ring. |
| 0:18–0:28 | Keep scrolling to the "planted non-matched invoices" table. Pause 3 seconds. | 5 mismatch types listed with rows. |
| 0:28–0:40 | Scroll down to seed 9 audit section. Pause on its ring table briefly. Scroll to seed 5 (clean, no ring). | Consistency across seeds; rings only when planted. |
| 0:40–0:55 | Switch to **live HF Space tab** (huggingface.co/spaces/akashkathole/reconcile_gst2b_env). Let the hero block render. Slow-scroll/zoom over the "14M businesses · Scaler AI Labs · 16 verbs · 6 red-team <0.45 · 42 tests" line. Pause 2 s. | Public deployment + value prop + sub-theme keyword. |
| 0:55–1:08 | On HF Space, click **Tab 3 (Circular-Ring Viewer)**. Click **"render supplier graph"** button (seed 9502 is pre-selected). Wait for 3D plot. Rotate once. | Same env works remotely. Deployment is real, not just a repo. |
| 1:08–1:18 | Click **Files** tab on HF Space (top bar). Scroll slowly past `README.md`, `BLOG.md`, `scripts/`, `data/`, `server/`. | Rigor signal, full repo browsable, not a stub. |
| 1:18–1:30 | Click back to **App** tab. Settle on top of the App view showing the "14M businesses…" hero block, with the green **Running** badge visible at the top. Hold for 3+ seconds. | URL + badge burned into viewer's memory → screenshottable close. |

**Add on-screen text overlays in post-production** (see Step 3a) at these moments:
- 0:00: "reconcile_gst2b_env, 14M businesses × monthly × manual"
- 0:18: "5 planted mismatch types + circular-trading rings"
- 0:40: "Live on Hugging Face Spaces"
- 1:08: "42 tests · red-team CI · bit-identical reproduce"
- 1:20: "Scaler AI Labs Enterprise Workflow target"

Stop recording. Review once. If any segment has a 2+ second lag (loading, scroll hesitation), re-record just that segment or trim.

---

### Approach B, Voice-over walkthrough (90–150 min total)

Use the revised script from earlier (197 words, word-for-word).

**Pre-recording mental rehearsal (15 min):**
Read the script aloud **5 times**. Not silent, not lip-moving, full voice. This unlocks tongue memory for the hard words (*entropy, zero-point-one-two, macro-F1*).

**Check your mic level:**
Say the first line into your mic while watching the level meter. Peak should hit **-6 dB**, not -20 dB (too quiet) or 0 dB (clipping). In OBS: Audio Mixer panel. In GNOME recorder: use `alsamixer` to adjust mic input.

**Recording strategy, pick one:**

**B-1 "One take, all together":** Talk through the script while switching tabs. Most efficient if it works. Expect **5–10 retakes.** Budget 90 minutes.

**B-2 "Voice first, visuals after":** Record the voice track in Audacity (silent screen). Then record the screen with the voice playing on headphones for timing. Assemble in kdenlive or OpenShot. More work but cleaner audio. Budget 2.5 hours.

**Hit record. Timeline (matches the revised script):**

| T+ | Visual | Voice (READ WORD-FOR-WORD) |
|---|---|---|
| 0:00–0:09 | Tab 3 Ring Viewer rotating | "Three businesses. Round-tripping invoices. Manufacturing fake tax credit, caught by my OpenEnv reinforcement-learning environment for enterprise compliance workflows." |
| 0:09–0:24 | Tab 1 audit.html scrolling through seed 3 | "Every month, millions of businesses match their internal books against a regulator's filing. In India alone, fourteen million businesses. Still manual. Rule-heavy. The rules are published, so correctness is crisp, the textbook RL target." |
| 0:24–0:48 | Static overlay: reward table + red-team scores + test count | "Reconcile-GST-env on OpenEnv. Sixteen typed tool verbs. Five labels. Four arithmetic reward components, each clamped. No LLM judge in the reward path. Six red-team attacks enforced in CI, all under zero-point-four-five. Forty-two tests green. Bit-identical reproducibility." |
| 0:48–1:13 | Three-column failure-mode matrix (from BLOG §6) | "I pre-ran three Qwen3 models. The point-six B plateaued. The one-point-seven B collapsed, entropy to zero-point-one-two, zero tool calls. The four B over-queried until the step budget exhausted. Three scales, three distinct reward components pinned to floor. The environment is structurally hard." |
| 1:13–1:30 | Text slide: "Tool-SFT → GRPO polish" + URLs | "On-site plan: supervised fine-tuning on synthetic expert trajectories, then GRPO polish. Targeting Scaler AI Labs enterprise-workflow track. Environment live on Hugging Face, link stays on screen for five seconds. Try it." |

**Critical voice delivery cues:**
- Stand up while recording. Posture lifts voice energy 20%.
- Smile before each major beat's opening word. Mic picks up the vocal warmth.
- **Pause 0.5s after "14 million businesses"**, let the number land.
- **Pause 0.3s after each "0.6B / 1.7B / 4B"** in the training beat.
- Do NOT say "um", "uh", "like". If you catch yourself, stop, start over from the beat.
- Last 5 seconds: do NOT speed up. Judges remember the last sentence.

---

## Step 3, Post-production (30–45 minutes)

### 3a. Trim + edit

**Trim targets:** 85–95 seconds final runtime. Anything longer gets cut mid-play by judges' "next" click. Anything shorter feels rushed.

**Tools, in order of preference:**
- **Kdenlive** (Ubuntu native): `sudo apt install kdenlive`. Full-featured.
- **OpenShot**: `sudo apt install openshot-qt`. Simpler.
- **Built-in trim:** if your recording is close to 90s already, Ubuntu's Videos app (Totem) has a basic trim.

**Must-do edits:**
1. **Trim first 1–2 seconds** of any setup dead-air.
2. **Cut out page loads, dropdown-open delays**, anything > 1.5s of nothing happening.
3. **Add on-screen text overlays** (Approach A only) at the 4 moments specified.
4. **Add a 0.5-second intro card** with title: `reconcile_gst2b_env`.
5. **Add a 3-second outro card** with: HF Space URL, GitHub URL, your name. **This is the single most screenshot-worthy frame, make it readable.**

**Do NOT:**
- Add background music, unprofessional if not mixed perfectly, distracts at 1.5x.
- Add transitions (wipes, fades) between clips, looks amateur. Hard cuts are cleaner.
- Add a face-cam overlay unless your framing and lighting are good. A bad face-cam is worse than none.

### 3b. Export settings

| Setting | Value | Why |
|---|---|---|
| Resolution | 1920×1080 | YouTube standard, readable on phones |
| Frame rate | 30 fps | Smooth enough, smaller file |
| Bitrate | 8–12 Mbps | Quality without huge upload |
| Codec | H.264 | YouTube's preferred |
| Audio (Approach B) | AAC, 128 kbps, mono | Voice doesn't need stereo |
| Container | MP4 | Universal |

Export to `~/Videos/reconcile_demo_v1.mp4`. Check filesize, should be 80–150 MB.

### 3c. Upload to YouTube

1. Go to https://studio.youtube.com (sign in with Google account).
2. Upload video → select `reconcile_demo_v1.mp4`.
3. **Title:** `reconcile_gst2b_env, OpenEnv RL for Enterprise Compliance Workflows`
4. **Description:**
   ```
   90-second demo of reconcile_gst2b_env, an OpenEnv reinforcement-learning
   environment for multi-turn enterprise compliance workflows, instantiated
   on Indian GST input-tax-credit reconciliation. Targeting Scaler AI Labs
   enterprise-workflow sub-theme at the Meta × Scaler Hackathon, Bangalore
   2026.

   Live Space: https://huggingface.co/spaces/akashkathole/reconcile_gst2b_env
   Repo: https://github.com/akashkathole7/OpenEnv (branch scaffold/reconcile-gst2b)
   ```
5. **Visibility: Unlisted** (not Public, not Private). Anyone with link can watch.
6. **Thumbnail:** upload a custom frame. The Ring Viewer with red nodes visible + text overlay of the project name. Auto-generated thumbnails are always bad.
7. Copy the share URL after upload finishes.

**Backup if YouTube fails:** Upload the MP4 directly to the HF Space as an asset. Less standard, but works.

### 3d. Update docs with the URL

Tell me (Claude) once you have the YouTube URL:

```
Video uploaded to YouTube unlisted at <paste URL>.

Add it to:
  - envs/reconcile_gst2b_env/README.md under a new "## Live Demo" section at the top (right after the intro paragraph, before any other section)
  - envs/reconcile_gst2b_env/BLOG.md at the very top under the headline, formatted as "> 📹 90-second demo: <URL>"
  - envs/reconcile_gst2b_env/EXEC_SUMMARY.md, add a line: "11. Video: <URL>"
  - envs/reconcile_gst2b_env/ROUND2_PROBLEM_STATEMENT.md, update the "At a glance" table with a "Video" row
  - HF Space README (huggingface.co/spaces/akashkathole/reconcile_gst2b_env/README.md in your local Space clone)

One commit: "docs: add 90s demo video URL across judge-facing docs".
Do not push automatically. Stop after commit.
```

I'll do the edits + commit. You push.

---

## Multi-perspective risk matrix

| Risk | Probability | Mitigation |
|---|---:|---|
| HF Space cold-starts mid-recording | **60%** | Pre-warm with curl. Verify green "Running" badge before hitting record. |
| Gradio ring viewer fails to render | 15% | Rotate the plot manually once before recording to force render. Have a screenshot fallback. |
| Audio levels wrong (too quiet/clipping) | 30% (Approach B) | Test a 10-second clip before the real take. Check peak at -6 dB. |
| Tongue-twister flubs (Approach B) | 40% | Read script 5x aloud before recording. Pre-identify danger words: *entropy, reproducibility, zero-point-one-two*. |
| Video upload fails / rejected | 5% | Upload from a stable wifi connection, not mobile tether. Retry once. |
| Final runtime > 95s | 30% | Hard-trim in post. Cut the outro slide to 2s if needed. Shave scroll-hesitation milliseconds. |
| You forget to pre-warm HF Space | 50% | Set a phone alarm 5 min before recording: "curl HF Space". |
| Recording too long (rambling) | 20% (Approach B) | Use a stopwatch visible on a second screen. Redo if you exceed 1:35. |
| Laptop fan noise audible on mic | 25% | Run `sudo cpupower frequency-set -g powersave` before recording to quiet fans. |
| Notifications pop up despite silencing | 10% | Quit apps entirely, don't rely on DND mode. |

---

## Judge-psychology cheat sheet (read before Step 2)

Print this or keep on a second screen while recording:

- **First 5 seconds:** must have motion on screen. The Ring Viewer rotating IS the hook. A title card is not.
- **Text overlays:** must be readable at **1.5x playback on a 6-inch phone screen**. This means 48pt+ font for key numbers. Test this by playing your trimmed video on your phone at 1.5x.
- **Numbers land with pauses.** 14 million. 16 verbs. 42 tests. 0.45 ceiling. Each needs a 0.3s audible or visual beat.
- **One idea per frame.** If a text overlay has 3 points, the viewer reads none of them. Show one point, then cut.
- **End frame on screen for 3+ seconds.** Judges screenshot the URL. Give them time.
- **No music under voice.** Ever. Silence is confidence.

---

## Final commit sequence (after you have the video URL)

After uploading and telling me to update docs:

```bash
# After Claude makes the commit, you push:
cd /home/aakash/Videos/ReconcileEnv-GST2B
git push fork scaffold/reconcile-gst2b
```

That's the last required action for the pitch/submission. After that: pitch rehearsal only. No more code.

---

## If everything goes wrong, the 30-minute emergency path

Short on time? Panicking? Here's the minimum viable video:

1. Open `localhost:7860` → Tab 3 → seed 9502 → Render. That's it.
2. Ctrl+Alt+Shift+R. Record 60 seconds of rotating the 3D plot from multiple angles.
3. Open your text editor, type: "reconcile_gst2b_env, 14M businesses, monthly, still manual. 16 verbs, 5 labels, 6 red-team attacks CI-enforced <0.45. Live on HF Spaces. Targeting Scaler AI Labs."
4. Screen-record the text for 15 seconds.
5. Concatenate with `ffmpeg`:
   ```bash
   ffmpeg -f concat -safe 0 -i <(for f in ring.mp4 text.mp4; do echo "file '$PWD/$f'"; done) -c copy combined.mp4
   ```
6. Upload unlisted to YouTube.
7. Ship.

**This hits the minimum requirement**, misses the polish score, but survives the min-requirement filter. Better than no video.

---

## When to stop iterating

Your third take is never noticeably better than your fifth take. **Ship the fourth-or-fifth take.** Perfectionism on this video eats pitch-rehearsal time, which has higher ROI. A B+ video that ships beats an A+ video that doesn't.

**Done criteria:**
- [ ] Video runs 85–95 seconds
- [ ] Motion visible in first 5 seconds
- [ ] All 5 key numbers on screen or spoken (14M, 16 verbs, 6 attacks <0.45, 42 tests, 1.18 baseline delta)
- [ ] HF Space URL visible for ≥ 3 seconds at end
- [ ] No dead-air > 1.5s anywhere
- [ ] Audio (if any) no clipping, no echo, no laptop fan noise
- [ ] Uploaded unlisted to YouTube
- [ ] URL copied to clipboard

When all 8 boxes are checked, stop. Paste the URL to me. Move on.
