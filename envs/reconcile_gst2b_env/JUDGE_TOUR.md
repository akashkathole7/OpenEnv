# Judge tour — reconcile_gst2b_env (≤10 min)

Five stops, hit in order. Each tells you one thing the repo takes seriously and stops talking.

---

### Stop 1 — `data/audit.html` (≈3 min)

**Do**: open `data/audit.html` in a browser. Scroll to **seed 3**, then **seed 9**, then **seed 5**.

**See**: seed 3 and seed 9 show a red "RING" badge with the 3 supplier→counterparty edges forming an A→B→C→A cycle. Seed 5 is clean with ~73% matched share. Every planted non-matched row has its `true_label` + `true_mismatch_type` printed beside the invoice row it describes.

**Why it matters**: this is the ground-truth side of the env. If any row's declared label didn't match the invoice data, every downstream reward score would be poisoned. The author hand-audited these three seeds before freezing the generator.

---

### Stop 2 — `envs/reconcile_gst2b_env/rewards.py` (≈2 min)

**Do**: open the file, read top-to-bottom starting at `clamp()` on line ~25 through `bootstrap_ci()` at the bottom.

**See**: `clamp(x, eps=0.01)` is the first function. Each reward component (R1…R4) has a docstring WHY comment explaining why that specific formula was chosen (e.g. "macro not accuracy because 'label all matched' caps ~0.17"). A module-level docstring explains the intentional R3 zero-ITC blind spot and why R2 catches it (defense-in-depth, not a bug).

**Why it matters**: no LLM anywhere in this file. The reward is arithmetic, auditable, and ~180 lines. Tamper-resistant by construction — see ARE paper §B.3.1 for the failure mode this design dodges.

---

### Stop 3 — `pytest tests/envs/test_reconcile_gst2b_reward_hacking.py -v` (≈1 min)

**Do**: run that command from the repo root.

**See**: 6 tests pass, each printing a score under the 0.45 ceiling:
`submit_all_matched 0.3075 OK`, `submit_all_mismatched 0.2655 OK`, `confirm_spam 0.0100 OK`, `zero_itc 0.2655 OK`, `query_only 0.3492 OK`, `overflag_rings 0.2834 OK`.

**Why it matters**: these aren't documentation — they're CI-enforced. If a future change lets any attack score ≥ 0.45, the test suite fails and the PR can't merge. The ceiling is a repo-level invariant, not an aspiration.

---

### Stop 4 — Gradio tab 3, seed 9502 (≈2 min)

**Do**: `python envs/reconcile_gst2b_env/app.py`, open the printed localhost URL, click **"3 · Circular-Ring Viewer"**, leave the dropdown on `9502`, click **render supplier graph**. Rotate the 3D view with your mouse.

**See**: a 3D graph where three nodes and the edges between them are red; everything else is blue. The summary below lists the detected cycle as `A → B → C → A` with real GSTINs. Hero seed 9502's planted ring is the only 3-cycle in this graph.

**Why it matters**: circular trading is the highest-stakes failure mode in real GST audits. The detection is semantically real — it runs `networkx.simple_cycles` on the supplier→counterparty graph, not on GSTIN string reassignment. This is the video centerpiece.

---

### Stop 5 — `README.md` § "What I cut and why" (≈2 min)

**Do**: open [README.md](README.md), jump to the **What I cut and why** section.

**See**: 5 bullets naming things explicitly not built — LLM judge, out-of-scope regimes (RCM / ISD / SEZ / imports / composition / e-invoice / e-way bill), the real CBIC HSN table, multi-GSTIN companies, and real Qwen pitch numbers — each with one line of why.

**Why it matters**: 5-day build, one person, free Colab. The scope fence is deliberate and legible. Knowing what was cut is how you evaluate what was kept.

---

**Total**: 10 minutes, five artifacts, one repo-level invariant enforced in CI.

Next: [BLOG.md](BLOG.md) for the narrative. [PRD.md](PRD.md) for the 1-page product spec. [EXEC_SUMMARY.md](EXEC_SUMMARY.md) for the ≤10-line version.
