# Kaggle instructions — real GRPO training

Step-by-step for running `notebooks/train_grpo_real_kaggle.ipynb` on
Kaggle's free T4, in background mode, end-to-end.

## 1. Push the branch

Your fork must have the latest `scaffold/reconcile-gst2b`:

```bash
cd /home/aakash/Videos/ReconcileEnv-GST2B
git push fork scaffold/reconcile-gst2b
```

## 2. Create the Kaggle notebook

1. Go to https://www.kaggle.com/code → **+ New Notebook**.
2. Top-right: **File → Import Notebook**, paste the GitHub raw URL for
   `notebooks/train_grpo_real_kaggle.ipynb` on your fork
   (`https://raw.githubusercontent.com/akashkathole7/OpenEnv/scaffold/reconcile-gst2b/notebooks/train_grpo_real_kaggle.ipynb`),
   and click **Import**.
3. On the right sidebar:
   - **Accelerator**: **GPU T4 x2** (or just **GPU T4** if only one is offered).
   - **Internet**: **On** (required for pip install + model download).
   - **Environment**: leave at "Latest".

## 3. Enable background execution ("Save & Run All")

This is Kaggle's equivalent of a detached job. The notebook runs to
completion even if you close the tab.

1. Top-right: **Save Version**.
2. In the dialog, pick **Save & Run All (Commit)**.
3. Optional: give the version a name like "grpo-real-150-steps-v1".
4. Click **Save**.

Kaggle now runs every cell top-to-bottom on a fresh container. When it
finishes (or fails), you get an email and the **Versions** panel shows
the output.

## 4. Monitor progress

- **Versions** tab (right sidebar): shows Running / Success / Error.
- Click the running version → **Notebook** view → scroll to Cell 3:
  the training logs stream live (eval prints every 25 steps).
- If it dies early, the top of the cell output has the traceback.

## 5. Grab the outputs

Once the version shows **Success**:

1. Click the version → **Output** tab (right sidebar).
2. Download:
   - `curves.json` — the 7-eval-point training curves (schema matches
     `data/dryrun/training_curves.json`)
   - `curves.png` — plot for the blog post
   - `final_checkpoint.tar.gz` — LoRA adapter for the trained model

3. On your laptop:

```bash
cd /home/aakash/Videos/ReconcileEnv-GST2B
mkdir -p data/real_training
cp ~/Downloads/curves.json data/real_training/curves.json
cp ~/Downloads/curves.png  data/real_training/curves.png   # optional
# Don't commit the tar — it's ~30 MB of LoRA weights, doesn't belong in git.
# Instead, push it to a Hugging Face model repo if you want it public.

# Allowlist the new artifacts in .gitignore (analog to baseline_metrics_real.json).
echo '!data/real_training/curves.json' >> .gitignore
echo '!data/real_training/curves.png' >> .gitignore
git add data/real_training/curves.json data/real_training/curves.png .gitignore
git commit -m "data: real GRPO training curves (Qwen3-0.6B, 150 steps)"
git push fork scaffold/reconcile-gst2b
```

Then paste the numbers back to Claude and ask for the README / BLOG
update commit.

## 6. OOM fallback

Qwen3-0.6B is already the smallest TRL-supported Qwen3 — there's no
smaller model to swap to. If Cell 3 fails with `[FATAL] CUDA OOM on
model load`, recover by reducing the training footprint:

1. Open `envs/reconcile_gst2b_env/scripts/train_grpo_real.py` in a fresh
   branch on your laptop.
2. Change `NUM_GENERATIONS = 4` → `2`, OR `MAX_COMPLETION_LENGTH = 512` → `256`.
3. Push, re-clone in Kaggle (Cell 2 force-refresh handles this), and
   **Save & Run All** again.

If both fail, switch the Kaggle accelerator from T4 to L4 (24 GB) if
your account has access, or move the run to Colab A100 (40 GB).

## 7. If training completes but curves are flat

Per the script header: flat curves after 150 real steps is a REAL
finding ("env too hard for 0.6B at this scale") and should still be
reported honestly. Do NOT hand-fake the plot. Update the blog's
training-status paragraph to name this as the current state, and use
the compute ask to argue for A100 × 10 h on a larger Qwen3 instead of
T4 × 3 h on 0.6B.
