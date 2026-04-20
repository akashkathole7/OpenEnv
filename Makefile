# reconcile_gst2b_env — top-level Makefile.
# Judge-facing. `make reproduce` regenerates every data/ artifact from a
# clean checkout; outputs are byte-identical to the committed JSONs.

PY ?= uv run
ENV_DIR := envs/reconcile_gst2b_env
ENV_SCRIPTS := $(ENV_DIR)/scripts
PYTHONPATH_VAR := PYTHONPATH=src:envs

.PHONY: help install test reproduce launch clean

help:
	@echo "reconcile_gst2b_env — make targets"
	@echo ""
	@echo "  make install   — uv sync + regenerate env uv.lock"
	@echo "  make test      — run the 41-test suite for reconcile_gst2b_env"
	@echo "  make reproduce — regenerate all data/ artifacts (diversity, baseline,"
	@echo "                   ablation, audit HTML, hero baseline). Outputs are"
	@echo "                   byte-identical to committed JSONs on a clean repo."
	@echo "  make launch    — start the Gradio UI (ring viewer on tab 3)"
	@echo "  make clean     — remove generated artifacts and __pycache__"

install:
	uv sync --all-extras
	cd $(ENV_DIR) && uv lock

test:
	$(PY) pytest tests/envs/test_reconcile_gst2b_*.py -v

reproduce:
	@echo "=== diversity_test (100 seeds × 2 modes) ==="
	$(PYTHONPATH_VAR) $(PY) python $(ENV_SCRIPTS)/diversity_test.py
	@echo ""
	@echo "=== baseline (3 conditions × 30 seeds × 3 samples, mock) ==="
	$(PYTHONPATH_VAR) $(PY) python $(ENV_SCRIPTS)/baseline.py
	@echo ""
	@echo "=== ablation (4 component drops) ==="
	$(PYTHONPATH_VAR) $(PY) python $(ENV_SCRIPTS)/ablation.py
	@echo ""
	@echo "=== audit HTML (seeds 0..9) ==="
	$(PYTHONPATH_VAR) $(PY) python -c "from envs.reconcile_gst2b_env.audit import generate_audit_html; generate_audit_html(list(range(10)), 'data/audit.html')"
	@echo ""
	@echo "=== hero baseline (seeds 9500/9501/9502) ==="
	$(PYTHONPATH_VAR) $(PY) python $(ENV_SCRIPTS)/hero_baseline.py
	@echo ""
	@echo "=== bit-identical check against committed artifacts ==="
	@$(PY) git diff --stat -- data/ 2>/dev/null | tail -20 || true
	@if $(PY) git diff --quiet -- data/ 2>/dev/null; then \
		echo "OK: all data/ artifacts byte-identical to committed versions"; \
	else \
		echo "WARN: data/ differs from committed versions — see diff above"; \
	fi

launch:
	$(PYTHONPATH_VAR) $(PY) python $(ENV_DIR)/app.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(ENV_DIR)/.venv
	@echo "cleaned pycache, egg-info, .pytest_cache, env .venv"
	@echo "note: data/ artifacts preserved — use 'git clean -fd data/' to wipe those"
