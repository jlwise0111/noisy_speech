# noisy_speech: tests, baseline data and the diagnostic viewer.
#
#   make            show this help
#   make all        check → unit tests → build viewer → viewer tests → collect data → metrics → start viewer
#   make train-evaluate   train/run every model, then score them all with the metric suite
#   make pipeline   baselines in parallel → U-Nets (28spk, 56spk, both) one by one → score all → open the viewer
#   make pipeline-all       unit tests → smoke pipeline → full pipeline, stopping at the first failure
#
# The dataset path comes from DATASET, else $VALENTINI_DIR. Set it once per machine in an untracked
# local.mk (one line: DATASET := /path/to/valentini_noisy_speech_database), or on the command line:
#   make data DATASET=/path/to/valentini_noisy_speech_database
#   make data-ridge FORCE=1          # redo runs that already exist
#
# Paths in this project contain spaces, so every path is quoted in the recipes and no file
# targets are used: data targets skip a run whose completion marker already exists.

SHELL   := /bin/bash
-include local.mk
PYTHON  ?= python3
DATASET ?= $(VALENTINI_DIR)
FORCE   ?=
PORT    ?= 8501
URL     := http://localhost:$(PORT)

LINEAR  := linear_baseline
WIENER  := wiener_baseline
VIEWER  := diagnostic_viewer
METRICS := metric_suite
UNET    := unet_model
UNET_PY := .venv/bin/python
UNET_ARGS ?=
METRICS_PY := .venv/bin/python
METRIC_ARGS ?=
FIGURES := figures
FIGURES_PY := .venv/bin/python
PIPELINE_ARGS ?=
COMPOSE := cd $(VIEWER) && PORT=$(PORT) docker compose
PLAYER  := model_player
PLAYER_PORT ?= 8502
PLAYER_URL  := http://localhost:$(PLAYER_PORT)
# DATASET is made absolute (from the project folder) before the cd, so a relative path such as
# valentini_eda still points at the dataset rather than at model_player/valentini_eda. Empty or
# missing: the player mounts its empty no_dataset folder and offers uploads only.
PLAYER_COMPOSE := DS="$(if $(strip $(DATASET)),$$(cd "$(DATASET)" 2>/dev/null && pwd))"; \
  cd $(PLAYER) && PORT=$(PLAYER_PORT) DATASET_DIR="$$DS" docker compose

# $(call already_ran,<marker file>): succeeds (and says so) if the run already finished and FORCE is unset.
already_ran = test -z "$(FORCE)" && test -f "$(1)" && echo "  ✓ skip: $(1) exists (FORCE=1 to redo)"

.DEFAULT_GOAL := help
.PHONY: help all check check-python check-docker check-dataset \
        test test-linear test-wiener test-viewer test-metrics test-unet metrics-venv unet-venv evaluate train-evaluate figures-venv figures \
        data-unet unet-smoke unet-all pipeline pipeline-smoke pipeline-all \
        data data-ridge data-ridge-28spk data-ridge-56spk data-ridge-both \
        data-wiener data-wiener-leading data-wiener-minstats data-wiener-oracle data-wiener-train \
        smoke viewer viewer-close viewer-build viewer-up viewer-down viewer-logs viewer-open \
        player player-build player-up player-down player-logs player-open test-player status clean

help: ## Show this help
	@echo "Targets (run from the project folder):"
	@grep -hE '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[1m%-22s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Variables: DATASET=\"$(DATASET)\""
	@echo "           PYTHON=$(PYTHON)  PORT=$(PORT)  FORCE=$(FORCE)"

# ---------------------------------------------------------------------------- full pipeline
all: check test-linear test-wiener test-metrics viewer-build test-viewer data evaluate viewer-up ## Whole suite in order, then start the viewer
	@echo
	@echo "✅ All tests passed, data collected and scored ($(METRICS)/results/summary.md), viewer at $(URL)"

# ---------------------------------------------------------------------------- environment checks
check: check-python check-docker check-dataset ## Check Python packages, Docker and the dataset

check-python:
	@$(PYTHON) -c "import numpy, scipy, soundfile, pytest" 2>/dev/null \
	  && echo "  ✓ $(PYTHON): numpy, scipy, soundfile, pytest" \
	  || { echo "  ✗ $(PYTHON) is missing packages. Install with:"; \
	       echo "      $(PYTHON) -m pip install numpy scipy soundfile pytest"; exit 1; }

check-docker:
	@docker info >/dev/null 2>&1 && echo "  ✓ Docker is running" \
	  || { echo "  ✗ Docker is not running (start Docker Desktop)"; exit 1; }

check-dataset:
	@test -n "$(DATASET)" || { echo "  ✗ no dataset path: set DATASET=/path/to/valentini_noisy_speech_database"; \
	  echo "    (on the command line, in local.mk, or export VALENTINI_DIR)"; exit 1; }
	@for d in clean_testset_wav noisy_testset_wav clean_trainset_28spk_wav noisy_trainset_28spk_wav \
	          clean_trainset_56spk_wav noisy_trainset_56spk_wav; do \
	  test -d "$(DATASET)/$$d" || { echo "  ✗ missing $$d in DATASET=\"$(DATASET)\""; exit 1; }; \
	done; echo "  ✓ Valentini dataset found"

# ---------------------------------------------------------------------------- tests (each runs on its own)
test: test-linear test-wiener test-metrics test-unet test-viewer ## Run all five test suites

test-linear: check-python ## Ridge baseline unit tests (host Python)
	@echo "▶ $(LINEAR) tests"
	cd $(LINEAR) && $(PYTHON) -m pytest -q -p no:cacheprovider

test-wiener: check-python ## Wiener baseline unit tests (host Python)
	@echo "▶ $(WIENER) tests"
	cd $(WIENER) && $(PYTHON) -m pytest tests -q -p no:cacheprovider

test-metrics: metrics-venv ## Metric suite unit tests (its own .venv)
	@echo "▶ $(METRICS) tests"
	cd $(METRICS) && $(METRICS_PY) -m pytest -q -p no:cacheprovider

test-unet: unet-venv ## U-Net + shared preprocessing tests (its own .venv with PyTorch)
	@echo "▶ $(UNET) tests"
	cd $(UNET) && $(UNET_PY) -m pytest -q -p no:cacheprovider

test-viewer: viewer-build ## Viewer tests, inside the freshly built Docker image
	@echo "▶ $(VIEWER) tests (in Docker)"
	$(COMPOSE) run --rm --no-deps viewer python -m pytest -q -p no:cacheprovider

# ---------------------------------------------------------------------------- data for the viewer
# Each run writes <run>/{clean,noisy,enhanced|processed}/<stem>.wav for the 824-file test set.
data: data-ridge data-wiener ## Collect every test-set run the viewer compares

data-ridge: data-ridge-28spk data-ridge-56spk data-ridge-both ## Ridge baseline: train on 28spk, 56spk, both

data-ridge-28spk: check-dataset
	@$(call already_ran,$(LINEAR)/outputs/ridge_28spk/config.json) || \
	  (cd $(LINEAR) && $(PYTHON) ridge_baseline.py --dataset "$(DATASET)" --train-set 28spk --run-name ridge_28spk)

data-ridge-56spk: check-dataset
	@$(call already_ran,$(LINEAR)/outputs/ridge_56spk/config.json) || \
	  (cd $(LINEAR) && $(PYTHON) ridge_baseline.py --dataset "$(DATASET)" --train-set 56spk --run-name ridge_56spk)

data-ridge-both: check-dataset
	@$(call already_ran,$(LINEAR)/outputs/ridge_both/config.json) || \
	  (cd $(LINEAR) && $(PYTHON) ridge_baseline.py --dataset "$(DATASET)" --train-set both --run-name ridge_both)

data-wiener: data-wiener-leading data-wiener-minstats data-wiener-oracle ## Wiener baseline: leading, minstats, oracle (test set)

data-wiener-leading: check-dataset
	@$(call already_ran,$(WIENER)/outputs/wiener_leading_test/quick_metrics.csv) || \
	  (cd $(WIENER) && $(PYTHON) run_baseline.py --dataset "$(DATASET)" --split test --noise-est leading)

data-wiener-minstats: check-dataset
	@$(call already_ran,$(WIENER)/outputs/wiener_minstats_test/quick_metrics.csv) || \
	  (cd $(WIENER) && $(PYTHON) run_baseline.py --dataset "$(DATASET)" --split test --noise-est minstats)

data-wiener-oracle: check-dataset
	@$(call already_ran,$(WIENER)/outputs/oracle_test/quick_metrics.csv) || \
	  (cd $(WIENER) && $(PYTHON) run_baseline.py --dataset "$(DATASET)" --split test --method oracle)

data-wiener-train: check-dataset ## Optional: Wiener on the train28/train56 splits (~10 GB at 16 kHz, not part of 'all')
	@$(call already_ran,$(WIENER)/outputs/wiener_leading_train28/quick_metrics.csv) || \
	  (cd $(WIENER) && $(PYTHON) run_baseline.py --dataset "$(DATASET)" --split train28)
	@$(call already_ran,$(WIENER)/outputs/wiener_leading_train56/quick_metrics.csv) || \
	  (cd $(WIENER) && $(PYTHON) run_baseline.py --dataset "$(DATASET)" --split train56)

smoke: check-dataset ## Quick end-to-end check of both baselines on a few files (writes outputs/smoke*)
	cd $(LINEAR) && $(PYTHON) ridge_baseline.py --dataset "$(DATASET)" --limit-files 500 --test-limit 5 --run-name smoke
	cd $(WIENER) && $(PYTHON) run_baseline.py --dataset "$(DATASET)" --split test --limit 5 --run-name smoke

# ---------------------------------------------------------------------------- U-Net (Person B)
# PyTorch lives in unet_model/.venv. Training takes hours, so data-unet is not part of 'data' or 'all'.
unet-venv: check-python
	@test -x "$(UNET)/$(UNET_PY)" || { echo "  creating $(UNET)/.venv (downloads PyTorch)"; \
	  cd $(UNET) && $(PYTHON) -m venv .venv && $(UNET_PY) -m pip install -q -r requirements.txt; }

# train_unet.py writes config.json before training starts and adds "seconds" only when the whole
# run (training + test enhancement) has finished, so that key, not the file, marks a finished run.
# An unfinished run continues from last.pt (--resume); FORCE=1 starts over.
UNET_RUN := unet_28spk
data-unet: unet-venv check-dataset ## Train the U-Net on 28spk and enhance the test set (hours; resumes if interrupted)
	@if test -z "$(FORCE)" && grep -qs '"seconds"' "$(UNET)/outputs/$(UNET_RUN)/config.json"; then \
	  echo "  ✓ skip: $(UNET)/outputs/$(UNET_RUN) is a finished run (FORCE=1 to redo)"; \
	elif pgrep -f -- "--run-name $(UNET_RUN)( |$$)" >/dev/null; then \
	  echo "  ✗ $(UNET_RUN) is already training (by make pipeline: see logs/$(UNET_RUN).log); not starting a second copy"; exit 1; \
	else \
	  cd $(UNET) && $(UNET_PY) -u train_unet.py --dataset "$(DATASET)" --train-set 28spk --run-name $(UNET_RUN) \
	    $(if $(FORCE),,--resume) $(UNET_ARGS); \
	fi

# Each step is a bigger commitment than the last, so a problem shows up at the cheapest step that
# can catch it. Sub-makes (not prerequisites) keep the order fixed even under make -j.
unet-all: ## U-Net in order: venv → tests → smoke run → full training (hours) → score it
	$(MAKE) test-unet
	$(MAKE) unet-smoke
	$(MAKE) data-unet
	$(MAKE) evaluate METRIC_ARGS="--include $(UNET_RUN)"

unet-smoke: unet-venv check-dataset ## U-Net end to end on 300 files, 2 epochs (~3 min)
	cd $(UNET) && $(UNET_PY) train_unet.py --dataset "$(DATASET)" --limit-files 300 --epochs 2 --test-limit 20 --run-name unet_smoke

# ---------------------------------------------------------------------------- everything at once
# run_pipeline.py does the scheduling in Python: one 16 kHz cache pass, then every baseline side by side,
# then the U-Net alone with every core, all within the core budget Python detects
# (PIPELINE_ARGS="--cores N" to change it), then one scoring pass, then the viewer.
# Finished runs are skipped; FORCE=1 redoes them. Logs go to logs/<run>.log.
pipeline: unet-venv metrics-venv check-dataset ## Baselines in parallel → 3 U-Nets one by one → score all → viewer
	@if pgrep -f -- "--run-name unet_(28spk|56spk|both)( |$$)" >/dev/null; then \
	  echo "  ✗ a U-Net run is already training; not starting a second pipeline"; exit 1; fi
	$(PYTHON) run_pipeline.py --dataset "$(DATASET)" --port $(PORT) $(if $(FORCE),--force) $(PIPELINE_ARGS)

# Smoke runs are throwaway, so they are always redone: skipping them would test nothing.
pipeline-smoke: unet-venv metrics-venv check-dataset ## The pipeline on a few files (smoke_* runs, ~2 min, no viewer)
	$(PYTHON) run_pipeline.py --dataset "$(DATASET)" --smoke --skip-viewer --force $(PIPELINE_ARGS)

# Each step is its own make run, one after another; make stops at the first one that fails, so the
# hours-long pipeline only starts once the tests and the smoke pipeline have passed.
pipeline-all: ## Unit tests → smoke pipeline → full pipeline; stops at the first failure
	$(MAKE) test
	$(MAKE) pipeline-smoke
	$(MAKE) pipeline
	@echo "✅ tests, smoke pipeline and full pipeline all passed"

# ---------------------------------------------------------------------------- metric suite
# PESQ/STOI live in metric_suite/.venv so the main Python environment is left alone.
metrics-venv: check-python
	@test -x "$(METRICS)/$(METRICS_PY)" || { echo "  creating $(METRICS)/.venv"; \
	  cd $(METRICS) && $(PYTHON) -m venv .venv && $(METRICS_PY) -m pip install -q -r requirements.txt; }

evaluate: metrics-venv ## Score every model run: PESQ-WB, STOI, SNR gain vs noisy (METRIC_ARGS="--force" to rescore)
	cd $(METRICS) && $(METRICS_PY) evaluate_models.py $(METRIC_ARGS)
	@echo "  Report: $(METRICS)/results/summary.md"

figures-venv: check-python
	@test -x "$(FIGURES)/$(FIGURES_PY)" || { echo "  creating $(FIGURES)/.venv"; \
	  cd $(FIGURES) && $(PYTHON) -m venv .venv && $(FIGURES_PY) -m pip install -q -r requirements.txt; }

figures: figures-venv ## Results figures for the slides (figures/*.png) from the scored runs; run make evaluate first
	cd $(FIGURES) && $(FIGURES_PY) make_figures.py

train-evaluate: data evaluate ## Train/run all models (skipping finished runs), then evaluate them all

# ---------------------------------------------------------------------------- diagnostic viewer (Docker)
# DATA_DIR defaults to the project folder (see diagnostic_viewer/docker-compose.yml), so all runs show up.
viewer: viewer-up viewer-open ## Build, start and open the viewer in your browser

viewer-close: ## Shut the viewer down (stops and removes its container and network)
	$(COMPOSE) down --remove-orphans
	@echo "  Viewer stopped."

viewer-build: check-docker ## Build the viewer image
	$(COMPOSE) build

viewer-up: viewer-build ## Start the viewer and wait until it answers
	$(COMPOSE) up -d
	@printf "  waiting for $(URL) "; for i in $$(seq 1 60); do \
	  curl -sf "$(URL)/_stcore/health" >/dev/null && { echo "✓"; exit 0; }; printf "."; sleep 2; done; \
	  echo " ✗ not healthy, see: make viewer-logs"; exit 1
	@echo "  Viewer: $(URL)"

viewer-down: ## Stop the viewer
	$(COMPOSE) down

viewer-logs: ## Follow the viewer logs
	$(COMPOSE) logs -f

viewer-open: ## Open the viewer in the default browser
	open "$(URL)" 2>/dev/null || xdg-open "$(URL)"

# ---------------------------------------------------------------------------- model player (Docker)
# Any .wav (upload) or any Valentini train/test file through every trained model, outputs switchable
# live in lock-step. Mounts the project folder (trained runs) and $(DATASET) read-only. Port 8502, so
# it runs next to the viewer.
player: player-up player-open ## Build, start and open the model player in your browser

player-build: check-docker ## Build the model player image (CPU torch, ~1 GB the first time)
	$(PLAYER_COMPOSE) build

player-up: player-build ## Start the model player and wait until it answers
	@test -d "$(DATASET)" || echo "  ⚠ no dataset at DATASET='$(DATASET)': the player will offer uploads only (set it in local.mk)"
	$(PLAYER_COMPOSE) up -d
	@printf "  waiting for $(PLAYER_URL) "; for i in $$(seq 1 60); do \
	  curl -sf "$(PLAYER_URL)/_stcore/health" >/dev/null && { echo "✓"; exit 0; }; printf "."; sleep 2; done; \
	  echo " ✗ not healthy, see: make player-logs"; exit 1
	@echo "  Model player: $(PLAYER_URL)"

player-down: ## Stop the model player
	$(PLAYER_COMPOSE) down --remove-orphans

player-logs: ## Follow the model player logs
	$(PLAYER_COMPOSE) logs -f

player-open: ## Open the model player in the default browser
	open "$(PLAYER_URL)" 2>/dev/null || xdg-open "$(PLAYER_URL)"

test-player: player-build ## Model player tests, inside its Docker image (trained runs mounted)
	$(PLAYER_COMPOSE) run --rm --no-deps player python -m pytest -q -p no:cacheprovider

# ---------------------------------------------------------------------------- housekeeping
status: ## List collected runs (files per run) and git branches
	@for d in $(LINEAR)/outputs/* $(WIENER)/outputs/* $(UNET)/outputs/*; do \
	  [ -d "$$d" ] || continue; \
	  n=$$(ls "$$d"/enhanced "$$d"/processed 2>/dev/null | grep -c '\.wav$$'); \
	  printf "  %-45s %6s files\n" "$$d" "$$n"; done
	@git branch -vv

clean: ## Remove Python caches (never touches outputs or the dataset)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name .pytest_cache -type d -prune -exec rm -rf {} +
