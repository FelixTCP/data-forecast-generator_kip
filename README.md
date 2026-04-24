# Data Forecast Generator

Automated discovery and development of regression forecasting models from customer CSV data using LLM-driven analysis.

Current system details: see [CURRENT_SYSTEM_DOCUMENTATION.md](CURRENT_SYSTEM_DOCUMENTATION.md).

## Quick Start

### Prerequisites

- NixOS with `nix flake` support
- Git

### Setup (NixOS)

```bash
# Enter development environment
nix flake update
nix develop

# Prepare dependencies
uv sync --extra dev
uv pip install pandas statsmodels scipy pyarrow

# Verify installation
pytest --version
ruff --version
```

### Setup (Other Systems)

Python 3.12+ with pip:

```bash
python -m venv .venv
source .venv/bin/activate
uv sync --extra dev
uv pip install pandas statsmodels scipy pyarrow
```

## Development

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_data_loader.py

# Specific test
pytest tests/test_data_loader.py::test_valid_csv

# With coverage
pytest
```

### Code Quality

```bash
# Lint
ruff check tests scripts

# Format
ruff format tests scripts

# Type check
mypy .
```

### Local Development Loop

```bash
# Make changes, then:
pytest tests/test_module.py -v
ruff format tests scripts
git add . && git commit -m "feat: ..."
```

### Run the pipeline

```bash
RUN_ID="singleagent_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="output/$RUN_ID"
mkdir -p "$OUT/code"
cp output/manual_run_001/code/*.py "$OUT/code/"

uv run python "$OUT/code/orchestrator.py" \
  --csv-path data/appliances_energy_prediction.csv \
  --target-column appliances \
  --output-dir "$OUT" \
  --run-id "$RUN_ID" \
  --split-mode auto
```

The run writes its code, progress file, model artifacts, JSON step outputs, leakage audit, and final report under `output/<RUN_ID>/`.

### Quick inference check for `model.joblib`

```bash
# Predict with model artifact
uv run python scripts/infer_model.py \
  --run-dir output/<RUN_ID> \
  --csv ./data/appliances_energy_prediction.csv \
  --target-column appliances
```

## Architecture

See [CURRENT_SYSTEM_DOCUMENTATION.md](CURRENT_SYSTEM_DOCUMENTATION.md) for the current architecture and run contract.

**Current core runtime**:

- `output/<RUN_ID>/code/` - one standalone Python script per pipeline step
- `runtime_utils.py` - shared progress, JSON, and code-audit helpers
- `orchestrator.py` - thin runner that executes steps in order
- `scripts/` - post-run inference and plotting utilities

**Pipeline framework docs (Issue #8, sub-issues #10-#16):**

- `docs/agentic-pipeline/contracts.md` - Runtime contracts, file layout, and resume rules
- `docs/agentic-pipeline/step-prompts.md` - Runtime Reason→Code→Validate wrappers per step
- `docs/pipeline-framework/00-pre_exploration.md`, `01-csv-read-cleansing.md`, `11-data-exploration.md` ... `16-result-presentation.md` - Canonical per-step logic

## Configuration

Copy `.env.example` to `.env` and update with your API keys:

```bash
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY
```

## Project Phases

- **Phase 1 (MVP)**: Core pipeline - CSV → Analysis → Model → Artifact
- **Phase 2**: Quality metrics & LLM-as-Judge evaluation
- **Phase 3**: Full-stack system with FastAPI + Web UI

## CI/CD

GitHub Actions runs on every push:

- Lint with ruff
- Type check with mypy
- Test with pytest (Python 3.12)
- Coverage reporting to Codecov

See `.github/workflows/tests.yml`

## Contributing

1. Create feature branch: `git checkout -b feat/my-feature`
2. Implement with tests
3. Run `pytest` and `ruff format tests scripts`
4. Push and create PR
5. CI/CD validates automatically

## License

See LICENSE file
