# Setup Summary

Current runtime details: see [CURRENT_SYSTEM_DOCUMENTATION.md](CURRENT_SYSTEM_DOCUMENTATION.md).

## What's Been Created

### 1. **Nix Development Environment** (`flake.nix`)
- Python 3.12 with all scientific packages pre-configured
- Tools included: uv, ruff, pytest, mypy, isort
- Dependencies: scikit-learn, polars, numpy, anthropic, fastapi, uvicorn
- GitHub Copilot CLI included for development

**Enter environment**:
```bash
nix develop
```

### 2. **Python Project Configuration** (`pyproject.toml`)
- uv-compatible package management
- Ruff configuration (100 char lines, Python 3.12)
- Pytest markers (unit, integration, slow)
- Black formatting config
- MyPy type checking setup
- Optional dev dependencies (pytest-cov, mypy, etc.)

### 3. **Code Quality & Linting**
- **Ruff**: Fast Python linter + formatter
- **MyPy**: Static type checking (configured, non-strict MVP)
- **isort**: Import sorting
- **Pytest**: Testing framework with coverage

### 4. **CI/CD Pipeline** (`.github/workflows/tests.yml`)
- Runs on: push to main/develop, all PRs
- Tests on Python 3.12
- Steps:
  1. Lint with ruff (check + format)
  2. Type check with mypy
  3. Run tests with pytest + coverage
  4. Upload coverage to Codecov

### 5. **Project Structure**
```
.
├── tests/                  # Test suite
│   └── test_*.py          # Test files
├── data/                   # Example CSV inputs
├── output/                 # Run directories and artifacts
│   └── <RUN_ID>/
│       ├── code/           # Step scripts for this run
│       ├── model.joblib
│       └── step-*.json
├── examples/              # Customer examples
├── scripts/               # Utility scripts
├── .github/
│   ├── workflows/         # CI/CD
│   └── copilot-instructions.md  # This project's Copilot guide
├── flake.nix             # Nix dev environment
├── pyproject.toml        # Python project config
├── .gitignore            # Git ignore rules
├── .env.example          # Environment template
└── README.md             # Updated with setup info
```

### 6. **Copilot Instructions** (`.github/copilot-instructions.md`)
- Architecture overview
- Expected modules and organization
- Key conventions and patterns
- Common workflows

## Getting Started

### First Time Setup
```bash
# Enter dev environment
nix develop

# Install dependencies used by the run-based pipeline
uv sync --extra dev
uv pip install pandas statsmodels scipy pyarrow

# Verify tools work
pytest --version
ruff --version
mypy --version
```

### Running Commands

**Inside `nix develop` shell**:
```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_data_loader.py::test_valid_csv

# Lint
ruff check tests scripts

# Format
ruff format tests scripts

# Type check
mypy .

# Coverage report
pytest
```

### Development Workflow

1. **Create feature branch**: `git checkout -b feat/my-feature`
2. **Run or adjust the step scripts in `output/<RUN_ID>/code/`**
3. **Write tests in `tests/test_*.py` where useful**
4. **Run locally**:
   ```bash
   pytest tests/test_my_feature.py
   ruff format tests scripts
   mypy .
   ```
5. **Commit**: `git commit -m "feat: description"`
6. **Push**: `git push origin feat/my-feature`
7. **Create PR** → GitHub Actions runs automatically

### Configuration

Copy environment template:
```bash
cp .env.example .env
```

Edit `.env` with your:
- `ANTHROPIC_API_KEY`: Get from https://console.anthropic.com
- `ARTIFACT_OUTPUT_DIR`: Where to save models
- Other settings as needed

## Current Pipeline Entry Point

Create a new run directory, copy the current step scripts, then run the orchestrator:

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

## Key Files Reference

| File | Purpose |
|------|---------|
| `flake.nix` | NixOS dev environment definition |
| `pyproject.toml` | Python project config (uv, pytest, ruff, etc.) |
| `.github/workflows/tests.yml` | CI/CD pipeline |
| `.github/copilot-instructions.md` | Copilot project guide |
| `.env.example` | Environment variable template |
| `.gitignore` | Git ignore rules |
| `README.md` | Quick start guide |
| `CURRENT_SYSTEM_DOCUMENTATION.md` | Current run-based system documentation |

## Tools & Versions

- **Python**: 3.12
- **Ruff**: Latest (linter + formatter)
- **Pytest**: 7.4+
- **MyPy**: 1.5+ (optional, non-strict)
- **scikit-learn**: 1.3+
- **polars**: 2.0+
- **anthropic**: 0.7+ (for Claude)
- **fastapi**: 0.100+ (Phase 3)

## Next Steps

1. ✅ Enter dev environment: `nix develop`
2. ✅ Install dependencies: `uv sync --extra dev`
3. ✅ Verify setup: `pytest --version`
4. ➜ Run the pipeline from `output/<RUN_ID>/code`
5. ➜ Inspect `progress.json`, `step-*.json`, `model.joblib`, and `step-16-report.md`
6. ➜ Push to GitHub when ready

## Troubleshooting

**"nix: command not found"**
- Install Nix: https://nixos.org/download.html

**"uv: command not found" inside nix develop**
- Flake not updated: `nix flake update`
- Shell not properly entered: `exit` then `nix develop` again

**Pytest not finding tests**
- Check `tests/__init__.py` exists
- Run from the repository root: `pytest`

**Missing runtime dependency**
- Run `uv sync --extra dev`
- Then run `uv pip install pandas statsmodels scipy pyarrow`

## Questions?

Refer to:
- `.github/copilot-instructions.md` - Architecture & conventions
- `CURRENT_SYSTEM_DOCUMENTATION.md` - Current runtime model and run commands
- `README.md` - Quick reference
