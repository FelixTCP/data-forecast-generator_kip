# START HERE

This file points to the current run-based workflow.

## ✅ What Was Set Up

A Python development environment with:
- **NixOS flake.nix** - Reproducible dev environment
- **uv + ruff + pytest** - Fast, modern Python tooling
- **GitHub Actions CI/CD** - Automatic testing on every push
- **Current system documentation** - see `CURRENT_SYSTEM_DOCUMENTATION.md`
- **Run-based pipeline structure** - step scripts and artifacts under `output/<RUN_ID>/`

**Total setup**: 13 configuration/documentation files

## 🚀 Quick Start (30 seconds)

```bash
# Run from the repository root
nix develop
uv sync --extra dev
uv pip install pandas statsmodels scipy pyarrow
```

Done! ✓ You now have Python 3.12 + all tools + all dependencies.

## 📚 What to Read Next

**Choose based on your need:**

| I want to... | Read this |
|---|---|
| Quick overview of everything | **[INDEX.md](INDEX.md)** (2 min) |
| Setup & common commands | **[QUICK_START.md](QUICK_START.md)** (3 min) |
| Understand current workflow | **[CURRENT_SYSTEM_DOCUMENTATION.md](CURRENT_SYSTEM_DOCUMENTATION.md)** (10 min) |
| See historical module outline | **[IMPLEMENTATION_OUTLINE.md](IMPLEMENTATION_OUTLINE.md)** (10 min) |
| Architecture & conventions | **[.github/copilot-instructions.md](.github/copilot-instructions.md)** (5 min) |
| Detailed setup reference | **[SETUP_SUMMARY.md](SETUP_SUMMARY.md)** (10 min) |

## 🎯 Recommended Path

**First time?** Follow this order:

1. ✅ Setup: `nix develop` + `uv sync --extra dev`
2. → Read **[INDEX.md](INDEX.md)** (quick navigation guide)
3. → Read **[QUICK_START.md](QUICK_START.md)** (setup & commands)
4. → Read **[CURRENT_SYSTEM_DOCUMENTATION.md](CURRENT_SYSTEM_DOCUMENTATION.md)** (current run model)
5. → **Run the pipeline** with a new `output/<RUN_ID>/` directory

Total time: ~15 minutes → ready to code

## 💻 Verify Everything Works

```bash
pytest --version
ruff --version
mypy --version
python --version
```

All should show version numbers. If any fails, see [SETUP_SUMMARY.md](SETUP_SUMMARY.md).

## 🔧 Set Up Your Environment

```bash
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY from https://console.anthropic.com
```

## 📂 Current Run Structure

```
output/<RUN_ID>/
├── code/                      # Step scripts for this run
├── progress.json
├── cleaned.parquet
├── features.parquet
├── model.joblib
├── holdout.npz
├── step-*.json
└── step-16-report.md
```

## Run the Pipeline

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

## 🔄 Development Workflow

For a pipeline change:

```bash
# 1. Create feature branch
git checkout -b feat/pipeline-step

# 2. Run a fresh pipeline once
# See CURRENT_SYSTEM_DOCUMENTATION.md for the full command

# 3. Adjust the relevant step script
vim output/<RUN_ID>/code/step_13_training.py
pytest

# 4. Format & lint
ruff format tests scripts
ruff check tests scripts

# 5. Commit & push
git add .
git commit -m "feat: adjust pipeline step"
git push origin feat/pipeline-step

# 6. Create PR on GitHub
# → GitHub Actions runs automatically
# → Tests pass ✓
# → Merge
```

## 📋 Commands You'll Use

```bash
# Testing
pytest                                # Run all tests
pytest tests/test_data_loader.py     # Specific file
pytest                               # Run current tests

# Code quality
ruff check tests scripts             # Lint check
ruff format tests scripts            # Auto-format code
mypy .                               # Type check

# Development
nix develop                          # Enter dev environment
nix flake update                     # Update dependencies
uv sync --extra dev                  # Sync environment
```

## 🎯 Success Criteria

- ✓ Setup works: `pytest --version` shows 7.4+
- ✓ Dependencies installed: `import sklearn` works
- ✓ Tests pass locally before pushing
- ✓ Code formatted: `ruff format tests scripts`
- ✓ Linting passes: `ruff check tests scripts`
- ✓ GitHub Actions passes on PR
- ✓ 80%+ test coverage achieved
- ✓ Commit message: "feat: description"

## 📞 Need Help?

**Setup issues?** → See [SETUP_SUMMARY.md](SETUP_SUMMARY.md) - Troubleshooting

**Current workflow?** → See [CURRENT_SYSTEM_DOCUMENTATION.md](CURRENT_SYSTEM_DOCUMENTATION.md)

**Architecture?** → See [.github/copilot-instructions.md](.github/copilot-instructions.md)

**Lost?** → Start with [INDEX.md](INDEX.md) - Navigation guide

## 🚀 You're Ready!

Everything is set up for the current run-based workflow.

**Next step:** Read [CURRENT_SYSTEM_DOCUMENTATION.md](CURRENT_SYSTEM_DOCUMENTATION.md), then run the pipeline.

Good luck! 🎉

---

*For the complete setup breakdown, see [FILES_CREATED.txt](FILES_CREATED.txt)*
