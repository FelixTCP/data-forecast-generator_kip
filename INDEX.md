# Documentation Index

Quick navigation for the Data Forecast Generator project.

## 📖 Start Here

**First time?** Read in this order:

1. **[QUICK_START.md](QUICK_START.md)** - Setup & run command
2. **[CURRENT_SYSTEM_DOCUMENTATION.md](CURRENT_SYSTEM_DOCUMENTATION.md)** - Current run-based workflow
3. **[IMPLEMENTATION_OUTLINE.md](IMPLEMENTATION_OUTLINE.md)** - Historical module outline

## 📚 Documentation Files

### Setup & Configuration
- **[QUICK_START.md](QUICK_START.md)** - Fast setup guide + common commands
- **[SETUP_SUMMARY.md](SETUP_SUMMARY.md)** - Detailed setup reference, troubleshooting
- **[README.md](README.md)** - Project overview & quick reference
- **[.env.example](.env.example)** - Environment variable template

### Implementation
- **[CURRENT_SYSTEM_DOCUMENTATION.md](CURRENT_SYSTEM_DOCUMENTATION.md)** - Current runtime model, commands, artifacts, and validation gates
- **[IMPLEMENTATION_OUTLINE.md](IMPLEMENTATION_OUTLINE.md)** - Historical 8-module outline
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - Copilot guide
- **[docs/agentic-pipeline/contracts.md](docs/agentic-pipeline/contracts.md)** - Single-agent runtime contracts
- **[docs/agentic-pipeline/step-prompts.md](docs/agentic-pipeline/step-prompts.md)** - Runtime step protocol wrappers
- **[docs/pipeline-framework/00-pre_exploration.md](docs/pipeline-framework/00-pre_exploration.md)**, **[01-csv-read-cleansing.md](docs/pipeline-framework/01-csv-read-cleansing.md)**, **[11-data-exploration.md](docs/pipeline-framework/11-data-exploration.md)** ... **[16-result-presentation.md](docs/pipeline-framework/16-result-presentation.md)** - Canonical per-step specs

### Reference
- **[FILES_CREATED.txt](FILES_CREATED.txt)** - Inventory of all created files
- **[project-description.md](project-description.md)** - Original project specification

## 🛠️ Configuration Files

- **[flake.nix](flake.nix)** - NixOS dev environment
- **[pyproject.toml](pyproject.toml)** - Python project config (uv, ruff, pytest, etc.)
- **[.github/workflows/tests.yml](.github/workflows/tests.yml)** - GitHub Actions CI/CD
- **[.gitignore](.gitignore)** - Git ignore rules

## 📂 Directory Structure

```
.
├── data/                   # Example CSV inputs
├── output/                 # Run directories and artifacts
│   └── <RUN_ID>/code/      # Step scripts for each run
├── tests/                  # Test suite
├── examples/               # Customer example data
├── scripts/                # Utility scripts
├── .github/
│   ├── workflows/         # CI/CD pipelines
│   └── copilot-instructions.md
├── flake.nix              # NixOS environment
├── pyproject.toml         # Python config
├── README.md
├── QUICK_START.md
├── IMPLEMENTATION_OUTLINE.md
├── SETUP_SUMMARY.md
├── FILES_CREATED.txt
├── INDEX.md (this file)
└── .env.example
```

## 🚀 Quick Commands

```bash
# Setup
nix develop
uv sync --extra dev
uv pip install pandas statsmodels scipy pyarrow

# Development
pytest                                    # Run tests
pytest tests/test_module.py -v           # Specific test verbose
ruff format tests scripts                # Format code
ruff check tests scripts                 # Lint check
mypy .                                   # Type check

# Coverage / tests
pytest                                  # Run current tests
```

## 📋 Current Pipeline Steps

1. `00-pre-exploration`
2. `01-csv-read-cleansing`
3. `11-data-exploration`
4. `12-feature-extraction`
5. `13-model-training`
6. `14-model-evaluation`
7. `15-model-selection`
8. `16-result-presentation`

See **[CURRENT_SYSTEM_DOCUMENTATION.md](CURRENT_SYSTEM_DOCUMENTATION.md)** for details.

## 🔗 Key Sections by Purpose

### I want to...

**Set up my environment**
→ [QUICK_START.md](QUICK_START.md)

**Understand the architecture**
→ [CURRENT_SYSTEM_DOCUMENTATION.md](CURRENT_SYSTEM_DOCUMENTATION.md)

**Run the pipeline**
→ [QUICK_START.md](QUICK_START.md)

**Debug setup issues**
→ [SETUP_SUMMARY.md](SETUP_SUMMARY.md) - Troubleshooting section

**Review what was created**
→ [FILES_CREATED.txt](FILES_CREATED.txt)

## ✅ What's Ready

- ✅ NixOS dev environment (flake.nix)
- ✅ Python project config (pyproject.toml)
- ✅ GitHub Actions CI/CD
- ✅ Run-based pipeline structure (`output/<RUN_ID>/code`, artifacts)
- ✅ Comprehensive documentation
- ✅ Current run documentation

## 📝 Development Workflow

1. Create feature branch: `git checkout -b feat/module-name`
2. Edit the relevant step script under `output/<RUN_ID>/code/`
3. Add or adjust tests where useful
4. Run locally: `pytest tests/test_module_name.py`
5. Format code: `ruff format tests scripts`
6. Commit: `git commit -m "feat: description"`
7. Push: `git push origin feat/module-name`
8. Create PR → GitHub Actions validates automatically

See [QUICK_START.md](QUICK_START.md) for detailed workflow.

## 🎯 Success Criteria

- [ ] Nix environment set up: `nix develop`
- [ ] Dependencies installed: `uv sync --extra dev`
- [ ] Tools verified: `pytest --version && ruff --version`
- [ ] Tests passing locally
- [ ] Code formatted: `ruff format tests scripts`
- [ ] Linting passes: `ruff check tests scripts`
- [ ] GitHub Actions passing on PRs
- [ ] 80%+ test coverage achieved

## 📞 Getting Help

1. **Setup issues?** → [SETUP_SUMMARY.md](SETUP_SUMMARY.md)
2. **Current workflow?** → [CURRENT_SYSTEM_DOCUMENTATION.md](CURRENT_SYSTEM_DOCUMENTATION.md)
3. **Architecture?** → [CURRENT_SYSTEM_DOCUMENTATION.md](CURRENT_SYSTEM_DOCUMENTATION.md)

---

Last updated: 2026-03-19

Next step: Read [QUICK_START.md](QUICK_START.md) →
