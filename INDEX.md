# Documentation Index

Quick navigation for the Data Forecast Generator project.

## 📖 Start Here

**First time?** Read in this order:

1. **[QUICK_START.md](QUICK_START.md)** - 30-second setup & common commands
2. **[IMPLEMENTATION_OUTLINE.md](IMPLEMENTATION_OUTLINE.md)** - Module structure with pseudocode
3. **[plan.md](plan.md)** - Full 15-step roadmap (in session folder)

## 📚 Documentation Files

### Setup & Configuration
- **[QUICK_START.md](QUICK_START.md)** - Fast setup guide + common commands
- **[SETUP_SUMMARY.md](SETUP_SUMMARY.md)** - Detailed setup reference, troubleshooting
- **[README.md](README.md)** - Project overview & quick reference
- **[.env.example](.env.example)** - Environment variable template

### Implementation
- **[IMPLEMENTATION_OUTLINE.md](IMPLEMENTATION_OUTLINE.md)** - 8 modules with pseudocode & signatures
- **[plan.md](plan.md)** - Full 15-step Phase 1 roadmap (session folder)
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - Copilot guide

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
├── src/                    # Main source code (to be implemented)
├── tests/                  # Test suite (to be implemented)
├── examples/               # Customer example data
├── scripts/                # Utility scripts (setup.sh)
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
uv pip install -e ".[dev]"

# Development
pytest                                    # Run tests
pytest tests/test_module.py -v           # Specific test verbose
ruff format src tests                    # Format code
ruff check src tests                     # Lint check
mypy src                                 # Type check

# Coverage
pytest --cov=src --cov-report=html      # Generate coverage report
```

## 📋 Implementation Phases

### Phase 1 (MVP) - 8 Modules
1. **Data Loader** - CSV loading & validation
2. **CSV Analyzer** - LLM-based use case discovery
3. **Feature Engineer** - Feature preprocessing
4. **Regressor** - Model training
5. **Evaluator** - Model metrics & diagnostics
6. **Artifact Generator** - Model serialization
7. **Orchestrator** - End-to-end pipeline
8. **CLI** - User interface

See **[IMPLEMENTATION_OUTLINE.md](IMPLEMENTATION_OUTLINE.md)** for details.

### Phase 2 & 3
See **[plan.md](plan.md)** in session folder for future phases.

## 🔗 Key Sections by Purpose

### I want to...

**Set up my environment**
→ [QUICK_START.md](QUICK_START.md)

**Understand the architecture**
→ [.github/copilot-instructions.md](.github/copilot-instructions.md)

**Start implementing**
→ [IMPLEMENTATION_OUTLINE.md](IMPLEMENTATION_OUTLINE.md)

**See the full roadmap**
→ [plan.md](plan.md) (session folder)

**Debug setup issues**
→ [SETUP_SUMMARY.md](SETUP_SUMMARY.md) - Troubleshooting section

**Review what was created**
→ [FILES_CREATED.txt](FILES_CREATED.txt)

## ✅ What's Ready

- ✅ NixOS dev environment (flake.nix)
- ✅ Python project config (pyproject.toml)
- ✅ GitHub Actions CI/CD
- ✅ Project structure (src/, tests/)
- ✅ Comprehensive documentation
- ✅ Implementation roadmap

## 📝 Development Workflow

1. Create feature branch: `git checkout -b feat/module-name`
2. Write tests first: `tests/test_module_name.py`
3. Implement module: `src/module_name.py`
4. Run locally: `pytest tests/test_module_name.py`
5. Format code: `ruff format src tests`
6. Commit: `git commit -m "feat: description"`
7. Push: `git push origin feat/module-name`
8. Create PR → GitHub Actions validates automatically

See [QUICK_START.md](QUICK_START.md) for detailed workflow.

## 🎯 Success Criteria

- [ ] Nix environment set up: `nix develop`
- [ ] Dependencies installed: `uv pip install -e ".[dev]"`
- [ ] Tools verified: `pytest --version && ruff --version`
- [ ] Tests passing locally
- [ ] Code formatted: `ruff format src tests`
- [ ] Linting passes: `ruff check src tests`
- [ ] GitHub Actions passing on PRs
- [ ] 80%+ test coverage achieved

## 📞 Getting Help

1. **Setup issues?** → [SETUP_SUMMARY.md](SETUP_SUMMARY.md)
2. **Implementation questions?** → [IMPLEMENTATION_OUTLINE.md](IMPLEMENTATION_OUTLINE.md)
3. **Architecture?** → [.github/copilot-instructions.md](.github/copilot-instructions.md)
4. **Full details?** → [plan.md](plan.md)

---

Last updated: 2026-03-19

Next step: Read [QUICK_START.md](QUICK_START.md) →
