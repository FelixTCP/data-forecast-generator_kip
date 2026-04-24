from __future__ import annotations

import tomllib
from pathlib import Path


def test_pyproject_declares_run_based_runtime_dependencies() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["tool"]["uv"]["package"] is False
    assert "forecast" not in pyproject["project"].get("scripts", {})

    dependencies = set(pyproject["project"]["dependencies"])
    required_prefixes = {
        "polars",
        "pandas",
        "pyarrow",
        "statsmodels",
        "scipy",
        "scikit-learn",
        "joblib",
    }
    assert all(
        any(dependency.startswith(prefix) for dependency in dependencies)
        for prefix in required_prefixes
    )
