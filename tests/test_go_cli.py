from __future__ import annotations

import json
from pathlib import Path

import joblib
import polars as pl
import pytest
from sklearn.dummy import DummyRegressor

pytest.skip(
    "Historical forecast CLI tests; current workflow runs per-run step scripts under output/<RUN_ID>/code.",
    allow_module_level=True,
)

from data_forecast_generator import orchestrator
from data_forecast_generator.cli import main


def _fake_agent_runner_factory(run_root: Path):
    def _fake_runner(
        prompt: str,
        cwd: Path,
        *,
        copilot_model: str,
        reasoning_effort: str,
    ) -> str:
        del cwd
        assert copilot_model
        assert reasoning_effort

        def write_json(name: str, payload: dict) -> None:
            (run_root / name).write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )

        if "step-10-cleanse.json" in prompt:
            write_json("step-10-cleanse.json", {"ok": True})
        elif "step-11-exploration.json" in prompt:
            write_json("step-11-exploration.json", {"ok": True})
        elif "step-12-features.json" in prompt:
            write_json("step-12-features.json", {"ok": True})
        elif "step-13-training.json" in prompt:
            write_json("step-13-training.json", {"ok": True})
            model = DummyRegressor(strategy="mean")
            model.fit([[0.0], [1.0]], [0.0, 1.0])
            joblib.dump(model, run_root / "model.joblib")
            code_dir = None
            for line in prompt.splitlines():
                if line.startswith("Code workspace: "):
                    code_dir = Path(line.split("Code workspace: ", 1)[1].strip())
                    break
            if code_dir is not None:
                code_dir.mkdir(parents=True, exist_ok=True)
                (code_dir / "step13_model_training.py").write_text(
                    "from sklearn.dummy import DummyRegressor\\n",
                    encoding="utf-8",
                )
        elif "step-14-evaluation.json" in prompt:
            write_json(
                "step-14-evaluation.json",
                {
                    "selected_model": "ridge",
                    "metrics": {"r2": 0.7, "rmse": 10.0, "mae": 5.0},
                },
            )
        elif "step-15-selection.json" in prompt:
            write_json(
                "step-15-selection.json",
                {"selected_model": "ridge", "reason": "best r2"},
            )
        elif "step-16-report.md" in prompt:
            (run_root / "step-16-report.md").write_text("# report", encoding="utf-8")
        return "ok"

    return _fake_runner


def test_forecast_cli_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "input.csv"
    out_dir = tmp_path / "artifacts"

    df = pl.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5, 6, 7, 8],
            "feature_b": [2, 3, 4, 5, 6, 7, 8, 9],
            "target": [3, 5, 7, 9, 11, 13, 15, 17],
        }
    )
    df.write_csv(csv_path)

    run_id = "20260404T000000Z"
    run_dir = out_dir / run_id
    monkeypatch.setattr(orchestrator, "_iso_run_id", lambda: run_id)
    monkeypatch.setattr(
        orchestrator, "_run_copilot_prompt", _fake_agent_runner_factory(run_dir)
    )

    code = main(
        [
            "--csv",
            str(csv_path),
            "--output-dir",
            str(out_dir),
            "--target-column",
            "target",
        ]
    )

    assert code == 0

    expected = [
        "progress.json",
        "code_audit.json",
        "step-10-cleanse.json",
        "step-11-exploration.json",
        "step-12-features.json",
        "step-13-training.json",
        "model.joblib",
        "step-14-evaluation.json",
        "step-15-selection.json",
        "step-16-report.md",
    ]
    for name in expected:
        assert (run_dir / name).exists(), f"Missing artifact: {name}"

    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    assert "command_hash" in progress
    assert "code_dir" in progress

    evaluation = json.loads(
        (run_dir / "step-14-evaluation.json").read_text(encoding="utf-8")
    )
    assert "selected_model" in evaluation
    assert "metrics" in evaluation
    assert set(evaluation["metrics"]).issuperset({"r2", "rmse", "mae"})


def test_forecast_cli_passes_split_mode_to_agent_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "timeseries.csv"
    out_dir = tmp_path / "artifacts"

    df = pl.DataFrame(
        {
            "date": [
                "2016-01-01 00:00:00",
                "2016-01-01 00:10:00",
                "2016-01-01 00:20:00",
                "2016-01-01 00:30:00",
                "2016-01-01 00:40:00",
                "2016-01-01 00:50:00",
                "2016-01-01 01:00:00",
                "2016-01-01 01:10:00",
                "2016-01-01 01:20:00",
                "2016-01-01 01:30:00",
            ],
            "t1": [19.0, 19.1, 19.3, 19.2, 19.5, 19.7, 19.8, 20.0, 20.1, 20.2],
            "appliances": [50, 55, 53, 58, 62, 66, 70, 75, 80, 84],
        }
    )
    df.write_csv(csv_path)

    run_id = "20260404T000100Z"
    run_dir = out_dir / run_id
    seen_prompts: list[str] = []

    def fake_runner(
        prompt: str,
        cwd: Path,
        *,
        copilot_model: str,
        reasoning_effort: str,
    ) -> str:
        del cwd
        seen_prompts.append(prompt)
        return _fake_agent_runner_factory(run_dir)(
            prompt,
            tmp_path,
            copilot_model=copilot_model,
            reasoning_effort=reasoning_effort,
        )

    monkeypatch.setattr(orchestrator, "_iso_run_id", lambda: run_id)
    monkeypatch.setattr(orchestrator, "_run_copilot_prompt", fake_runner)

    code = main(
        [
            "--csv",
            str(csv_path),
            "--output-dir",
            str(out_dir),
            "--target-column",
            "appliances",
            "--split-mode",
            "auto",
        ]
    )
    assert code == 0

    training_prompts = [p for p in seen_prompts if "Run step 13" in p]
    assert training_prompts, "Expected step 13 training prompt"
    assert "Use split mode: auto." in training_prompts[0]


def test_forecast_cli_budget_mode_applies_low_cost_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "input.csv"
    out_dir = tmp_path / "artifacts"
    run_id = "20260404T000200Z"
    run_dir = out_dir / run_id

    df = pl.DataFrame({"x": [1, 2, 3, 4], "target": [2, 4, 6, 8]})
    df.write_csv(csv_path)

    captured: list[tuple[str, str]] = []

    def fake_runner_all_steps(
        prompt: str,
        cwd: Path,
        *,
        copilot_model: str,
        reasoning_effort: str,
    ) -> str:
        captured.append((copilot_model, reasoning_effort))
        return _fake_agent_runner_factory(run_dir)(
            prompt, cwd, copilot_model=copilot_model, reasoning_effort=reasoning_effort
        )

    monkeypatch.setattr(orchestrator, "_iso_run_id", lambda: run_id)
    monkeypatch.setattr(orchestrator, "_run_copilot_prompt", fake_runner_all_steps)

    code = main(
        [
            "--csv",
            str(csv_path),
            "--output-dir",
            str(out_dir),
            "--target-column",
            "target",
            "--budget-mode",
            "low",
        ]
    )
    assert code == 0
    assert captured
    assert all(model == "gpt-5-mini" for model, _ in captured)
    assert all(reasoning == "low" for _, reasoning in captured)


def test_forecast_cli_continue_reuses_hashed_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "input.csv"
    out_dir = tmp_path / "artifacts"
    df = pl.DataFrame({"x": [1, 2, 3, 4], "target": [2, 4, 6, 8]})
    df.write_csv(csv_path)

    run_id_1 = "20260409T000000Z"
    run_id_2 = "20260409T000100Z"
    run_dir_1 = out_dir / run_id_1
    run_dir_2 = out_dir / run_id_2

    run_ids = iter([run_id_1, run_id_2])
    monkeypatch.setattr(orchestrator, "_iso_run_id", lambda: next(run_ids))

    def fake_runner(
        prompt: str, cwd: Path, *, copilot_model: str, reasoning_effort: str
    ) -> str:
        del cwd
        del copilot_model
        del reasoning_effort
        if str(run_dir_1) in prompt:
            return _fake_agent_runner_factory(run_dir_1)(
                prompt, tmp_path, copilot_model="gpt-5-mini", reasoning_effort="low"
            )
        return _fake_agent_runner_factory(run_dir_2)(
            prompt, tmp_path, copilot_model="gpt-5-mini", reasoning_effort="low"
        )

    monkeypatch.setattr(orchestrator, "_run_copilot_prompt", fake_runner)

    code1 = main(
        [
            "--csv",
            str(csv_path),
            "--output-dir",
            str(out_dir),
            "--target-column",
            "target",
        ]
    )
    assert code1 == 0

    progress1 = json.loads((run_dir_1 / "progress.json").read_text(encoding="utf-8"))
    code_dir = Path(progress1["code_dir"])
    marker = code_dir / "carryover_marker.py"
    marker.write_text("# keep me\\n", encoding="utf-8")

    code2 = main(
        [
            "--csv",
            str(csv_path),
            "--output-dir",
            str(out_dir),
            "--target-column",
            "target",
            "--continue",
        ]
    )
    assert code2 == 0

    progress2 = json.loads((run_dir_2 / "progress.json").read_text(encoding="utf-8"))
    assert progress2["code_dir"] == progress1["code_dir"]
    assert (
        marker.exists()
    ), "Expected marker from previous workspace to remain with --continue"
