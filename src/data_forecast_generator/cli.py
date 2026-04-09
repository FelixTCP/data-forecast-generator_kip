from __future__ import annotations

import argparse
import json
import sys

from data_forecast_generator.orchestrator import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forecast", description="Run forecast pipeline end-to-end"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="forecast",
        choices=["forecast"],
        help="Run full pipeline",
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument(
        "--output-dir", default="artifacts", help="Output directory for run artifacts"
    )
    parser.add_argument("--target-column", default=None, help="Optional target column")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test size fraction"
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--split-mode",
        default="auto",
        choices=["auto", "random", "time_series"],
        help="Data split strategy: auto detects time column and uses chronological holdout",
    )
    parser.add_argument(
        "--budget-mode",
        default="low",
        choices=["low", "balanced", "high"],
        help="Copilot runtime budget profile (low = cheapest/fastest)",
    )
    parser.add_argument(
        "--copilot-model",
        default=None,
        help="Optional explicit Copilot model override (e.g. gpt-5-mini, claude-haiku-4.5)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["low", "medium", "high", "xhigh"],
        help="Optional reasoning effort override",
    )
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Reuse hashed code workspace from previous identical command fingerprint",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    result = run_pipeline(
        csv_path=args.csv,
        output_dir=args.output_dir,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
        split_mode=args.split_mode,
        budget_mode=args.budget_mode,
        copilot_model=args.copilot_model,
        reasoning_effort=args.reasoning_effort,
        continue_run=args.continue_run,
    )

    print("✅ Pipeline finished")
    print(f"Run ID: {result.run_id}")
    print(f"Output: {result.output_dir}")
    print(
        json.dumps(
            {"selected_model": result.selected_model, "metrics": result.metrics},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
