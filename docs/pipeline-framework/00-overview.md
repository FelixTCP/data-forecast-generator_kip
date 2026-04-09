# Single-Agent Pipeline Framework (Issue #8)

This framework covers the full MVP flow requested in [#8](https://github.com/FelixTCP/data-forecast-generator_kip/issues/8) and sub-issues [#10-#16](https://github.com/FelixTCP/data-forecast-generator_kip/issues/8).

## Goal

Create a reusable, instruction-first pipeline that works well with:

- GitHub Copilot-driven implementation
- single-agent execution
- multi-agent orchestration (LangChain, LangGraph, custom agents)

## Pipeline Steps

1. [`10-csv-read-cleansing.md`](./10-csv-read-cleansing.md)
2. [`11-data-exploration.md`](./11-data-exploration.md)
3. [`12-feature-extraction.md`](./12-feature-extraction.md)
4. [`13-model-training.md`](./13-model-training.md) **(deep focus)**
5. [`14-model-evaluation.md`](./14-model-evaluation.md)
6. [`15-model-selection.md`](./15-model-selection.md)
7. [`16-result-presentation.md`](./16-result-presentation.md)

## Canonical Data Contract

Use this contract between steps to stay agent-compatible and framework-agnostic.

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class PipelineContext:
    dataset_id: str
    target_column: str
    time_column: str | None
    features: list[str]
    split_strategy: dict[str, Any]
    model_candidates: list[dict[str, Any]]
    metrics: dict[str, float]
    artifacts: dict[str, str]
    notes: list[str]
```

## Shared Copilot Instruction Template

```markdown
# Task

Implement step <STEP_NAME> of the forecasting pipeline.

# Inputs

- `PipelineContext` (current state)
- DataFrame (`polars.DataFrame`)
- Config (`dict`)

# Output

- Updated `PipelineContext`
- Persisted artifacts for this step

# Constraints

- Use `polars` for data operations.
- Use `scikit-learn` for modeling.
- Add/extend tests in `tests/`.
- Keep deterministic behavior (`random_state` everywhere relevant).
- Emit structured logs with key metrics.

# Acceptance Criteria

- Step runs standalone.
- Step integrates into orchestrator flow.
- Error handling is explicit and actionable.
```

## Done Criteria for MVP

- Each step has: objective, input/output contract, guardrails, code skeleton, and tests.
- Model training includes reproducible split, caching, history/logging, and hyperparameter search path.
- Evaluation and selection are traceable and reproducible.
