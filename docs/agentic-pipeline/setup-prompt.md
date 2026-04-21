# Setup Prompt (Agentic Mode)

Use this once to bootstrap or refresh generated runtime code for a run.

```markdown
You are executing the agentic single-agent regression pipeline for this repository.

Read these context sources before starting:
- docs/agentic-pipeline/contracts.md
- docs/agentic-pipeline/step-prompts.md
- docs/pipeline-framework/10-csv-read-cleansing.md
- docs/pipeline-framework/11-data-exploration.md
- docs/pipeline-framework/12-feature-extraction.md
- docs/pipeline-framework/13-model-training.md
- docs/pipeline-framework/14-model-evaluation.md
- docs/pipeline-framework/15-model-selection.md
- docs/pipeline-framework/16-result-presentation.md

Operating model:
- `docs/agentic-pipeline/*` defines runtime contracts and execution protocol.
- `docs/pipeline-framework/*` defines canonical step behavior and thresholds.
- If guidance conflicts, prefer `docs/pipeline-framework/*` for step logic.

Core rules:
- One Python file per step (`step_10_cleanse.py` … `step_16_report.py`) plus `orchestrator.py`.
- Never write a monolithic pipeline script that executes all steps in one file.
- Follow the Reason → Code → Validate protocol for every step.
- Check resume state before executing each step; skip already-completed valid steps.
- Validate each step output before proceeding.
- Do not create planning or summary markdown files.
- All generated Python code lives under `CODE_DIR`.
- Enforce strict no-leakage policy: causal features only; no target-at-t information in predictors.
- Hard-fail the run if leakage checks fail; do not proceed to selection/reporting with suspect metrics.
```
