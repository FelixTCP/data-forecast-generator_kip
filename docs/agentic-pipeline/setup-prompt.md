# Setup Prompt (Agentic Mode)

Use this once to bootstrap or refresh generated runtime code for a run.

```markdown
You are executing the agentic single-agent regression pipeline for this repository.

Context sources (read before starting):
- docs/agentic-pipeline/contracts.md        ← code organisation rules, step-file contract, resume logic
- docs/agentic-pipeline/step-prompts.md     ← per-step reason/code/validate protocol
- docs/pipeline-framework/00-overview.md
- docs/pipeline-framework/10-csv-read-cleansing.md
- docs/pipeline-framework/11-data-exploration.md
- docs/pipeline-framework/12-feature-extraction.md
- docs/pipeline-framework/13-model-training.md
- docs/pipeline-framework/14-model-evaluation.md
- docs/pipeline-framework/15-model-selection.md
- docs/pipeline-framework/16-result-presentation.md

Core rules:
- One Python file per step (step_10_cleanse.py … step_16_report.py) plus orchestrator.py.
- Never write a monolithic pipeline script that runs all steps in one execution.
- Follow the Reason → Code → Validate protocol for every step (see step-prompts.md).
- Check resume state before executing each step; skip completed steps.
- Validate each step's output before proceeding to the next.
- Do not create planning or summary markdown files.
- All generated Python code lives under CODE_DIR.
- Enforce a strict no-leakage policy: causal features only; no target-at-t information may appear in predictors.
- Hard-fail the run if leakage checks fail. Do not continue to selection/reporting with suspect metrics.
```
