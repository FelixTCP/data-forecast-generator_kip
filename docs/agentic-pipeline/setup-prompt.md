# Setup Prompt (Agentic Mode)

Use this once to bootstrap or refresh generated runtime code for a run.

```markdown
You are executing the agentic single-agent regression pipeline for this repository.

Read these context sources before starting:
- docs/agentic-pipeline/contracts.md
- docs/agentic-pipeline/step-prompts.md
- docs/self-audit/overview.md
- docs/self-audit/audit-rules.md
- docs/self-audit/data-type-profiles.md
- docs/self-audit/remediation.md
- docs/pipeline-framework/00-pre_exploration.md
- docs/pipeline-framework/01-csv-read-cleansing.md
- docs/pipeline-framework/11-data-exploration.md
- docs/pipeline-framework/12-feature-extraction.md
- docs/pipeline-framework/13-model-training.md
- docs/pipeline-framework/14-model-evaluation.md
- docs/pipeline-framework/15-model-selection.md
- docs/pipeline-framework/16-result-presentation.md
- docs/pipeline-framework/17-critical-self-audit.md

Operating model:
- `docs/agentic-pipeline/*` defines runtime contracts and execution protocol.
- `docs/pipeline-framework/*` defines canonical step behavior and thresholds.
- If guidance conflicts, prefer `docs/pipeline-framework/*` for step logic.

Core rules:
- One Python file per step (`step_00_pre_exploration.py` … `step_16_report.py`) plus `orchestrator.py`.
- Never write a monolithic pipeline script that executes all steps in one file.
- **FORBIDDEN: Never copy, recycle, or reuse step scripts from previous run directories (`output/<OLD_RUN_ID>/code/`). Every script must be written fresh using file-creation tools (create_file, write_text, etc.). Violation makes the run invalid.**
- Step 17 (`step_17_audit.py`) is a required step and must always be generated — it is never optional.
- Follow the Reason → Code → Validate protocol for every step.
- **Resume behavior:**
  - If orchestrator is called WITH `--resume`: skip steps whose valid outputs already exist in OUTPUT_DIR.
  - If orchestrator is called WITHOUT `--resume`: execute ALL steps from scratch, ignoring prior artifacts. Old code must not be reused.
  - When starting a fresh run (same CSV, new RUN_ID), always omit the `--resume` flag.
- Validate each step output before proceeding.
- Do not create planning or summary markdown files.
- All generated Python code lives under `CODE_DIR`.
- Enforce strict no-leakage policy: causal features only; no target-at-t information in predictors.
- Hard-fail the run if leakage checks fail; do not proceed to selection/reporting with suspect metrics.
- **Abort/Cleanup:** If a run is interrupted (Ctrl+C, exception), the orchestrator must delete the entire OUTPUT_DIR before exiting. Partial artifacts are invalid and must not persist.
```
