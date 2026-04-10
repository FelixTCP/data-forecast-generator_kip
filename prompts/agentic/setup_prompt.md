You are executing the agentic single-agent regression pipeline for this repository.

Read and follow:
- docs/agentic-pipeline/contracts.md
- docs/pipeline-framework/00-overview.md
- docs/pipeline-framework/10-csv-read-cleansing.md
- docs/pipeline-framework/11-data-exploration.md
- docs/pipeline-framework/12-feature-extraction.md
- docs/pipeline-framework/13-model-training.md
- docs/pipeline-framework/14-model-evaluation.md
- docs/pipeline-framework/15-model-selection.md
- docs/pipeline-framework/16-result-presentation.md

Runtime variables:
- CSV_PATH={{CSV_PATH}}
- TARGET_COLUMN={{TARGET_COLUMN}}
- OUTPUT_DIR={{OUTPUT_DIR}}
- CODE_DIR={{CODE_DIR}}
- SPLIT_MODE={{SPLIT_MODE}}

Task:
Initialize/refresh runtime code under CODE_DIR for this run. Keep code deterministic and debuggable.
