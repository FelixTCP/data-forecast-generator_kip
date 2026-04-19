# #00 Context Engineering: Data Profiling

## Objective

Safely inspect the CSV without loading it entirely into memory to avoid missing hidden anomalies, assess structural consistency, and generate a data profile report.

## Inputs

- CSV path
- Output directory
- Run ID

## Outputs

- `step-00_profiler.json`
- `step-00_data_profile_report.md` (Contains file metadata, delimiters, columns & guessed types, data anomalies, and recommended Polars cleansing steps)

## Guardrails

- Do not load the entire file into memory to avoid memory limits with large files.
- Catch inconsistent types or varying missing value representations.
- Check for footers or trailing summary rows.

## Copilot Prompt Snippet

```markdown
Implement a lightweight Python script to extract the exact header, first 5 rows (`head`), last 5 rows (`tail`), and a random sample of 10 rows. Get exact row count. Write output into `step-00_profiler.json`.
Assess structural consistency (e.g., identical delimiter count) and generate a `step-00_data_profile_report.md` outlining metadata, expected structure, columns, anomalies, and recommended Polars cleansing steps.
```

## Code Skeleton

```python
import json
import os

def generate_data_profile(csv_path: str, output_dir: str, run_id: str) -> None:
    # 1. Read file incrementally (head, tail, sample, line count)
    # 2. Extract metadata and identify anomalies
    # 3. Write step-00_profiler.json
    
    profiler_output_path = os.path.join(output_dir, "step-00_profiler.json")
    with open(profiler_output_path, "w") as f:
        json.dump({"metadata": "...", "head": [], "tail": [], "sample": []}, f)
        
    # 4. Write step-00_data_profile_report.md
    report_path = os.path.join(output_dir, "step-00_data_profile_report.md")
    with open(report_path, "w") as f:
        f.write("# Data Profile Report\n...")
```

## Tests

- very large csv file
- missing or unexpected delimiters
- trailing summary rows present
- inconsistent column types in random sample