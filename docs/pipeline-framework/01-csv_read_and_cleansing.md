# Objective: POLARS INGESTION
You are a Senior Data Engineer. Your task is to write the actual ingestion code.

**Input:** Read the `$ARTIFACTS_DIR/00_data_profile_report.md` file carefully. It contains the exact schema, pitfalls, and cleaning instructions for our dataset.

**Constraints:**
- Use strictly `polars`. No pandas.
- Use the Polars Lazy API (`pl.scan_csv()`) for memory-efficient execution, evaluating only at the final export step.
- Implement every recommended cleansing step from the `00_data_profile_report.md`.
- Use defensive programming (`strict=False` when casting, handle nulls explicitly, define an explicit `schema_overrides` if necessary).
- Include basic print statements logging the initial vs. final row counts and final schema.

**Output:** 
1. Write the python script to `$CODE_DIR/01_ingest.py`. 
2. The script must execute the cleansing pipeline and output the cleaned dataset to `$OUTPUT_DIR/interim_01_cleaned.parquet`. 
3. Write a `pytest`-compatible test file to `$CODE_DIR/tests/test_01_ingest.py`. The tests must validate:
   - The output file exists.
   - The schema matches expectations (e.g., Dates are actually datetime, not strings).
   - Expected null handling was applied correctly.
