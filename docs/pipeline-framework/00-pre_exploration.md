# TASK: DATA PROFILING
You are a Senior Data Analyst. We need to safely inspect the $CSV_FILE without loading it entirely into memory to avoid missing hidden anomalies.

**Step 1:** Write a tiny, temporary python script into $OUTPUT_DIR
- Extract the exact header and the first 5 rows (`head`).
- Extract the last 5 rows (`tail`) to check for footers or trailing summary rows.
- Extract a random sample of 10 rows to catch inconsistent types or varying missing value representations.
- Get the exact row count 
- Write output into 00_profiler_output.json

**Step 2:** Analyze this raw text output and assess structural consistency (e.g., ensure the delimiter count is identical across all sampled rows).

**Step 3:** Create a file named `00_data_profile_report.md` in the $ARTIFACTS_DIR folder. 

The `00_data_profile_report.md` MUST contain:
- **File Metadata:** File encoding natively detected, estimated total rows, and file size.
- **Expected Delimiter & Structural Integrity:** (e.g., comma, semicolon, tab, and whether delimiters are evenly distributed per row).
- **Detected Columns & Guessed Types:** List them based on the diverse sample.
- **Data Anomalies & Pitfalls:** (e.g., "Row 99 has a missing value represented as `\N`", "The 'price' column uses a comma as a decimal separator", "Strings contain unescaped newline characters", "Trailing blank lines exist at the end of the file").
- **Recommended Polars Cleansing Steps:** Bullet points on how to safely cast these specific columns and handle the identified anomalies using Polars lazy evaluation where possible.

When done continue with `01-csv_read_and_cleansing.md`.