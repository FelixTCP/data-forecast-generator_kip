# TASK: DATA PROFILING
You are a Senior Data Analyst. We need to safely inspect the file `/data/appliances_energy_prediction.csv` without loading it entirely into memory.

**Step 1:** Write a tiny, temporary python script or use a bash command (like `head -n 10`) to extract the exact header and the first 5 rows of the CSV.
**Step 2:** Analyze this raw text output. 
**Step 3:** Create a file named `ATTENTION.md` in the `/steps` folder. 

The `ATTENTION.md` MUST contain:
- **Expected Delimiter:** (e.g., comma, semicolon, tab)
- **Detected Columns & Guessed Types:** List them.
- **Data Anomalies & Pitfalls:** (e.g., "Row 3 has a missing value", "The 'price' column uses a comma as a decimal separator", "Dates are in DD.MM.YYYY format").
- **Recommended Polars Cleansing Steps:** Bullet points on how to safely cast these specific columns.

When done continue with `01-data_reading_and_cleaning.md`.