# TASK: POLARS INGESTION
You are a Senior Data Engineer. Your task is to write the actual ingestion code.

**Input:** Read the `ATTENTION.md` file carefully. It contains the exact schema, pitfalls, and cleaning instructions for our dataset.
**Constraints:**
- Use strictly `polars`. No pandas.
- Implement every recommended cleansing step from the `ATTENTION.md`.
- Use defensive programming (`strict=False` when casting, handle nulls explicitly).

**Output:** Write the python script to `/src/01_ingest.py`. The script must output the cleaned dataframe to `/data/interim_01.parquet` and print `df.head(5)` to the terminal.