# Objective: POLARS INGESTION
You are a Senior Data Engineer. Your task is to write the actual ingestion code.

**Input:** Read the `00-exploration.md` file carefully. It contains the exact schema, pitfalls, and cleaning instructions for our dataset.
**Constraints:**
- Use strictly `polars`. No pandas.
- Implement every recommended cleansing step from the `ATTENTION.md`.
- Use defensive programming (`strict=False` when casting, handle nulls explicitly).

**Output:** 
1. Write the python script to `/src/01_ingest.py`. The script must output the cleaned dataframe to `/data/interim_01.parquet`. 
2. Write necessary tests into "/tests" 