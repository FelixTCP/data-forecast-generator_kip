import polars as pl

df = pl.read_parquet(r'C:\Users\T21728A\PRIVAT\data-forecast-generator_kip\output\20260524T203556Z\cleaned.parquet')
print(f"Columns: {df.columns}")
print(f"Schema: {df.schema}")
print(f"Date columns: {[col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]}")
print(f"First row:")
print(df.head(1))
