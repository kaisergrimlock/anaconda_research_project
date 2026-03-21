import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
rows_per_part = 500
base_name = "all_topics_trecdl_2021"

csv_files = list(SCRIPT_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV files found in script folder.")
if len(csv_files) > 1:
    raise RuntimeError(f"Multiple CSV files found: {[f.name for f in csv_files]}")

input_file = csv_files[0]
df = pd.read_csv(input_file)

for i in range(0, len(df), rows_per_part):
    part_number = (i // rows_per_part) + 1
    part_df = df.iloc[i:i + rows_per_part]
    output_file = SCRIPT_DIR / f"{base_name}_part{part_number}.csv"
    part_df.to_csv(output_file, index=False)

print(f"Done. Split {input_file.name} into {part_number} part(s).")