import csv
import json
from pathlib import Path

# Folder where this script is located
input_dir = Path(__file__).resolve().parent

# RAGDoll-style output
output_path = input_dir / "trecdl_2022_ragdoll.requests.jsonl"

csv_files = sorted(input_dir.glob("all_topics_trecdl_2022_part1.csv"))

print("Script folder:", input_dir)
print("CSV files found:", len(csv_files))

for file in csv_files:
    print("Found:", file.name)

# Store one record per query
grouped = {}
total_input_rows = 0

for csv_file in csv_files:
    file_count = 0
    print("\nReading:", csv_file.name)

    with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        print("Columns:", reader.fieldnames)

        required_columns = {"qid", "query", "pid", "passage", "relevance"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {csv_file.name}: {missing}")

        for row in reader:
            qid = row["qid"]
            query = row["query"]

            if qid not in grouped:
                grouped[qid] = {
                    "qid": qid,
                    "query": query,
                    "candidates": []
                }

            grouped[qid]["candidates"].append({
                "docid": row["pid"],
                "doc": {
                    "segment": row["passage"]
                },
                "metadata": {
                    "original_judgment": int(row["relevance"])
                }
            })

            file_count += 1
            total_input_rows += 1

    print("Rows read from this file:", file_count)

# Write RAGDoll JSONL
with output_path.open("w", encoding="utf-8") as out:
    for item in grouped.values():
        out.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\nDone.")
print("Total input rows read:", total_input_rows)
print("Total query rows written:", len(grouped))
print("Output file:", output_path)