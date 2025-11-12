#!/usr/bin/env python3
import csv, sys
from pathlib import Path

# --- big-field CSV safety (no external import needed) ---
def bump_field_limit():
    # Robustly push the csv field size limit as high as this platform allows
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)

bump_field_limit()

# --- script-relative paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
FILENAME   = "all_topics_trecdl_2023_part46_labels_openai.gpt-oss-20b-1_0.csv"

def find_input(script_dir: Path, name: str) -> Path:
    exact = script_dir / name
    if exact.exists():
        return exact
    # help diagnose if the name is slightly off
    candidates = sorted(script_dir.glob("*.csv"))
    print(f"[INFO] Looking in: {script_dir}")
    if candidates:
        print("[INFO] CSVs in this folder:")
        for c in candidates:
            print("   -", c.name)
    else:
        print("[INFO] No CSV files found in this folder.")
    # loose match by prefix
    stem = name[:-4] if name.lower().endswith(".csv") else name
    prefix_matches = [c for c in candidates if c.stem.startswith(stem)]
    if len(prefix_matches) == 1:
        print(f"[INFO] Using closest match: {prefix_matches[0].name}")
        return prefix_matches[0]
    sys.exit(f"[ERROR] File not found next to script: {name}")

INPUT_CSV  = find_input(SCRIPT_DIR, FILENAME)
OUTPUT_CSV = INPUT_CSV.with_name(f"{INPUT_CSV.stem}_clean.csv")

TARGET_COLS = ["qid", "query", "pid_qrels", "pid_resolved", "passage", "relevance", "llm_relevance"]

print(f"[INFO] Script dir  : {SCRIPT_DIR}")
print(f"[INFO] Input file  : {INPUT_CSV.name}")
print(f"[INFO] Output file : {OUTPUT_CSV.name}")

with INPUT_CSV.open("r", encoding="utf-8", newline="") as fin, \
     OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fout:

    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=TARGET_COLS)
    writer.writeheader()

    for row in reader:
        writer.writerow({
            "qid":           row.get("qid", ""),
            "query":         row.get("query", row.get("query_raw", "")),
            "pid_qrels":     row.get("pid_qrels", ""),
            "pid_resolved":  row.get("pid_resolved", ""),
            "passage":       row.get("passage", row.get("passage_injected", "")),
            "relevance":     row.get("relevance", ""),
            "llm_relevance": row.get("llm_relevance", ""),
        })

print(f"✅ Converted {INPUT_CSV.name} → {OUTPUT_CSV.name}")
