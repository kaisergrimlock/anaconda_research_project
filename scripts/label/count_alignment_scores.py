#!/usr/bin/env python3
import csv
from pathlib import Path
from collections import Counter

# ===== repo imports =====
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

def count_scores():
    base_dir = PROJECT_ROOT / "outputs/alignment_checker"
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return
        
    results = []
    all_known_scores = set()
    
    for file_path in base_dir.rglob("*.csv"):
        # Skip previously generated summaries or token usage logs
        if "token_usage" in file_path.name or "summary" in file_path.name:
            continue
            
        print(f"Processing {file_path.name}...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "alignment_score" not in reader.fieldnames:
                    print(f"  -> Skipping (no 'alignment_score' column)")
                    continue
                    
                counts = Counter()
                total_rows = 0
                for row in reader:
                    val = row.get("alignment_score", "").strip()
                    if not val:
                        val = "Empty/Missing"
                    counts[val] += 1
                    total_rows += 1
                
                row_data = {"filename": file_path.name, "total_rows": total_rows}
                row_data.update(counts)
                
                for k in counts.keys():
                    all_known_scores.add(k)
                    
                results.append(row_data)
        except Exception as e:
            print(f"  -> Error: {e}")
            
    if not results:
        print("No valid CSV files found containing 'alignment_score'.")
        return
        
    output_path = base_dir / "alignment_scores_summary.csv"
    
    # Ensure standard ordering of columns
    fieldnames = ["filename", "total_rows"] + sorted(list(all_known_scores))
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            # fill missing items with 0
            out_row = {k: r.get(k, 0) for k in fieldnames}
            out_row["filename"] = r["filename"]
            out_row["total_rows"] = r["total_rows"]
            writer.writerow(out_row)
            
    print(f"\nSummary successfully saved to: {output_path}")

if __name__ == "__main__":
    count_scores()
