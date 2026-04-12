#!/usr/bin/env python3
import csv
from pathlib import Path

# ===== repo imports =====
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]


def should_skip_file(file_path: Path) -> bool:
    """Return True if this CSV should not be processed."""
    name = file_path.name.lower()

    skip_keywords = [
        "token_usage",
        "summary",
        "totals",
        "all_topics",
    ]

    return any(keyword in name for keyword in skip_keywords)


def sum_scores():
    base_dir = PROJECT_ROOT / "outputs/alignment_checker"
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return

    results = []

    for file_path in base_dir.rglob("*.csv"):
        if should_skip_file(file_path):
            print(f"Skipping {file_path.name}...")
            continue

        print(f"Processing {file_path.name}...")

        try:
            with open(file_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)

                if not reader.fieldnames or "alignment_score" not in reader.fieldnames:
                    print("  -> Skipping (no 'alignment_score' column)")
                    continue

                total_score = 0.0
                total_rows = 0
                valid_score_rows = 0

                for row in reader:
                    total_rows += 1
                    val = (row.get("alignment_score") or "").strip()

                    try:
                        total_score += float(val)
                        valid_score_rows += 1
                    except ValueError:
                        # Ignore non-numeric values
                        pass

                results.append(
                    {
                        "filename": file_path.name,
                        "total_alignment_score": total_score,
                        "valid_score_rows": valid_score_rows,
                        "total_rows": total_rows,
                    }
                )

        except Exception as e:
            print(f"  -> Error: {e}")

    if not results:
        print("No valid CSV files found containing 'alignment_score'.")
        return

    output_path = base_dir / "alignment_scores_totals.csv"
    fieldnames = [
        "filename",
        "total_alignment_score",
        "valid_score_rows",
        "total_rows",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nTotals successfully saved to: {output_path}")


if __name__ == "__main__":
    sum_scores()