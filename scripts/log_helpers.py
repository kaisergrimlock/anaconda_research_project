# scripts/log_helpers.py
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


# -----------------------
# timestamps
# -----------------------
def timestamp_id() -> str:
    """Short run id, good for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def timestamp_iso() -> str:
    """ISO timestamp for CSV logs."""
    return datetime.now().isoformat(timespec="seconds")


# -----------------------
# model pricing / cost
# -----------------------
def load_model_prices(csv_path: Path) -> Dict[str, Tuple[float, float]]:
    """
    CSV schema: llm,input,output
    prices are per 1K tokens.
    """
    prices: Dict[str, Tuple[float, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = (row["llm"] or "").strip().strip('"').strip("'")
            pin = float((row["input"] or "0").strip())
            pout = float((row["output"] or "0").strip())
            prices[name] = (pin, pout)
    return prices


def estimate_run_cost(model: str, tin: int, tout: int, csv_path: Path) -> float:
    """
    Cost in USD, using per-1K-token price.
    """
    prices = load_model_prices(csv_path)
    if model not in prices:
        raise KeyError(f"Model '{model}' not found in {csv_path}")
    pin, pout = prices[model]
    return (tin * pin + tout * pout) / 1000.0


# -----------------------
# CSV run log
# -----------------------
def append_token_row(tokens_csv: Path, row: dict) -> None:
    """
    Append a single run row to tokens_csv.
    Creates file + header if missing.
    """
    file_exists = tokens_csv.exists()
    with tokens_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "timestamp",
                "model",
                "num_examples",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "estimated_cost_usd",
                "labels_csv",
                "log_json",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# -----------------------
# JSON run index
# -----------------------
def write_run_log_index(entries: List[Dict[str, Any]], out_path: Path) -> None:
    """
    entries example: [{"part": "all_topics...part_001.csv", "log_json": "logs/...json"}, ...]
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
