#!/usr/bin/env python3
import csv
from pathlib import Path
import re
from collections import defaultdict

# ----------------------------
# Config
# ----------------------------
BASE_DIR = Path("outputs/trec_dl_llm_label")
IRR_DIR  = BASE_DIR / "irrelevant"
TRANS_DIR = BASE_DIR / "translated/zh"
OUT_DIR   = BASE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Regex to extract model name from filename
MODEL_RE = re.compile(r"llm_labels_([^_]+)")

def extract_model_name(filename: str) -> str:
    """Extract a simplified model name from filename."""
    match = MODEL_RE.search(filename)
    if match:
        return match.group(1)
    return "unknown"

def load_doc_rels_by_model(folder: Path) -> dict:
    """
    Load relevance labels grouped by model.
    Returns: {model_name: {docid: relevance}}
    """
    model_map = defaultdict(dict)
    for f in sorted(folder.glob("*.csv")):
        model = extract_model_name(f.name)
        with f.open("r", encoding="utf-8", newline="") as fin:
            reader = csv.DictReader(fin)
            if not reader.fieldnames:
                continue
            # find docid and relevance keys
            field_lc = {c.lower(): c for c in reader.fieldnames}
            doc_key = field_lc.get("docid")
            rel_key = field_lc.get("relevance") or field_lc.get("rel")
            if not doc_key or not rel_key:
                continue
            for row in reader:
                docid = row.get(doc_key, "").strip()
                rel   = row.get(rel_key, "").strip()
                if docid:
                    model_map[model][docid] = rel
    return model_map

def main():
    irr_map  = load_doc_rels_by_model(IRR_DIR)
    trans_map = load_doc_rels_by_model(TRANS_DIR)

    for model in sorted(set(irr_map.keys()) | set(trans_map.keys())):
        irr_rels = irr_map.get(model, {})
        trans_rels = trans_map.get(model, {})

        # find common docids
        common_ids = sorted(set(irr_rels.keys()) & set(trans_rels.keys()), key=lambda x: (len(x), x))

        if not common_ids:
            print(f"No common docids found for model: {model}")
            continue

        out_csv = OUT_DIR / f"doc_rel_compare_{model}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(["docid", "rel", "rel_translated"])
            for docid in common_ids:
                writer.writerow([docid, irr_rels[docid], trans_rels[docid]])

        print(f"Wrote {len(common_ids)} rows to {out_csv}")

if __name__ == "__main__":
    main()
