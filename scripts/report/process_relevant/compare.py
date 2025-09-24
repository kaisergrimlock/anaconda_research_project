#!/usr/bin/env python3
import csv
from pathlib import Path
import re
from datetime import datetime
"""
Compare LLM-produced relevance labels against a nisterence set.

Reads a single nisterence CSV:
  outputs/trec_dl_llm_label/relevant/relevant_results_100.csv

and multiple model CSVs:
  outputs/trec_dl_llm_label/relevant/utility/llm_labels_*.csv

For each model file, it outputs:
  outputs/trec_dl_llm_label/processed/doc_rel_compare_<model>.csv
with columns: docid, nist_rel, llm_rel
"""

# ----------------------------
# Paths for your layout
# ----------------------------
BASE_DIR  = Path("outputs/trec_dl_llm_label")
nist_FILE  = BASE_DIR / "relevant" / "trecdl_passage_2019_combined.csv"   # <-- the nisterence
MODEL_DIR = BASE_DIR / "relevant" / "utility/20250920_210007"                    # <-- the LLM label files

OUT_DIR   = BASE_DIR / "processed" / "utility/" / datetime.now().strftime("%Y%m%d_%H%M%S")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Extract model name from filenames like:
# 20250917_011043_llm_labels_anthropic.claude-3-haiku-20240307-v1.csv
MODEL_RE = re.compile(r"llm_labels_(.+)\.csv$", re.IGNORECASE)

def extract_model_name(filename: str) -> str:
    m = MODEL_RE.search(filename)
    return m.group(1) if m else Path(filename).stem

def load_doc_rels(csv_path: Path) -> dict[str, str]:
    """Load {docid: relevance} from a single CSV (accepts 'docid' or 'pid', 'relevance' or 'rel')."""
    with csv_path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            return {}
        fields = {c.lower(): c for c in reader.fieldnames}
        doc_key = fields.get("docid") or fields.get("pid")
        rel_key = fields.get("relevance") or fields.get("rel")
        if not doc_key or not rel_key:
            return {}

        out = {}
        for row in reader:
            did = (row.get(doc_key) or "").strip()
            rel = (row.get(rel_key) or "").strip()
            if did:
                out[did] = rel
        return out

def load_models(folder: Path) -> dict[str, dict[str,str]]:
    """Load all llm_labels_*.csv under folder -> {model_name: {docid: rel}}."""
    models = {}
    for f in sorted(folder.glob("*llm_labels_*.csv")):
        models[extract_model_name(f.name)] = load_doc_rels(f)
    return models

def main():
    nist = load_doc_rels(nist_FILE)
    if not nist:
        print(f"No usable rows in nisterence file: {nist_FILE}")
        return

    models = load_models(MODEL_DIR)
    for model, rels in models.items():
        common = sorted(set(rels) & set(nist), key=lambda x: (len(x), x))
        if not common:
            print(f"No common docids for model: {model}")
            continue

        out_csv = OUT_DIR / f"doc_rel_compare_{model}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as fout:
            w = csv.writer(fout)
            w.writerow(["docid", "nist_rel", "llm_rel"])
            for did in common:
                w.writerow([did, nist[did], rels[did]])

        print(f"Wrote {len(common)} rows -> {out_csv}")

if __name__ == "__main__":
    main()
