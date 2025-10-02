#!/usr/bin/env python3
"""
Summarize LLM vs NIST agreement and compute:
- Cohen's kappa (unweighted) AFTER binarizing labels with threshold 1:
  <=1 -> 0, >1 -> 1
- Cohen's kappa (unweighted) on a binarized view (same as above)
- Krippendorff's alpha (ordinal only) on original graded labels

Inputs: compare CSVs in outputs/trec_dl_llm_label/processed/
Accepted headers: (docid, nist_rel, llm_rel) or (docid, rel_reference, rel_model)
"""

from pathlib import Path
import csv
import re
import numpy as np
from sklearn.metrics import cohen_kappa_score        # pip install scikit-learn
import krippendorff as kd                            # pip install krippendorff

BASE_DIR = Path("outputs/trec_dl_llm_label")
IN_DIR   = BASE_DIR / "processed/utility/20250917_211535"
OUT_CSV  = IN_DIR / "doc_rel_summary_2.csv"
MODEL_RE = re.compile(r"doc_rel_compare_(.+)\.csv$", re.IGNORECASE)

def as_int(s):
    try:
        return int(str(s).strip())
    except Exception:
        return None

def extract_model_name(path: Path) -> str:
    m = MODEL_RE.match(path.name)
    return m.group(1) if m else path.stem

def _binarize_threshold(vals):
    """
    Map ordinal relevance to binary with threshold at 1:
      <=1 -> 0,  >1 -> 1
    """
    out = []
    for v in vals:
        if v is None:
            out.append(None)
        else:
            out.append(0 if v <= 1 else 1)
    return out

def summarize_file(path: Path):
    """Return (model, eq, lt, gt, n, k_bin, a_ord, nist_vals, llm_vals)."""
    model = extract_model_name(path)
    eq = lt = gt = n = 0
    nist_vals, llm_vals = [], []

    with path.open("r", encoding="utf-8", newline="") as fin:
        rdr = csv.DictReader(fin)
        if not rdr.fieldnames:
            return (model, 0,0,0,0, float("nan"), float("nan"), [], [])

        f = {c.lower(): c for c in rdr.fieldnames}
        nkey = f.get("nist_rel") or f.get("rel_reference")
        lkey = f.get("llm_rel")  or f.get("rel_model")
        if not nkey or not lkey:
            return (model, 0,0,0,0, float("nan"), float("nan"), [], [])

        for row in rdr:
            nist = as_int(row.get(nkey, ""))
            llm  = as_int(row.get(lkey,  ""))
            if nist is None or llm is None:
                continue
            nist_vals.append(nist)
            llm_vals.append(llm)
            n += 1
            if llm == nist: eq += 1
            elif llm < nist: lt += 1
            else: gt += 1

    if n == 0:
        return (model, 0,0,0,0, float("nan"), float("nan"), [], [])

    # --- Cohen's kappa on BINARIZED labels (<=1 -> 0, >1 -> 1)
    nist_b = _binarize_threshold(nist_vals)
    llm_b  = _binarize_threshold(llm_vals)

    # cohen_kappa_score needs concrete labels; we've already filtered None above.
    k_bin = cohen_kappa_score(nist_b, llm_b)

    # Krippendorff's alpha (ordinal) on original graded labels
    a_ord = kd.alpha(reliability_data=np.array([nist_vals, llm_vals]),
                     level_of_measurement="ordinal")

    return (model, eq, lt, gt, n, k_bin, a_ord, nist_vals, llm_vals)

def main():
    files = sorted(IN_DIR.glob("doc_rel_compare_*.csv")) or sorted(IN_DIR.glob("*.csv"))
    if not files:
        print(f"No CSVs found in {IN_DIR}")
        return

    rows = []
    ge = gl = gm = gt = 0
    all_nist, all_llm = [], []

    for f in files:
        model, eq, lt, gt1, n, kbin, aord, nv, lv = summarize_file(f)
        if n == 0:
            continue
        rows.append([model, eq, lt, gt1, n, kbin, aord])
        ge += eq; gl += lt; gm += gt1; gt += n
        all_nist.extend(nv); all_llm.extend(lv)

    if not rows:
        print("No usable rows found across input files.")
        return

    # Overall metrics on pooled items (binary Cohen's kappa, ordinal alpha on graded)
    all_nist_b = _binarize_threshold(all_nist)
    all_llm_b  = _binarize_threshold(all_llm)
    overall_kbin = cohen_kappa_score(all_nist_b, all_llm_b)
    overall_aord = kd.alpha(reliability_data=np.array([all_nist, all_llm]),
                            level_of_measurement="ordinal")

    # Write CSV
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout)
        w.writerow([
            "model",
            "equal_count","less_relevant_count","more_relevant_count","total_rows",
            "kappa_binary_threshold_1",      # <=1 -> 0, >1 -> 1
            "alpha_ordinal"
        ])
        w.writerows(rows)
        w.writerow([
            "__OVERALL__", ge, gl, gm, gt,
            overall_kbin,
            overall_aord
        ])

    # Console
    print(f"Wrote summary -> {OUT_CSV}")
    for r in rows:
        model, eq, lt, gt1, n, kbin, aord = r
        print(f"{model:40s} | = {eq:4d}  < {lt:4d}  > {gt1:4d}  (n={n:4d})  "
              f"k_bin={kbin:.3f}  α_ord={aord:.3f}")
    print(f"{'OVERALL':40s} | = {ge:4d}  < {gl:4d}  > {gm:4d}  (n={gt:4d})  "
          f"k_bin={overall_kbin:.3f}  α_ord={overall_aord:.3f}")

if __name__ == "__main__":
    main()
