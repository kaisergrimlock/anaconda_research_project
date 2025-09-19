#!/usr/bin/env python3
"""
Summarize LLM vs NIST agreement and compute:
- Cohen's kappa (unweighted, linear, quadratic) on graded labels (0–3)
- Cohen's kappa (unweighted) on a binarized view (0 vs ≥1)
- Krippendorff's alpha (ordinal only)

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
IN_DIR   = BASE_DIR / "processed/few_shot_2/20250918_205610"
OUT_CSV  = IN_DIR / "doc_rel_summary.csv"
MODEL_RE = re.compile(r"doc_rel_compare_(.+)\.csv$", re.IGNORECASE)

def as_int(s):
    try:
        return int(str(s).strip())
    except Exception:
        return None

def extract_model_name(path: Path) -> str:
    m = MODEL_RE.match(path.name)
    return m.group(1) if m else path.stem

def _binarize(vals):
    """0 -> 0; {1,2,3,...} -> 1"""
    return [0 if (v is None or v <= 0) else 1 for v in vals]

def summarize_file(path: Path):
    """Return (model, eq, lt, gt, n, k, k_lin, k_quad, k_bin, a_ord, nist_vals, llm_vals)."""
    model = extract_model_name(path)
    eq = lt = gt = n = 0
    nist_vals, llm_vals = [], []

    with path.open("r", encoding="utf-8", newline="") as fin:
        rdr = csv.DictReader(fin)
        if not rdr.fieldnames:
            return (model, 0,0,0,0, *(float("nan"),)*4, float("nan"), [], [])

        f = {c.lower(): c for c in rdr.fieldnames}
        nkey = f.get("nist_rel") or f.get("rel_reference")
        lkey = f.get("llm_rel")  or f.get("rel_model")
        if not nkey or not lkey:
            return (model, 0,0,0,0, *(float("nan"),)*4, float("nan"), [], [])

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
        return (model, 0,0,0,0, *(float("nan"),)*4, float("nan"), [], [])

    # Cohen's kappa on graded and binarized labels
    k      = cohen_kappa_score(nist_vals, llm_vals)
    k_lin  = cohen_kappa_score(nist_vals, llm_vals, weights="linear")
    k_quad = cohen_kappa_score(nist_vals, llm_vals, weights="quadratic")
    k_bin  = cohen_kappa_score(_binarize(nist_vals), _binarize(llm_vals))

    # Krippendorff's alpha (ordinal)
    a_ord = kd.alpha(reliability_data=np.array([nist_vals, llm_vals]),
                     level_of_measurement="ordinal")

    return (model, eq, lt, gt, n, k, k_lin, k_quad, k_bin, a_ord, nist_vals, llm_vals)

def main():
    files = sorted(IN_DIR.glob("doc_rel_compare_*.csv")) or sorted(IN_DIR.glob("*.csv"))
    if not files:
        print(f"No CSVs found in {IN_DIR}")
        return

    rows = []
    ge = gl = gm = gt = 0
    all_nist, all_llm = [], []

    for f in files:
        model, eq, lt, gt1, n, k, kl, kq, kbin, aord, nv, lv = summarize_file(f)
        if n == 0:
            continue
        rows.append([model, eq, lt, gt1, n, k, kl, kq, kbin, aord])
        ge += eq; gl += lt; gm += gt1; gt += n
        all_nist.extend(nv); all_llm.extend(lv)

    if not rows:
        print("No usable rows found across input files.")
        return

    # Overall metrics on pooled items
    overall_k    = cohen_kappa_score(all_nist, all_llm)
    overall_kl   = cohen_kappa_score(all_nist, all_llm, weights="linear")
    overall_kq   = cohen_kappa_score(all_nist, all_llm, weights="quadratic")
    overall_kbin = cohen_kappa_score(_binarize(all_nist), _binarize(all_llm))
    overall_aord = kd.alpha(reliability_data=np.array([all_nist, all_llm]),
                            level_of_measurement="ordinal")

    # Write CSV (no alpha_nominal)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout)
        w.writerow([
            "model",
            "equal_count","less_relevant_count","more_relevant_count","total_rows",
            "kappa","kappa_linear","kappa_quadratic","kappa_binarized",
            "alpha_ordinal"
        ])
        w.writerows(rows)
        w.writerow([
            "__OVERALL__", ge, gl, gm, gt,
            overall_k, overall_kl, overall_kq, overall_kbin,
            overall_aord
        ])

    # Console
    print(f"Wrote summary -> {OUT_CSV}")
    for r in rows:
        model, eq, lt, gt1, n, k, kl, kq, kbin, aord = r
        print(f"{model:40s} | = {eq:4d}  < {lt:4d}  > {gt1:4d}  (n={n:4d})  "
              f"k={k:.3f}  lin={kl:.3f}  quad={kq:.3f}  bin={kbin:.3f}  "
              f"α_ord={aord:.3f}")
    print(f"{'OVERALL':40s} | = {ge:4d}  < {gl:4d}  > {gm:4d}  (n={gt:4d})  "
          f"k={overall_k:.3f}  lin={overall_kl:.3f}  quad={overall_kq:.3f}  bin={overall_kbin:.3f}  "
          f"α_ord={overall_aord:.3f}")

if __name__ == "__main__":
    main()
