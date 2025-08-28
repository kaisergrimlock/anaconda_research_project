import pandas as pd
from pathlib import Path
from collections import defaultdict

# --- Config ---
ROOT = Path("outputs/llm_label")
PATTERN = "llm_labels_q524332_*.tsv"

# --- Helpers ---
def is_original(p: Path) -> bool:
    name = p.name.lower()
    return not any(tag in name for tag in ["viet", "eng_2", "eng", "thai", "translated"])

def variant_of(p: Path) -> str:
    name = p.name.lower()
    if "eng_2" in name:     return "eng_2"
    if "eng" in name:       return "eng"
    if "viet" in name:      return "viet"
    if "thai" in name:      return "thai"
    if "translated" in name:return "translated"
    return "original"

def base_key(p: Path) -> str:
    parts = p.stem.split("_")
    return "_".join(parts[:4])

def tau_ap_from_scores(docids, s_ref, s_cmp):
    order_ref = sorted(docids, key=lambda d: (-s_ref[d], d))
    order_cmp = sorted(docids, key=lambda d: (-s_cmp[d], d))
    pos_cmp = {d: i for i, d in enumerate(order_cmp)}
    n = len(order_ref)
    if n <= 1:
        return 1.0
    S = 0.0
    for i in range(1, n):
        di = order_ref[i]
        pi = pos_cmp[di]
        c = 0
        for j in range(i):
            dj = order_ref[j]
            if pos_cmp[dj] > pi:
                c += 1
        S += c / i
    return 1.0 - (2.0 / (n - 1)) * S

def read_labels(path: Path):
    """Read a TSV and return DataFrame with docid (str) and relevance (float), NaNs coerced."""
    df = pd.read_csv(path, sep="\t", comment="#", usecols=["docid", "relevance"])
    df["docid"] = df["docid"].astype(str)
    df["relevance"] = pd.to_numeric(df["relevance"], errors="coerce")
    return df

# --- Load files and group ---
files = sorted(ROOT.glob(PATTERN))
groups = defaultdict(list)
for f in files:
    groups[base_key(f)].append(f)

summary_rows = []

for base, fs in groups.items():
    originals = [f for f in fs if is_original(f)]
    if not originals:
        print(f"[SKIP] No original found for group: {base}")
        continue

    original_file = originals[0]
    print(f"\n=== Comparing (AP-τ) for base model: {base} ===")

    df_orig = read_labels(original_file).dropna(subset=["relevance"])
    s_ref_all = dict(zip(df_orig["docid"], df_orig["relevance"]))

    for f in fs:
        if f == original_file:
            continue

        df_other = read_labels(f).dropna(subset=["relevance"])
        s_other_all = dict(zip(df_other["docid"], df_other["relevance"]))

        # Overlap
        overlap = sorted(set(s_ref_all.keys()) & set(s_other_all.keys()))
        if len(overlap) < 2:
            print(f"[SKIP] Not enough overlapping docids (n={len(overlap)}) for {f.name}.")
            continue

        s_ref = {d: float(s_ref_all[d]) for d in overlap}
        s_cmp = {d: float(s_other_all[d]) for d in overlap}

        tau_ap = tau_ap_from_scores(overlap, s_ref, s_cmp)
        vlabel = variant_of(f)
        print(f"[{vlabel.upper():8}] AP-τ vs original: {tau_ap:.3f}  (n={len(overlap)})")

        summary_rows.append({
            "base": base,
            "variant_file": f.name,
            "variant": vlabel,
            "original_file": original_file.name,
            "n_overlap": len(overlap),
            "AP_tau": round(tau_ap, 3)
        })

# --- Write consolidated CSV ---
if summary_rows:
    out_csv = ROOT / "comparison_ap_tau_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"\nSaved AP-τ summary: {out_csv}")
else:
    print("\nNo comparisons produced a result.")
