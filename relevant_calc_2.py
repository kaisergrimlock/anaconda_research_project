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
    if "eng_2" in name: return "eng_2"
    if "eng" in name:   return "eng"
    if "viet" in name:  return "viet"
    return "other"

def base_key(p: Path) -> str:
    parts = p.stem.split("_")
    return "_".join(parts[:4])

def read_labels(path: Path):
    df = pd.read_csv(path, sep="\t", comment="#", usecols=["docid", "relevance"])
    df["docid"] = df["docid"].astype(str)
    df["relevance"] = pd.to_numeric(df["relevance"], errors="coerce")
    return df.dropna(subset=["relevance"])

# --- Load files and group ---
files = sorted(ROOT.glob(PATTERN))
groups = defaultdict(list)
for f in files:
    groups[base_key(f)].append(f)

# Collect summary rows
summary_rows = []

for base, fs in groups.items():
    originals = [f for f in fs if is_original(f)]
    if not originals:
        print(f"[SKIP] No original found for group: {base}")
        continue

    original_file = originals[0]
    print(f"\n=== Flip Analysis for base model: {base} ===")

    # Load original labels
    df_orig = read_labels(original_file)
    s_ref = dict(zip(df_orig["docid"], df_orig["relevance"].astype(int)))

    for f in fs:
        if f == original_file:
            continue

        # Load variant labels
        df_variant = read_labels(f)
        s_var = dict(zip(df_variant["docid"], df_variant["relevance"].astype(int)))

        # Find overlapping docids
        overlap = set(s_ref.keys()) & set(s_var.keys())
        if not overlap:
            print(f"[SKIP] No overlapping docids for {f.name}.")
            continue

        flips_1_to_0 = sum(1 for d in overlap if s_ref[d] == 1 and s_var[d] == 0)
        flips_0_to_1 = sum(1 for d in overlap if s_ref[d] == 0 and s_var[d] == 1)

        vlabel = variant_of(f)
        print(f"[{vlabel.upper():8}] 1→0 flips: {flips_1_to_0} | 0→1 flips: {flips_0_to_1} | Overlap: {len(overlap)}")

        summary_rows.append({
            "base": base,
            "variant_file": f.name,
            "variant": vlabel,
            "original_file": original_file.name,
            "n_overlap": len(overlap),
            "Flips_1_to_0": flips_1_to_0,
            "Flips_0_to_1": flips_0_to_1
        })

# --- Save summary ---
if summary_rows:
    out_csv = ROOT / "comparison_label_flips.csv"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"\nSaved label flip summary: {out_csv}")
else:
    print("\nNo label flips calculated.")
