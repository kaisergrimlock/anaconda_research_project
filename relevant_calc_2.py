import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path("outputs/llm_label")
files = sorted(ROOT.glob("llm_labels_q524332_*.tsv"))

# Helpers
def is_original(name: str) -> bool:
    return not any(x in name for x in ["viet", "eng", "thai", "translated"])

def is_eng(name: str) -> bool:
    return "eng" in name

def is_viet(name: str) -> bool:
    return "viet" in name

def base_key(p: Path) -> str:
    # Group up to the model identifier chunk, e.g.
    # llm_labels_q524332_anthropic.claude-3-haiku-20240307-v1_0_...
    parts = p.stem.split("_")
    return "_".join(parts[:4])

# Group files by base model
groups = defaultdict(list)
for f in files:
    groups[base_key(f)].append(f)

for base, group_files in groups.items():
    # Identify original / eng / viet within the group
    original = next((f for f in group_files if is_original(f.name)), None)
    eng     = next((f for f in group_files if is_eng(f.name)), None)
    viet    = next((f for f in group_files if is_viet(f.name)), None)

    if not (eng and viet):
        print(f"[SKIP] Need both eng & viet variants in group: {base}")
        continue

    # Load required files (original optional but useful to include)
    usecols = ["docid", "relevance"]
    df_eng  = pd.read_csv(eng,  sep="\t", comment="#", usecols=usecols).rename(columns={"relevance":"eng"})
    df_viet = pd.read_csv(viet, sep="\t", comment="#", usecols=usecols).rename(columns={"relevance":"viet"})

    df = pd.merge(df_eng, df_viet, on="docid", how="inner")

    if original:
        df_orig = pd.read_csv(original, sep="\t", comment="#", usecols=usecols).rename(columns={"relevance":"orig"})
        df = pd.merge(df, df_orig, on="docid", how="left")
    else:
        df["orig"] = pd.NA

    # Clean and cast
    df = df.dropna(subset=["eng", "viet"])  # keep only rows where both variants have labels
    if df.empty:
        print(f"[SKIP] No overlapping docids for eng vs viet in group: {base}")
        continue

    df["eng"]  = df["eng"].astype(int)
    df["viet"] = df["viet"].astype(int)
    # orig might be NaN; cast safely
    df["orig"] = pd.to_numeric(df["orig"], errors="coerce").astype("Int64")

    # For each document: who marked it relevant?
    # Categories:
    # - "ENG_ONLY": eng==1 and viet==0
    # - "VIET_ONLY": viet==1 and eng==0
    # - "BOTH": eng==1 and viet==1
    # - "NEITHER": eng==0 and viet==0
    def winner(row):
        if row.eng == 1 and row.viet == 0: return "ENG_ONLY"
        if row.eng == 0 and row.viet == 1: return "VIET_ONLY"
        if row.eng == 1 and row.viet == 1: return "BOTH"
        return "NEITHER"

    df["winner"] = df.apply(winner, axis=1)

    # Totals
    eng_only   = (df["winner"] == "ENG_ONLY").sum()
    viet_only  = (df["winner"] == "VIET_ONLY").sum()
    both       = (df["winner"] == "BOTH").sum()
    neither    = (df["winner"] == "NEITHER").sum()

    print(f"\n=== {base} ===")
    print(f"ENG_ONLY:  {eng_only}")
    print(f"VIET_ONLY: {viet_only}")
    print(f"BOTH:      {both}")
    print(f"NEITHER:   {neither}")

    # Also show total relevant counts per variant across all docs
    total_relevant_eng  = (df["eng"] == 1).sum()
    total_relevant_viet = (df["viet"] == 1).sum()
    print(f"Total relevant (ENG):  {total_relevant_eng}")
    print(f"Total relevant (VIET): {total_relevant_viet}")

    # Optional: compare against original if you want to see which variant agrees with orig==1 more often
    if df["orig"].notna().any():
        eng_agrees_with_orig1  = ((df["orig"] == 1) & (df["eng"] == 1)).sum()
        viet_agrees_with_orig1 = ((df["orig"] == 1) & (df["viet"] == 1)).sum()
        print(f"Agree with ORIG==1 (ENG):  {eng_agrees_with_orig1}")
        print(f"Agree with ORIG==1 (VIET): {viet_agrees_with_orig1}")

    # Write per-doc CSV so you can inspect exactly which docs each variant marked relevant
    out_csv = ROOT / f"{base}_eng_vs_viet_per_doc.csv"
    df[["docid", "orig", "eng", "viet", "winner"]].to_csv(out_csv, index=False)
    print(f"Saved per-doc analysis: {out_csv}")
