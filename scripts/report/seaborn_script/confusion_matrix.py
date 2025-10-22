#!/usr/bin/env python3
import re
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Config ----------
TREC_DL_YEAR = "2022"
MODEL = "gpt-oss-20b"
NIST_DIR = Path(f"retrieved/trec_dl_{TREC_DL_YEAR}") / "judged"
LLM_FILE   = Path("outputs/llm_label/" + MODEL + "/" + MODEL + "_trec_dl_" + TREC_DL_YEAR + "_raw_with_ids.csv")
OUT_DIR    = Path("outputs/baseline/" + TREC_DL_YEAR)
OUT_COUNTS = OUT_DIR / "confusion_matrix_llm_vs_nist.csv"
OUT_PCT    = OUT_DIR / "confusion_matrix_llm_vs_nist_pct.csv"
OUT_PNG    = OUT_DIR / "confusion_matrix_llm_vs_nist.png"

KEY_CHOICES = [("query", "docid"), ("topic", "docid")]
LABEL_COL_CHOICES = ["relevance", "label"]
LABELS = [0, 1, 2, 3]

# If True, rows where we cannot parse an LLM label become 0.
# If False (recommended for sanity), we DROP those rows from the matrix.
MAP_INVALID_TO_ZERO = False
# ---------------------------

def read_csv_smart(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, engine="python")  # tolerant + handles utf-8-sig

def pick_keys(df: pd.DataFrame):
    cols = {c.strip(): c for c in df.columns}
    for a, b in KEY_CHOICES:
        if a in cols and b in cols:
            return cols[a], cols[b]
    raise KeyError(f"No (query,docid) or (topic,docid) in {list(df.columns)}")

def pick_label_col(df: pd.DataFrame):
    cols = {c.strip(): c for c in df.columns}
    for c in LABEL_COL_CHOICES:
        if c in cols:
            return cols[c]
    raise KeyError(f"Neither 'relevance' nor 'label' in {list(df.columns)}")

_digit_0_3 = re.compile(r"\b([0-3])\b")  # safe, won’t match 10/30/etc.

def parse_label(value):
    """Try hard to extract a label in {0,1,2,3} from messy strings."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    # exact?
    if s in {"0", "1", "2", "3"}:
        return int(s)
    # common patterns: 'O=2', 'label_3', '2 (mostly)', 'L2', etc.
    m = _digit_0_3.search(s)
    if m:
        return int(m.group(1))
    return None  # unparseable

# 1) Load NIST (all CSVs)
nist_files = sorted(NIST_DIR.rglob("*.csv"))
if not nist_files:
    raise FileNotFoundError(f"No CSV files under {NIST_DIR}")

nist_parts = []
for fp in nist_files:
    df = read_csv_smart(fp)
    k1, k2 = pick_keys(df)
    lcol   = pick_label_col(df)
    part = df[[k1, k2, lcol]].rename(columns={k1: "query", k2: "docid", lcol: "NIST"})
    part["query"] = part["query"].astype(str).str.strip()
    part["docid"] = part["docid"].astype(str).str.strip()
    part["NIST"]  = part["NIST"].apply(parse_label).fillna(0).astype(int)  # NIST should be clean, fallback 0
    nist_parts.append(part)

nist = pd.concat(nist_parts, ignore_index=True).drop_duplicates(subset=["query", "docid"], keep="first")

# 2) Load LLM
llm = read_csv_smart(LLM_FILE)
k1, k2 = pick_keys(llm)
lcol   = pick_label_col(llm)
llm = llm[[k1, k2, lcol]].rename(columns={k1: "query", k2: "docid", lcol: "LLM_raw"})
llm["query"] = llm["query"].astype(str).str.strip()
llm["docid"] = llm["docid"].astype(str).str.strip()
llm["LLM_parsed"] = llm["LLM_raw"].apply(parse_label)

# parsing summary
total_llm = len(llm)
parsed_ok = llm["LLM_parsed"].notna().sum()
parsed_bad = total_llm - parsed_ok
print(f"LLM rows: {total_llm} | parsed: {parsed_ok} | unparseable: {parsed_bad}")

if MAP_INVALID_TO_ZERO:
    llm["LLM"] = llm["LLM_parsed"].fillna(0).astype(int)
else:
    llm = llm[llm["LLM_parsed"].notna()].copy()
    llm["LLM"] = llm["LLM_parsed"].astype(int)

# 3) Join & sanity info
paired = nist.merge(llm[["query","docid","LLM"]], on=["query","docid"], how="inner")
print(f"Pairs after join: {len(paired)}")

# 4) Confusion matrix (counts)
cm = pd.crosstab(
    index=pd.Categorical(paired["NIST"], categories=LABELS, ordered=True),
    columns=pd.Categorical(paired["LLM"],  categories=LABELS, ordered=True),
    dropna=False
)
cm.index.name = "NIST"; cm.columns.name = "LLM"

# 5) Row-normalized (%)
cm_pct = cm.div(cm.sum(axis=1).replace(0, 1), axis=0) * 100.0

# 6) Save + plot
OUT_DIR.mkdir(parents=True, exist_ok=True)
cm.to_csv(OUT_COUNTS)
cm_pct.round(2).to_csv(OUT_PCT)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cbar=True)
plt.title("Confusion Matrix: NIST vs LLM (counts)")
plt.ylabel("NIST label"); plt.xlabel("LLM label")
plt.tight_layout(); plt.savefig(OUT_PNG, dpi=200); plt.show()

print(f"Wrote counts to: {OUT_COUNTS}")
print(f"Wrote row-% to:  {OUT_PCT}")
print(f"Saved heatmap to: {OUT_PNG}")
