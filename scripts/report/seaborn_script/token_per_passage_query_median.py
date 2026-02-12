#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Path setup (so helpers/ resolves)
# =========================
THIS_FILE = Path(__file__).resolve()
SEABORN_SCRIPT_DIR = THIS_FILE.parents[1]  # .../scripts/report/seaborn_script
if str(SEABORN_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SEABORN_SCRIPT_DIR))

from helpers.lang_profiles import get_langs  # type: ignore
from helpers.settings import apply_paper_fmt  # type: ignore

# =========================
# CONFIG
# =========================
PROJECT_ROOT = THIS_FILE.parents[3]

YEAR = "2021"
PROFILE = "lang"  # e.g. "lang", "word", "crit", "first", "last", ...

INPUT_ROOT = PROJECT_ROOT / "retrieved" / f"trec_dl_{YEAR}"

PASSAGE_COL = "passage_injected"
QUERY_COL_CANDIDATES = ["query_{lang}", "query", "query_eng"]

OUT_DIR = PROJECT_ROOT / "outputs" / "passage_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / f"median_passage_tokens_query_share_{PROFILE}_{YEAR}.csv"
OUT_PNG = OUT_DIR / f"median_passage_tokens_query_share_{PROFILE}_{YEAR}.png"
OUT_PDF = OUT_DIR / f"median_passage_tokens_query_share_{PROFILE}_{YEAR}.pdf"

# =========================
# TOKENIZER (multilingual-ish, no external deps)
# =========================
_CJK_RE = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u30FF\uAC00-\uD7AF]")
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

def tokenize(text: str) -> list[str]:
    if not text:
        return []
    text = _CJK_RE.sub(lambda m: f" {m.group(0)} ", text)
    tokens: list[str] = []
    i = 0
    while i < len(text):
        m = _WORD_RE.match(text, i)
        if m:
            tokens.append(m.group(0))
            i = m.end()
            continue
        ch = text[i]
        if not ch.isspace():
            tokens.append(ch)
        i += 1
    return tokens

# =========================
# IO helpers
# =========================
def iter_files(root: Path) -> Iterable[Path]:
    exts = {".csv", ".tsv"}  # keep it simple; your retrieved files are typically csv/tsv
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def resolve_query_col(df: pd.DataFrame, lang_variant: str) -> str | None:
    base = lang_variant.split("_", 1)[0].lower()
    candidates = [c.format(lang=base) for c in QUERY_COL_CANDIDATES]
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def median_tokens_query_share_for_folder(folder: Path, lang_variant: str) -> dict:
    """
    Computes medians over rows for:
      - passage tokens (tokenize(passage_injected))
      - query tokens (tokenize(query_col))
      - non-query tokens = max(passage - query, 0)
      - query fraction = query / passage  (row-wise), then median
    """
    passage_lens: list[int] = []
    query_lens: list[int] = []
    nonquery_lens: list[int] = []
    frac_vals: list[float] = []

    missing_query_col_files = 0
    missing_passage_col_files = 0

    for path in iter_files(folder):
        try:
            if path.suffix.lower() == ".tsv":
                df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_MINIMAL, low_memory=False)
            else:
                df = pd.read_csv(path, low_memory=False)

            cols_lower = {c.lower(): c for c in df.columns}
            if PASSAGE_COL.lower() not in cols_lower:
                missing_passage_col_files += 1
                continue
            pcol = cols_lower[PASSAGE_COL.lower()]

            qcol = resolve_query_col(df, lang_variant)
            if qcol is None:
                missing_query_col_files += 1
                continue

            sub = df[[pcol, qcol]].dropna()

            for _, row in sub.iterrows():
                p = str(row[pcol])
                q = str(row[qcol])

                pt = len(tokenize(p))
                qt = len(tokenize(q))
                nt = max(pt - qt, 0)

                passage_lens.append(pt)
                query_lens.append(qt)
                nonquery_lens.append(nt)
                frac_vals.append((qt / pt) if pt > 0 else 0.0)

        except Exception as e:
            print(f"[WARN] failed reading {path}: {e}")
            continue

    n = len(passage_lens)
    if n == 0:
        return {
            "lang": lang_variant,
            "n": 0,
            "med_passage_tokens": 0.0,
            "med_query_tokens": 0.0,
            "med_nonquery_tokens": 0.0,
            "med_query_frac": 0.0,
            "missing_passage_col_files": missing_passage_col_files,
            "missing_query_col_files": missing_query_col_files,
        }

    # Use pandas for medians (robust + concise)
    med_passage = float(pd.Series(passage_lens).median())
    med_query = float(pd.Series(query_lens).median())
    med_nonquery = float(pd.Series(nonquery_lens).median())
    med_frac = float(pd.Series(frac_vals).median())

    return {
        "lang": lang_variant,
        "n": n,
        "med_passage_tokens": med_passage,
        "med_query_tokens": med_query,
        "med_nonquery_tokens": med_nonquery,
        "med_query_frac": med_frac,
        "missing_passage_col_files": missing_passage_col_files,
        "missing_query_col_files": missing_query_col_files,
    }

# =========================
# Plotting (stacked bar)
# =========================
def plot_stacked(df: pd.DataFrame) -> None:
    apply_paper_fmt()

    df = df[df["n"] > 0].copy()
    df["lang"] = pd.Categorical(df["lang"], categories=df["lang"].tolist(), ordered=True)

    w = max(3.0, 0.35 * len(df))
    plt.figure(figsize=(w, 2.8))

    x = list(range(len(df)))
    base_vals = df["med_nonquery_tokens"].to_numpy()
    inj_vals = df["med_query_tokens"].to_numpy()

    plt.bar(x, base_vals, label="Passage (excluding injected query)")
    plt.bar(x, inj_vals, bottom=base_vals, label="Injected query")

    plt.xticks(x, df["lang"].tolist(), rotation=45, ha="right")
    plt.ylabel("Median tokens per passage")
    plt.title(f"TREC-DL {YEAR} • Profile: {PROFILE} • Median query share in passage")

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.savefig(OUT_PDF)
    plt.close()

    print(f"Wrote: {OUT_PNG}")
    print(f"Wrote: {OUT_PDF}")

# =========================
# MAIN
# =========================
def main() -> None:
    langs = get_langs(PROFILE)
    print(f"PROFILE={PROFILE} -> {len(langs)} language folders")

    rows: list[dict] = []

    for lang_variant in langs:
        folder = INPUT_ROOT / lang_variant
        if not folder.exists():
            print(f"[WARN] missing folder: {folder}")
            rows.append({
                "lang": lang_variant,
                "n": 0,
                "med_passage_tokens": 0.0,
                "med_query_tokens": 0.0,
                "med_nonquery_tokens": 0.0,
                "med_query_frac": 0.0,
                "missing_passage_col_files": 0,
                "missing_query_col_files": 0,
            })
            continue

        stats = median_tokens_query_share_for_folder(folder, lang_variant)
        rows.append(stats)

        print(
            f"{lang_variant:>15}  n={stats['n']:<8}  "
            f"med_pass={stats['med_passage_tokens']:.3f}  "
            f"med_query={stats['med_query_tokens']:.3f}  "
            f"share={stats['med_query_frac']*100:.1f}%"
        )

        if stats["missing_query_col_files"] > 0:
            print(f"  [note] missing query col in {stats['missing_query_col_files']} file(s) under {lang_variant}")

    out = pd.DataFrame(rows)

    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote: {OUT_CSV}")

    plot_stacked(out)

if __name__ == "__main__":
    main()
