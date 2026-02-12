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
QID_COL = "qid"

QUERY_COL_CANDIDATES = ["query_{lang}", "query", "query_eng"]

OUT_DIR = PROJECT_ROOT / "outputs" / "passage_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / f"avg_passage_tokens_query_share_{PROFILE}_{YEAR}.csv"
OUT_PNG = OUT_DIR / f"avg_passage_tokens_query_share_{PROFILE}_{YEAR}.png"
OUT_PDF = OUT_DIR / f"avg_passage_tokens_query_share_{PROFILE}_{YEAR}.pdf"

# Skip any file whose name contains this marker (your "part0" shards)
SKIP_NAME_SUBSTRINGS = ["part0"]

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
def should_skip_file(p: Path) -> bool:
    name = p.name.lower()
    return any(s in name for s in SKIP_NAME_SUBSTRINGS)

def iter_files(root: Path) -> Iterable[Path]:
    exts = {".csv", ".tsv"}  # your retrieved files are typically csv/tsv
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if should_skip_file(p):
            continue
        yield p

def resolve_query_col(df: pd.DataFrame, lang_variant: str) -> str | None:
    base = lang_variant.split("_", 1)[0].lower()
    candidates = [c.format(lang=base) for c in QUERY_COL_CANDIDATES]
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def avg_tokens_query_share_for_folder(folder: Path, lang_variant: str) -> dict:
    total_passage = 0
    total_query = 0
    n = 0
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
                total_passage += pt
                total_query += qt
                n += 1

        except Exception as e:
            print(f"[WARN] failed reading {path}: {e}")
            continue

    if n == 0:
        return {
            "lang": lang_variant,
            "n": 0,
            "avg_passage_tokens": 0.0,
            "avg_query_tokens": 0.0,
            "avg_nonquery_tokens": 0.0,
            "avg_query_frac": 0.0,
            "missing_passage_col_files": missing_passage_col_files,
            "missing_query_col_files": missing_query_col_files,
        }

    avg_passage = total_passage / n
    avg_query = total_query / n
    avg_nonquery = max(avg_passage - avg_query, 0.0)
    avg_frac = (avg_query / avg_passage) if avg_passage > 0 else 0.0

    return {
        "lang": lang_variant,
        "n": n,
        "avg_passage_tokens": avg_passage,
        "avg_query_tokens": avg_query,
        "avg_nonquery_tokens": avg_nonquery,
        "avg_query_frac": avg_frac,
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
    base_vals = df["avg_nonquery_tokens"].to_numpy()
    inj_vals = df["avg_query_tokens"].to_numpy()

    plt.bar(x, base_vals, label="Passage (excluding injected query)")
    plt.bar(x, inj_vals, bottom=base_vals, label="Injected query")

    plt.xticks(x, df["lang"].tolist(), rotation=45, ha="right")
    plt.ylabel("Avg tokens per passage")
    plt.title(f"TREC-DL {YEAR} • Profile: {PROFILE} • Query share in passage")

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
    print(f"Skipping files whose name contains: {SKIP_NAME_SUBSTRINGS}")

    rows: list[dict] = []

    for lang_variant in langs:
        folder = INPUT_ROOT / lang_variant
        if not folder.exists():
            print(f"[WARN] missing folder: {folder}")
            rows.append({
                "lang": lang_variant,
                "n": 0,
                "avg_passage_tokens": 0.0,
                "avg_query_tokens": 0.0,
                "avg_nonquery_tokens": 0.0,
                "avg_query_frac": 0.0,
                "missing_passage_col_files": 0,
                "missing_query_col_files": 0,
            })
            continue

        stats = avg_tokens_query_share_for_folder(folder, lang_variant)
        rows.append(stats)

        print(
            f"{lang_variant:>15}  n={stats['n']:<8}  "
            f"avg_pass={stats['avg_passage_tokens']:.3f}  "
            f"avg_query={stats['avg_query_tokens']:.3f}  "
            f"share={stats['avg_query_frac']*100:.1f}%"
        )

        if stats["missing_query_col_files"] > 0:
            print(f"  [note] missing query col in {stats['missing_query_col_files']} file(s) under {lang_variant}")

    out = pd.DataFrame(rows)

    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote: {OUT_CSV}")

    plot_stacked(out)

if __name__ == "__main__":
    main()
