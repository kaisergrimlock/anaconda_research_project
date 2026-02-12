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

OUT_DIR = PROJECT_ROOT / "outputs" / "passage_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / f"avg_passage_tokens_by_profile_{PROFILE}_{YEAR}.csv"
OUT_PNG = OUT_DIR / f"avg_passage_tokens_by_profile_{PROFILE}_{YEAR}.png"
OUT_PDF = OUT_DIR / f"avg_passage_tokens_by_profile_{PROFILE}_{YEAR}.pdf"

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
    exts = {".csv", ".tsv", ".jsonl", ".json"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def passages_from_df(df: pd.DataFrame, path: Path) -> list[str]:
    if PASSAGE_COL in df.columns:
        col = PASSAGE_COL
    else:
        cols_lower = {c.lower(): c for c in df.columns}
        if PASSAGE_COL.lower() not in cols_lower:
            raise KeyError(f"Missing column '{PASSAGE_COL}' in {path}")
        col = cols_lower[PASSAGE_COL.lower()]
    return df[col].dropna().astype(str).tolist()


def passages_from_jsonl(path: Path) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if PASSAGE_COL in obj and obj[PASSAGE_COL] is not None:
                out.append(str(obj[PASSAGE_COL]))
    return out


def avg_tokens_for_folder(folder: Path) -> tuple[int, float]:
    """
    Returns (n_passages, avg_tokens). If no passages, returns (0, 0.0).
    """
    total_tokens = 0
    total_passages = 0

    for path in iter_files(folder):
        try:
            if path.suffix.lower() == ".tsv":
                df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_MINIMAL, low_memory=False)
                passages = passages_from_df(df, path)
            elif path.suffix.lower() == ".csv":
                df = pd.read_csv(path, low_memory=False)
                passages = passages_from_df(df, path)
            elif path.suffix.lower() == ".jsonl":
                passages = passages_from_jsonl(path)
            elif path.suffix.lower() == ".json":
                obj = json.loads(path.read_text(encoding="utf-8"))
                passages = []
                if isinstance(obj, list):
                    for row in obj:
                        if isinstance(row, dict) and PASSAGE_COL in row and row[PASSAGE_COL] is not None:
                            passages.append(str(row[PASSAGE_COL]))
            else:
                continue
        except KeyError:
            continue
        except Exception as e:
            print(f"[WARN] failed reading {path}: {e}")
            continue

        for text in passages:
            total_tokens += len(tokenize(text))
            total_passages += 1

    if total_passages == 0:
        return (0, 0.0)

    return (total_passages, total_tokens / total_passages)


# =========================
# Plotting
# =========================
def plot_bar(df: pd.DataFrame) -> None:
    apply_paper_fmt()

    # keep only langs with data
    df = df[df["n_passages"] > 0].copy()

    # preserve profile order on x axis
    df["lang"] = pd.Categorical(df["lang"], categories=df["lang"].tolist(), ordered=True)

    # figure size scales with #langs
    w = max(3.0, 0.35 * len(df))
    plt.figure(figsize=(w, 2.6))

    ax = sns.barplot(data=df, x="lang", y="avg_tokens", errorbar=None)

    ax.set_xlabel("")
    ax.set_ylabel("Avg tokens per passage")

    ax.set_title(f"TREC-DL {YEAR} • Profile: {PROFILE}")

    # rotate tick labels for readability
    plt.xticks(rotation=45, ha="right")

    # tight layout + save
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
            rows.append({"lang": lang_variant, "n_passages": 0, "avg_tokens": 0.0})
            continue

        n, avg = avg_tokens_for_folder(folder)
        rows.append({"lang": lang_variant, "n_passages": n, "avg_tokens": avg})
        print(f"{lang_variant:>15}  n={n:<8}  avg_tokens={avg:.6f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote: {OUT_CSV}")

    plot_bar(out)


if __name__ == "__main__":
    main()
