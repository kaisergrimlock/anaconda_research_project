#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    import regex as re  # pip install regex
except ImportError as e:
    raise SystemExit("Missing dependency: regex\nInstall it with: pip install regex") from e

# =========================
# CONFIG
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

YEAR = "2021"
INPUT_CSV = PROJECT_ROOT / "outputs" / "queries" / f"queries_trecdl_{YEAR}_all_lang.csv"
OUT_CSV = PROJECT_ROOT / "outputs" / "queries" / f"queries_trecdl_{YEAR}_all_lang_bowset.csv"

LANG_COLS = [
    "eng", "vi", "th", "fr", "ru", "ar", "he", "sw", "ga", "zh", "hi",
    # add any others you have
]

# Unicode token pattern:
# - sequences of letters/numbers across any script
# - keeps internal apostrophes (') and right single quote (’)
TOKEN_RE = re.compile(r"[\p{L}\p{N}]+(?:[’'][\p{L}\p{N}]+)?", re.UNICODE)


def bow_set(text: str) -> list[str]:
    # casefold is stronger/more Unicode-correct than lower()
    toks = [t.casefold() for t in TOKEN_RE.findall(text)]
    return sorted(set(toks))


def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit(f"[ERROR] Input not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    missing = [c for c in LANG_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] Missing columns in CSV: {missing}\nGot: {list(df.columns)}")

    def transform_cell(x) -> str:
        txt = "" if pd.isna(x) else str(x)
        return json.dumps(bow_set(txt), ensure_ascii=False)

    for col in LANG_COLS:
        df[col] = df[col].map(transform_cell)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[DONE] wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
