#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ===============================================================
# Repo root (same pattern as your scripts)
# ===============================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]


# ===============================================================
# Defaults
# ===============================================================
DEFAULT_SUFFIX_CSV = PROJECT_ROOT / "scripts" / "label" / "suffix.csv"
DEFAULT_RUN = PROJECT_ROOT / "retrieved" / "aserini" / "run.msmarco-v2-passage-injected.2021.txt"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "anserini_rank_deltas"
DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ===============================================================
# Helpers
# ===============================================================
LANG_FROM_SUFFIX_RE = re.compile(r"_inj-([a-z]{2,3})-", re.IGNORECASE)

def language_from_suffix(suffix: str) -> Optional[str]:
    """
    Parse language code from suffix like:
      _inj-th-qp-first  -> th
      _inj-eng-qp-last  -> eng
      _inj-en-...       -> eng
    """
    m = LANG_FROM_SUFFIX_RE.search(str(suffix))
    if not m:
        return None
    lang = m.group(1).lower().strip()
    if lang in {"en", "eng"}:
        return "eng"
    return lang


def read_suffixes(path: Path) -> List[str]:
    """
    Reads suffix.csv with columns: folder,suffix
    Returns list of suffix strings, sorted longest-first.
    """
    if not path.exists():
        raise FileNotFoundError(f"suffix.csv not found: {path}")

    out: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, skipinitialspace=True)
        if r.fieldnames is None:
            raise ValueError(f"{path} has no header row.")
        if "suffix" not in r.fieldnames:
            raise ValueError(f"{path} must contain a 'suffix' column. Found: {r.fieldnames}")

        for row in r:
            s = (row.get("suffix", "") or "").strip()
            if s:
                out.append(s)

    out.sort(key=len, reverse=True)  # longest-first to avoid partial matches
    return out


def base_docno(docno: str, suffixes: List[str]) -> Optional[Tuple[str, str]]:
    """
    Returns (base_docno, matched_suffix) if docno ends with a known suffix, else None.
    """
    for suf in suffixes:
        if docno.endswith(suf):
            return docno[: -len(suf)], suf
    return None


def load_ranks(run_path: Path) -> Dict[Tuple[str, str], int]:
    """
    Parses a TREC run file:
      qid Q0 docno rank score tag
    Returns: (qid, docno) -> rank
    """
    if not run_path.exists():
        raise FileNotFoundError(f"Run file not found: {run_path}")

    ranks: Dict[Tuple[str, str], int] = {}
    with run_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 6:
                raise ValueError(
                    f"Bad run line {ln} in {run_path}: expected 6 fields, got {len(parts)}: {line}"
                )

            qid, _q0, docno, rank_s, _score, _tag = parts[:6]
            try:
                rank = int(rank_s)
            except ValueError:
                raise ValueError(f"Bad rank on line {ln} in {run_path}: {rank_s}")

            ranks[(qid, docno)] = rank

    return ranks


# ===============================================================
# Main
# ===============================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--suffix-csv", type=Path, default=DEFAULT_SUFFIX_CSV)
    ap.add_argument(
        "--out-raw",
        type=Path,
        default=None,
        help="Raw output CSV. Default: outputs/anserini_rank_deltas/<run_stem>.delta_rank.csv",
    )
    ap.add_argument(
        "--out-mean",
        type=Path,
        default=None,
        help="Mean output CSV (optional). Default: outputs/anserini_rank_deltas/<run_stem>.delta_rank_mean.csv",
    )
    ap.add_argument(
        "--write-mean",
        action="store_true",
        help="Also write the per-suffix mean file (*.delta_rank_mean.csv).",
    )
    args = ap.parse_args()

    suffixes = read_suffixes(args.suffix_csv)
    ranks = load_ranks(args.run)

    out_raw = args.out_raw or (DEFAULT_OUT_DIR / f"{args.run.stem}.delta_rank.csv")
    out_mean = args.out_mean or (DEFAULT_OUT_DIR / f"{args.run.stem}.delta_rank_mean.csv")
    out_raw.parent.mkdir(parents=True, exist_ok=True)

    # For optional mean aggregation: suffix -> (sum_delta, count)
    agg: Dict[str, Tuple[int, int]] = {}

    wrote_rows = 0
    missing_original = 0
    non_injected = 0
    missing_language = 0

    # Write RAW rows
    with out_raw.open("w", encoding="utf-8", newline="") as out_f:
        w = csv.writer(out_f)
        w.writerow([
            "qid",
            "language",
            "suffix",
            "docno_injected",
            "docno_original",
            "rank_injected",
            "rank_original",
            "delta_rank",
        ])

        for (qid, docno), r_inj in ranks.items():
            bd = base_docno(docno, suffixes)
            if bd is None:
                non_injected += 1
                continue

            base, suf = bd
            r_orig = ranks.get((qid, base))
            if r_orig is None:
                missing_original += 1
                continue

            # ΔRank = original − injected (positive means injected moved UP / better rank number)
            delta = r_orig - r_inj

            lang = language_from_suffix(suf)
            if lang is None:
                missing_language += 1
                # still write row; plotter can drop None if needed
                lang = ""

            w.writerow([qid, lang, suf, docno, base, r_inj, r_orig, delta])
            wrote_rows += 1

            if args.write_mean:
                s, c = agg.get(suf, (0, 0))
                agg[suf] = (s + delta, c + 1)

    print(f"[DONE] wrote raw deltas -> {out_raw}")
    print(f"[INFO] rows written: {wrote_rows:,}")
    print(f"[INFO] skipped non-injected docnos: {non_injected:,}")
    print(f"[WARN] injected docnos missing original in same run: {missing_original:,}")
    if missing_language:
        print(f"[WARN] could not parse language from suffix for {missing_language:,} row(s)")

    # Optional: write mean-per-suffix
    if args.write_mean:
        with out_mean.open("w", encoding="utf-8", newline="") as out_f:
            w = csv.writer(out_f)
            w.writerow(["suffix", "n_pairs", "delta_rank_mean"])

            for suf, (s, c) in sorted(agg.items(), key=lambda kv: (-kv[1][1], kv[0])):
                mean = (s / c) if c else 0.0
                w.writerow([suf, c, f"{mean:.6f}"])

        print(f"[DONE] wrote suffix means -> {out_mean}")


if __name__ == "__main__":
    main()
