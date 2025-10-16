#!/usr/bin/env python3
from __future__ import annotations
import csv, sys, random
from pathlib import Path
from typing import Dict

# ====== EDIT THESE (no CLI) ==========================================
INPUT_NONREL = Path("outputs/queries/non_relevant/first_nonrelevant_per_query.csv")  # must contain: query,passage
EXPANDED_CSV = Path("outputs/queries/trec_dl_2023_expanded_queries.csv")  # columns: query,passage (passage = verbose)
OUTPUT_CSV   = Path("outputs/queries/verbose_injected.csv")             # writes: query,passage (injected)
RANDOM_SEED  = 42  # set None for non-deterministic
# =====================================================================

COL_QUERY   = "query"
COL_PASSAGE = "passage"

def _bump_field_limit():
    try:
        limit = min(2_000_000_000, getattr(sys, "maxsize", 2_000_000_000))
        while limit >= 131072:
            try:
                csv.field_size_limit(limit); return
            except OverflowError:
                limit //= 2
    except Exception:
        pass
_bump_field_limit()

def _load_verbose_by_query(path: Path) -> Dict[str, str]:
    if not path.exists():
        sys.exit(f"[FATAL] Expanded CSV not found: {path}")
    by_q: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, skipinitialspace=True)
        hdr = r.fieldnames or []
        if COL_QUERY not in hdr or COL_PASSAGE not in hdr:
            sys.exit(f"[FATAL] Expanded CSV must have columns ['{COL_QUERY}','{COL_PASSAGE}']. Header={hdr}")
        for row in r:
            q = (row.get(COL_QUERY) or "").strip()
            v = (row.get(COL_PASSAGE) or "").strip()
            if q and v:
                by_q[q] = v
    return by_q

def _inject_random(base_text: str, injection: str, rng: random.Random) -> str:
    if not base_text:
        return injection
    toks = base_text.split()
    if not toks:
        return injection
    k = rng.randint(0, len(toks))  # 0..len
    return " ".join(toks[:k] + [injection] + toks[k:])

def main():
    rng = random.Random(RANDOM_SEED) if RANDOM_SEED is not None else random.Random()

    if not INPUT_NONREL.exists():
        sys.exit(f"[FATAL] Input non-relevant CSV not found: {INPUT_NONREL}")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load the already-created non-relevant rows (query -> base passage)
    base_by_query: Dict[str, str] = {}
    with INPUT_NONREL.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, skipinitialspace=True)
        hdr = r.fieldnames or []
        if COL_QUERY not in hdr or COL_PASSAGE not in hdr:
            sys.exit(f"[FATAL] Non-relevant CSV must have columns ['{COL_QUERY}','{COL_PASSAGE}']. Header={hdr}")
        for row in r:
            q = (row.get(COL_QUERY) or "").strip()
            p = (row.get(COL_PASSAGE) or "").strip()
            if q and p:
                # keep first seen per query
                base_by_query.setdefault(q, p)

    if not base_by_query:
        sys.exit("[FATAL] No (query,passage) rows found in the non-relevant input file.")

    # 2) Load verbose expansions (query -> verbose text)
    verbose_by_query = _load_verbose_by_query(EXPANDED_CSV)

    # 3) Inject and write ONLY (query, injected passage)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=[COL_QUERY, COL_PASSAGE])
        w.writeheader()

        total = 0
        injected = 0
        for q, base_p in base_by_query.items():
            total += 1
            verbose = verbose_by_query.get(q, "")
            out_p = _inject_random(base_p, verbose, rng) if verbose else base_p
            injected += 1 if verbose else 0
            w.writerow({COL_QUERY: q, COL_PASSAGE: out_p})

    print(f"[DONE] wrote {OUTPUT_CSV} | queries={total} | injected={injected}")

if __name__ == "__main__":
    main()
