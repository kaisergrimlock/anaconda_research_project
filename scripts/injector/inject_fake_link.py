#!/usr/bin/env python3
from __future__ import annotations
import csv, sys, random
from pathlib import Path

# ====== EDIT THESE (no CLI) ==========================================
INPUT_CSV   = Path("outputs/queries/first_nonrelevant_per_query.csv")
OUTPUT_CSV  = Path("outputs/queries/first_nonrelevant_with_link.csv")
RANDOM_SEED = 42  # set None for non-deterministic insertion positions
# =====================================================================

# Required columns (exact order for output)
COL_QID         = "qid"
COL_QUERY       = "query"
COL_PID_QRELS   = "pid_qrels"
COL_PID_RES     = "pid_resolved"
COL_PASSAGE     = "passage"

def _bump_field_limit():
    try:
        import csv as _csv, sys as _sys
        limit = min(2_000_000_000, getattr(_sys, "maxsize", 2_000_000_000))
        while limit >= 131072:
            try:
                _csv.field_size_limit(limit); return
            except OverflowError:
                limit //= 2
    except Exception:
        pass

_bump_field_limit()

def _fake_wiki_link(query: str) -> str:
    # Replace spaces with underscores; leave other characters as-is
    slug = (query or "").strip().replace(" ", "_")
    return f"https://en.wikipedia.org/w/index.php?title={slug}"

def _inject_random(base_text: str, insertion: str, rng: random.Random) -> str:
    if not base_text:
        return insertion
    toks = base_text.split()
    if not toks:
        return insertion
    k = rng.randint(0, len(toks))  # position 0..len
    return " ".join(toks[:k] + [insertion] + toks[k:])

def main():
    if not INPUT_CSV.exists():
        sys.exit(f"[FATAL] Input CSV not found: {INPUT_CSV}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED) if RANDOM_SEED is not None else random.Random()

    with INPUT_CSV.open("r", encoding="utf-8", newline="") as fin, \
         OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin, skipinitialspace=True)
        hdr = reader.fieldnames or []

        required = [COL_QID, COL_QUERY, COL_PID_QRELS, COL_PID_RES, COL_PASSAGE]
        missing = [c for c in required if c not in hdr]
        if missing:
            sys.exit(f"[FATAL] Input CSV must have columns {required}. Header={hdr}")

        writer = csv.DictWriter(
            fout,
            fieldnames=[COL_QID, COL_QUERY, COL_PID_QRELS, COL_PID_RES, COL_PASSAGE]
        )
        writer.writeheader()

        total = injected = 0
        for row in reader:
            total += 1
            qid   = (row.get(COL_QID) or "").strip()
            query = (row.get(COL_QUERY) or "").strip()
            pid_q = (row.get(COL_PID_QRELS) or "").strip()
            pid_r = (row.get(COL_PID_RES) or "").strip()
            psg   = (row.get(COL_PASSAGE) or "").strip()

            link = _fake_wiki_link(query)
            new_p = _inject_random(psg, link, rng)
            injected += 1

            writer.writerow({
                COL_QID: qid,
                COL_QUERY: query,
                COL_PID_QRELS: pid_q,
                COL_PID_RES: pid_r,
                COL_PASSAGE: new_p
            })

    print(f"[DONE] wrote {OUTPUT_CSV} | rows processed={total}, injected={injected}")

if __name__ == "__main__":
    main()
