#!/usr/bin/env python3
"""
Fill llm_rel in a CSV from logged LLM responses by extracting the O score.

Input CSV format (must exist):
    docid,nist_rel,llm_rel
    1726,0,

Behavior:
- Only fills rows where llm_rel is empty.
- Pulls the "O" score (0..3) from each log entry.
- Supports either a single .json log file OR a directory of .json files.

Tested with logs shaped like the example in the user message.
"""

from __future__ import annotations
import json
import re
import shutil
from pathlib import Path
import csv
from typing import Dict, Iterable, Tuple, Optional, Any

# =========================
# CONFIG — EDIT THESE
# =========================
LOG_PATH   = Path(r"outputs/trec_dl/logs/20250920_210834_llm_responses_anthropic.claude-3-5-haiku-20241022-v1_0_top2.json")   # file OR directory
TARGET_CSV = Path(r"outputs/trec_dl_llm_label/processed/utility/20250920_223947/doc_rel_compare_anthropic.claude-3-5-haiku-20241022-v1_0_top2.csv")  # your CSV to update
MAKE_BACKUP = True   # create TARGET_CSV.bak before writing
DRY_RUN = False      # if True, do not write changes; just report what would change
# =========================


O_REGEX = re.compile(r'"O"\s*:\s*([0-3])')

def _get_text_blocks_from_item(item: dict) -> Iterable[str]:
    """Yield any text fields that might contain the JSON with M/T/O."""
    # 1) Top-level 'response_text' if present
    txt = item.get("response_text")
    if isinstance(txt, str) and txt.strip():
        yield txt

    # 2) Inside 'full_response' (Bedrock-style shape in your sample)
    fr = item.get("full_response") or {}
    out = fr.get("output") or {}
    msg = out.get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):
        for c in content:
            t = c.get("text")
            if isinstance(t, str) and t.strip():
                yield t

def _extract_O_from_text(text: str) -> Optional[int]:
    """
    Find the first occurrence of `"O": <digit>` (0..3) and return it as int.
    Works even if the text has extra prose before/after the JSON.
    """
    m = O_REGEX.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _extract_docid(item: dict) -> Optional[str]:
    """Pull docid as a string; returns None if missing."""
    docid = item.get("docid")
    if docid is None:
        return None
    return str(docid).strip()

def read_logs_collect_O_scores(path: Path) -> Dict[str, int]:
    """
    Reads one JSON file (array of objects) or all .json files in a folder.
    Returns a mapping: docid -> O score (last one wins if multiples).
    """
    files: Iterable[Path]
    if path.is_dir():
        files = sorted(p for p in path.glob("*.json") if p.is_file())
    else:
        files = [path]

    mapping: Dict[str, int] = {}

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Skipping {f} (JSON load failed): {e}")
            continue

        # Expect a list/array of entries
        if isinstance(data, dict):
            # sometimes logs wrap entries in a key; try common cases
            if "records" in data and isinstance(data["records"], list):
                data = data["records"]
            else:
                # fall back: treat as single entry
                data = [data]

        if not isinstance(data, list):
            print(f"[WARN] {f} isn't a list of entries; skipping.")
            continue

        for item in data:
            if not isinstance(item, dict):
                continue
            docid = _extract_docid(item)
            if not docid:
                continue

            O_val: Optional[int] = None
            # Try every candidate text block until we get an O
            for txt in _get_text_blocks_from_item(item):
                O_val = _extract_O_from_text(txt)
                if O_val is not None:
                    break

            if O_val is not None:
                mapping[docid] = O_val

    return mapping

def load_csv_rows(csv_path: Path) -> Tuple[list, list]:
    """Returns (rows, fieldnames)."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [row for row in reader]
    # enforce required columns exist
    for col in ["docid", "nist_rel", "llm_rel"]:
        if col not in fieldnames:
            raise SystemExit(f"[ERROR] CSV missing required column: {col}")
    return rows, fieldnames

def write_csv_rows(csv_path: Path, rows: list, fieldnames: list) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def main() -> None:
    if not TARGET_CSV.exists():
        raise SystemExit(f"[ERROR] CSV not found: {TARGET_CSV}")

    docid_to_O = read_logs_collect_O_scores(LOG_PATH)
    if not docid_to_O:
        print("[WARN] No O scores found in logs — nothing to do.")
        return

    rows, fieldnames = load_csv_rows(TARGET_CSV)

    updates = 0
    for row in rows:
        docid = str(row.get("docid", "")).strip()
        llm_rel = (row.get("llm_rel") or "").strip()
        if llm_rel == "" and docid in docid_to_O:
            row["llm_rel"] = str(docid_to_O[docid])
            updates += 1

    print(f"[INFO] Rows updated: {updates}")

    if updates == 0:
        print("[INFO] No changes required. Exiting.")
        return

    if DRY_RUN:
        print("[DRY RUN] Changes not written. Set DRY_RUN = False to apply.")
        return

    if MAKE_BACKUP:
        backup = TARGET_CSV.with_suffix(TARGET_CSV.suffix + ".bak")
        shutil.copy2(TARGET_CSV, backup)
        print(f"[INFO] Backup written: {backup}")

    write_csv_rows(TARGET_CSV, rows, fieldnames)
    print(f"[OK] CSV updated: {TARGET_CSV}")

if __name__ == "__main__":
    main()
