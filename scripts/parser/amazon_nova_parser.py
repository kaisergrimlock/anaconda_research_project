#!/usr/bin/env python3
"""
Fill LLM 'O' scores into a CSV by docid, robust to pure-JSON responses like:
    "[{\"M\": 3, \"T\": 3, \"O\": 3}]"

CSV header supported:
  - docid, nist_rel, llm_rel
  - docid, rel_reference, rel_model
Only fills empty llm_rel/rel_model.
"""

from __future__ import annotations
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List
import shutil

# =========================
# CONFIG — EDIT THESE
# =========================
LOG_PATH   = Path(r"outputs/trec_dl/logs/20250920_210623_llm_responses_us.amazon.nova-lite-v1_0_top2.json")       # file or directory
TARGET_CSV = Path(r"outputs/trec_dl_llm_label/processed/utility/20250920_223947/doc_rel_compare_us.amazon.nova-lite-v1_0_top2.csv")
MAKE_BACKUP = True
DRY_RUN = False
# =========================

O_REGEX = re.compile(r'"O"\s*:\s*([0-3])')

def _candidate_text_blocks(item: dict) -> Iterable[str]:
    """Yield any text fields that might contain the JSON (or JSON-ish) with M/T/O."""
    txt = item.get("response_text")
    if isinstance(txt, str) and txt.strip():
        yield txt

    fr = item.get("full_response") or {}
    out = (fr.get("output") or {})
    msg = (out.get("message") or {})
    content = msg.get("content")
    if isinstance(content, list):
        for c in content:
            t = c.get("text")
            if isinstance(t, str) and t.strip():
                yield t

def _try_parse_json_for_O(text: str) -> Optional[int]:
    """
    Tries to extract O by parsing JSON. Handles cases where text is:
      - a JSON array string:    [{"M":3,"T":3,"O":3}]
      - a JSON object string:   {"M":3,"T":3,"O":3}
      - a stringified JSON:     "[{\"M\":3,\"T\":3,\"O\":3}]"
    Returns int or None.
    """
    def load_any(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    # 1) direct load
    obj = load_any(text)
    # 2) if it's still a string that looks like JSON, try once more
    if isinstance(obj, str):
        obj2 = load_any(obj)
        if obj2 is not None:
            obj = obj2

    # Now inspect
    if isinstance(obj, list) and obj:
        first = obj[0]
        if isinstance(first, dict) and "O" in first:
            try:
                return int(first["O"])
            except Exception:
                return None
    if isinstance(obj, dict) and "O" in obj:
        try:
            return int(obj["O"])
        except Exception:
            return None
    return None

def _regex_O(text: str) -> Optional[int]:
    m = O_REGEX.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _extract_O_from_item(item: dict) -> Optional[int]:
    """Try JSON parse first; if it fails, fall back to regex."""
    for txt in _candidate_text_blocks(item):
        o = _try_parse_json_for_O(txt)
        if o is not None:
            return o
    # fallback: regex scan any block
    for txt in _candidate_text_blocks(item):
        o = _regex_O(txt)
        if o is not None:
            return o
    return None

def _extract_docid(item: dict) -> Optional[str]:
    d = item.get("docid")
    return str(d).strip() if d is not None else None

def read_logs_collect_O_scores(path: Path) -> Dict[str, int]:
    """Read one JSON file or all .json files in a folder. Return docid -> O."""
    files: List[Path]
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

        # Normalize to list
        if isinstance(data, dict):
            # common wrapper patterns
            if "records" in data and isinstance(data["records"], list):
                data = data["records"]
            else:
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
            o = _extract_O_from_item(item)
            if o is not None:
                mapping[docid] = o
    return mapping

def load_csv_rows(csv_path: Path) -> Tuple[List[dict], List[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [row for row in reader]
    if "docid" not in fieldnames:
        raise SystemExit("[ERROR] CSV missing required column: docid")
    return rows, fieldnames

def pick_target_column(fieldnames: List[str]) -> str:
    if "llm_rel" in fieldnames:
        return "llm_rel"
    if "rel_model" in fieldnames:
        return "rel_model"
    raise SystemExit(
        "[ERROR] No writable LLM column found. Expected one of: 'llm_rel' or 'rel_model'.\n"
        "Add one of those columns to your CSV header."
    )

def write_csv_rows(csv_path: Path, rows: List[dict], fieldnames: List[str]) -> None:
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
    target_col = pick_target_column(fieldnames)

    updates = 0
    for row in rows:
        docid = str(row.get("docid", "")).strip()
        cur = (row.get(target_col) or "").strip()
        if cur == "" and docid in docid_to_O:
            row[target_col] = str(docid_to_O[docid])
            updates += 1

    print(f"[INFO] Rows updated in '{target_col}': {updates}")
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
