#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

# =========================================================
# Config
# =========================================================

# Existing script in the same folder
TARGET_SCRIPT = "trec_label_concurrent.py"

# Languages only
LANGUAGES = [
    "ru_instruct",
    "zh_instruct",
    "ga_instruct",
    "ar_instruct"
]

# Shared part range for all languages
START_PART = 0
END_PART = 0

# =========================================================
# Load target script dynamically
# =========================================================

THIS_DIR = Path(__file__).resolve().parent
TARGET_PATH = THIS_DIR / TARGET_SCRIPT

if not TARGET_PATH.exists():
    raise FileNotFoundError(f"Could not find target script: {TARGET_PATH}")

spec = importlib.util.spec_from_file_location("label_runner_target", TARGET_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def set_run_globals(mod, lang: str, start_part: int, end_part: int) -> None:
    mod.LANG = lang
    mod.START_PART = start_part
    mod.END_PART = end_part

    if lang == "raw":
        mod.PART_DIR = Path(f"retrieved/trec_dl_{mod.TREC_DL_YEAR}/judged/")
    else:
        mod.PART_DIR = Path(f"retrieved/trec_dl_{mod.TREC_DL_YEAR}/{lang}/")

    mod.PART_PATTERN = f"all_topics_trecdl_{mod.TREC_DL_YEAR}_part{{n}}.csv"


async def run_all_languages():
    for lang in LANGUAGES:
        print("\n" + "=" * 80)
        print(
            f"[RUNNER] Starting language={lang} | "
            f"parts={START_PART}..{END_PART}"
        )
        print("=" * 80)

        set_run_globals(module, lang, START_PART, END_PART)

        try:
            await module.main()
            print(f"[RUNNER] Finished language={lang}")
        except KeyboardInterrupt:
            print(f"\n[RUNNER] Interrupted while processing language={lang}")
            break
        except Exception as e:
            print(f"[RUNNER] Error while processing language={lang}: {e}")


if __name__ == "__main__":
    asyncio.run(run_all_languages())