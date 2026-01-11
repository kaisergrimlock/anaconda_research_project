from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pandas as pd
import spacy
import textdescriptives as td


# Config
LANG = "fr"
YEAR = "2022"
PART_MIN = 1
PART_MAX = 6
OUTPUT_DIR = Path("outputs")
OUTPUT_CSV = OUTPUT_DIR / f"textdescriptives_perplexity_{YEAR}_{LANG}.csv"
CONCURRENCY = 4
PROGRESS_EVERY = 100
SPACY_MODEL = "en_core_web_sm"
_NLP_LOCAL = threading.local()


def load_passages_df(year, lang, base_dir="retrieved"):
    data_dir = Path(base_dir) / f"trec_dl_{year}" / lang
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")

    csv_files = []
    for part in range(PART_MIN, PART_MAX + 1):
        csv_files.extend(data_dir.glob(f"*part{part}.csv"))
    csv_files = sorted(csv_files)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    frames = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        missing = [col for col in ("passage", "passage_injected") if col not in df.columns]
        if missing:
            raise ValueError(
                f"Expected columns {missing} in {csv_path}, found: {list(df.columns)}"
            )
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def get_nlp():
    nlp = getattr(_NLP_LOCAL, "nlp", None)
    if nlp is None:
        nlp = spacy.load(SPACY_MODEL)
        if "textdescriptives/information_theory" not in nlp.pipe_names:
            nlp.add_pipe("textdescriptives/information_theory")
        _NLP_LOCAL.nlp = nlp
    return nlp


def compute_perplexity(text: str) -> float:
    nlp = get_nlp()
    doc = nlp(text)
    if not hasattr(doc._, "perplexity"):
        raise ValueError("Expected 'perplexity' attribute on doc._.")
    return float(doc._.perplexity)


def _compute_row_blocking(passage: str, passage_injected: str) -> tuple[float, float]:
    p_passage = compute_perplexity(passage)
    p_injected = compute_perplexity(passage_injected)
    return p_passage, p_injected


async def _compute_row(
    sem: asyncio.Semaphore,
    idx: int,
    passage: str,
    passage_injected: str,
) -> tuple[int, float, float]:
    async with sem:
        p_passage, p_injected = await asyncio.to_thread(
            _compute_row_blocking, passage, passage_injected
        )
        return idx, p_passage, p_injected


async def main() -> None:
    df = load_passages_df(YEAR, LANG)

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = []
    for idx, passage, passage_injected in df[
        ["passage", "passage_injected"]
    ].itertuples(index=True, name=None):
        tasks.append(
            asyncio.create_task(_compute_row(sem, idx, passage, passage_injected))
        )

    perplexity_passage = [None] * len(df)
    perplexity_injected = [None] * len(df)

    done = 0
    total = len(tasks)
    for task in asyncio.as_completed(tasks):
        idx, p_passage, p_injected = await task
        perplexity_passage[idx] = p_passage
        perplexity_injected[idx] = p_injected
        done += 1
        if done % PROGRESS_EVERY == 0 or done == total:
            print(f"[PROGRESS] {done}/{total}")

    df["perplexity_passage"] = perplexity_passage
    df["perplexity_injected"] = perplexity_injected
    df["perplexity_delta"] = df["perplexity_injected"] - df["perplexity_passage"]

    out_df = df[
        [
            "pid",
            "qid",
            "passage",
            "passage_injected",
            "perplexity_passage",
            "perplexity_injected",
            "perplexity_delta",
        ]
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print("output_csv:", OUTPUT_CSV)


if __name__ == "__main__":
    asyncio.run(main())
