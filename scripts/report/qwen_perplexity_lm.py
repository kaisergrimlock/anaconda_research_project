from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Config
LANG = "eng_first"
YEAR = "2022"
PART_MIN = 1
PART_MAX = 6
MODEL_ID = "Qwen/Qwen3-32B"
OUTPUT_DIR = Path("outputs") / "qwen_perplexity"
OUTPUT_CSV = OUTPUT_DIR / f"qwen_perplexity_{YEAR}_{LANG}.csv"
MAX_LENGTH = 2048
STRIDE = 1024
PROGRESS_EVERY = 50
CACHE_ROOT = Path("D:/hf_cache")


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


def get_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def compute_perplexity(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    *,
    max_length: int,
    stride: int,
) -> float:
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]
    if input_ids.numel() == 0:
        return float("nan")

    device = get_device(model)
    nlls = []
    seq_len = input_ids.size(0)
    for i in range(0, seq_len, stride):
        begin = max(i + stride - max_length, 0)
        end = min(i + stride, seq_len)
        trg_len = end - i
        input_ids_slice = input_ids[begin:end].unsqueeze(0).to(device)
        target_ids = input_ids_slice.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids_slice, labels=target_ids)
        nlls.append(outputs.loss * trg_len)

    ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
    return float(ppl.item())


def main() -> None:
    os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_ROOT / "transformers"))
    os.environ.setdefault("HF_HUB_CACHE", str(CACHE_ROOT / "hub"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
    model.eval()

    df = load_passages_df(YEAR, LANG)

    perplexity_passage = []
    perplexity_injected = []
    total = len(df)
    for idx, (passage, passage_injected) in enumerate(
        df[["passage", "passage_injected"]].itertuples(index=False, name=None),
        start=1,
    ):
        p_passage = compute_perplexity(
            passage, tokenizer, model, max_length=MAX_LENGTH, stride=STRIDE
        )
        p_injected = compute_perplexity(
            passage_injected, tokenizer, model, max_length=MAX_LENGTH, stride=STRIDE
        )
        perplexity_passage.append(p_passage)
        perplexity_injected.append(p_injected)

        if idx % PROGRESS_EVERY == 0 or idx == total:
            print(f"[PROGRESS] {idx}/{total}")

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
    main()
