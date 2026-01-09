from pathlib import Path

import pandas as pd
import tiktoken
enc = tiktoken.get_encoding("o200k_harmony")
assert enc.decode(enc.encode("hello world")) == "hello world"


# Config
LANG = "ga"
YEAR = "2022"
OUTPUT_DIR = Path("outputs") / "token"
OUTPUT_CSV = OUTPUT_DIR / f"passage_tokens_{YEAR}_{LANG}.csv"
PART_MIN = 1
PART_MAX = 6


# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-oss-120b")

text = "hello world"
token_ids = enc.encode(text)
print("token_ids:", token_ids)
print("tokens:", [enc.decode([t]) for t in token_ids])


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


df = load_passages_df(YEAR, LANG)
df["orig_token"] = df["passage"].apply(lambda p: len(enc.encode(p)))
df["inj_token"] = df["passage_injected"].apply(lambda p: len(enc.encode(p)))
df["delta_token"] = df["inj_token"] - df["orig_token"]
df["fertility_score"] = df["inj_token"] / df["orig_token"].replace(0, pd.NA)
out_df = df[
    [
        "pid",
        "qid",
        "passage",
        "passage_injected",
        "orig_token",
        "inj_token",
        "delta_token",
        "fertility_score",
    ]
]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUTPUT_CSV, index=False)

print("passage_count:", len(df))
print("output_csv:", OUTPUT_CSV)
print("passage_sample:", df["passage"].iloc[0][:200])
