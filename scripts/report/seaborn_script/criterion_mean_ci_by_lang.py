#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Config
# =========================
TREC_DL_YEAR = "2022"
MODEL = "qwen3-32b-v1"

# If empty, auto-discover from filenames in CRITERION_DIR
LANGS: list[str] = ["raw", "eng", "ar", "ru", "fr", "vi", "th", "ga", "sw"]
CRITERIA: list[str] = ["contextuality"]

VALID_LABELS = {0, 1, 2, 3}

# Bootstrap CI settings
CI_LEVEL = 95
N_BOOT = 2000
RNG_SEED = 7

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]

CRITERION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "llm_label"
    / f"trec_dl_{TREC_DL_YEAR}"
    / MODEL
    / "criterion"
)

FIG_DIR = PROJECT_ROOT / "figures" / TREC_DL_YEAR / MODEL / "criterion_mean_ci"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FIG_BASE = FIG_DIR / f"{MODEL}_trecdl_{TREC_DL_YEAR}_criterion_mean_ci"
FIG_PATH_PNG = FIG_BASE.with_suffix(".png")
FIG_PATH_SVG = FIG_BASE.with_suffix(".svg")
SUMMARY_CSV = FIG_BASE.with_suffix(".csv")


# =========================
# Helpers
# =========================

def parse_lang_criterion(path: Path) -> tuple[str, str] | None:
    """
    Parse:
      {MODEL}_trecdl_{YEAR}_{lang}_{criterion}_labels.csv
    """
    prefix = f"{MODEL}_trecdl_{TREC_DL_YEAR}_"
    if not path.name.startswith(prefix) or not path.name.endswith("_labels.csv"):
        return None

    stem = path.name[: -len("_labels.csv")]
    rest = stem[len(prefix) :]
    if "_" not in rest:
        return None

    lang, criterion = rest.rsplit("_", 1)
    return lang, criterion


def discover_langs_criteria() -> tuple[list[str], list[str]]:
    langs: set[str] = set()
    criteria: set[str] = set()

    for path in CRITERION_DIR.glob(f"{MODEL}_trecdl_{TREC_DL_YEAR}_*_labels.csv"):
        parsed = parse_lang_criterion(path)
        if parsed is None:
            continue
        lang, criterion = parsed
        langs.add(lang)
        criteria.add(criterion)

    return sorted(langs), sorted(criteria)


def pick_pid_col(columns: list[str]) -> str:
    candidates = ["pid", "pid_qrels", "pid_resolved", "docid", "passage_id", "doc_id"]
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(
        f"No pid-like column found. Need one of {candidates}. Columns={columns}"
    )


def load_df(lang: str, criterion: str) -> tuple[pd.DataFrame, str]:
    fname = f"{MODEL}_trecdl_{TREC_DL_YEAR}_{lang}_{criterion}_labels.csv"
    path = CRITERION_DIR / fname
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if criterion not in df.columns:
        raise ValueError(f"Column {criterion!r} missing in {path}")
    if "qid" not in df.columns:
        raise ValueError(f"Column 'qid' missing in {path}")

    pid_col = pick_pid_col(list(df.columns))
    return df, pid_col


def build_common_valid_keys(langs: list[str], criteria: list[str]) -> set[str]:
    """
    Compute keys (qid+pid) that are valid across ALL files considered.
    Any missing/invalid score in any file removes that key globally.
    """
    common_keys: set[str] | None = None
    missing_files: list[str] = []

    for lang in langs:
        for criterion in criteria:
            try:
                df, pid_col = load_df(lang, criterion)
            except FileNotFoundError as exc:
                missing_files.append(str(exc))
                continue

            scores = pd.to_numeric(df[criterion], errors="coerce")
            valid_mask = scores.isin(VALID_LABELS)

            keys = (
                df["qid"].astype(str).str.strip()
                + "|"
                + df[pid_col].astype(str).str.strip()
            )
            valid_keys = set(keys[valid_mask].tolist())

            if common_keys is None:
                common_keys = valid_keys
            else:
                common_keys = common_keys.intersection(valid_keys)

            if not common_keys:
                return set()

    if missing_files:
        for path in missing_files:
            print(f"[WARN] Missing file, skipping: {path}")

    return common_keys or set()


def load_scores(lang: str, criterion: str, common_keys: set[str]) -> np.ndarray | None:
    try:
        df, pid_col = load_df(lang, criterion)
    except FileNotFoundError as exc:
        print(f"[WARN] Missing file, skipping: {exc}")
        return None

    keys = (
        df["qid"].astype(str).str.strip()
        + "|"
        + df[pid_col].astype(str).str.strip()
    )

    scores = pd.to_numeric(df[criterion], errors="coerce")
    valid_mask = scores.isin(VALID_LABELS)
    keep_mask = valid_mask & keys.isin(common_keys)

    s = scores[keep_mask].dropna()
    return s.to_numpy(dtype=float)


def bootstrap_ci_mean(
    x: np.ndarray,
    *,
    ci_level: int,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    Percentile bootstrap CI for the mean.
    Returns (mean, lo, hi).
    """
    x = x.astype(float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))

    mean = float(x.mean())
    if x.size == 1:
        return (mean, mean, mean)

    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot_means = x[idx].mean(axis=1)

    alpha = 100 - ci_level
    lo = float(np.percentile(boot_means, alpha / 2))
    hi = float(np.percentile(boot_means, 100 - alpha / 2))
    return (mean, lo, hi)


def summarize_ci(langs: list[str], criteria: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict[str, object]] = []

    common_keys = build_common_valid_keys(langs, criteria)
    if not common_keys:
        raise ValueError("No shared valid (qid,pid) keys across all files.")
    print(f"[INFO] Shared valid keys across all files: {len(common_keys)}")

    for lang in langs:
        for criterion in criteria:
            x = load_scores(lang, criterion, common_keys)
            if x is None or x.size == 0:
                continue

            mean, lo, hi = bootstrap_ci_mean(
                x, ci_level=CI_LEVEL, n_boot=N_BOOT, rng=rng
            )
            if np.isnan(mean):
                continue

            rows.append(
                {
                    "language": lang,
                    "criterion": criterion,
                    "mean": mean,
                    "lo": lo,
                    "hi": hi,
                    "n": int(np.isfinite(x).sum()),
                }
            )

    if not rows:
        raise ValueError("After cleaning, no data remained to summarize.")

    return pd.DataFrame(rows)


def plot_ci(summary_df: pd.DataFrame, langs: list[str], criteria: list[str]) -> None:
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))

    x_base = np.arange(len(criteria), dtype=float)
    n_langs = len(langs)
    if n_langs <= 1:
        offsets = np.array([0.0])
    else:
        dodge = 0.35
        offsets = np.linspace(-dodge, dodge, n_langs)

    colors = sns.color_palette("tab10", n_langs)

    for li, lang in enumerate(langs):
        sub = summary_df[summary_df["language"] == lang]
        if sub.empty:
            continue

        xs: list[float] = []
        ys: list[float] = []
        yerr_lo: list[float] = []
        yerr_hi: list[float] = []

        for ci, criterion in enumerate(criteria):
            row = sub[sub["criterion"] == criterion]
            if row.empty:
                continue

            r = row.iloc[0]
            mean = float(r["mean"])
            lo = float(r["lo"])
            hi = float(r["hi"])

            xs.append(float(x_base[ci] + offsets[li]))
            ys.append(mean)
            yerr_lo.append(mean - lo)
            yerr_hi.append(hi - mean)

        if not xs:
            continue

        ax.errorbar(
            xs,
            ys,
            yerr=[yerr_lo, yerr_hi],
            fmt="o",
            capsize=3,
            elinewidth=1.2,
            markersize=4,
            color=colors[li],
            label=lang,
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels([c.capitalize() for c in criteria], rotation=0)
    ax.set_xlabel("Criterion")
    ax.set_ylabel("Mean score")
    ax.set_title(
        f"Mean criterion score by language (95% CI)\n{MODEL}, trec_dl_{TREC_DL_YEAR}"
    )

    if VALID_LABELS == {0, 1, 2, 3}:
        ax.set_ylim(-0.1, 3.1)

    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9")
    ax.grid(axis="x", visible=False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=max(1, min(6, n_langs)),
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(FIG_PATH_PNG, dpi=300)
    plt.savefig(FIG_PATH_SVG)
    plt.close()

    print(f"[DONE] Saved PNG {FIG_PATH_PNG}")
    print(f"[DONE] Saved SVG {FIG_PATH_SVG}")


def main() -> None:
    if not CRITERION_DIR.exists():
        print(f"[FATAL] Criterion dir not found: {CRITERION_DIR}")
        sys.exit(1)

    langs = LANGS
    criteria = CRITERIA

    if not langs or not criteria:
        auto_langs, auto_criteria = discover_langs_criteria()
        if not langs:
            langs = auto_langs
        if not criteria:
            criteria = auto_criteria

    if not langs:
        raise ValueError("No languages found. Check LANGS or filenames.")
    if not criteria:
        raise ValueError("No criteria found. Check CRITERIA or filenames.")

    summary_df = summarize_ci(langs, criteria)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")
    print(f"[INFO] Wrote summary CSV {SUMMARY_CSV}")

    plot_ci(summary_df, langs, criteria)


if __name__ == "__main__":
    main()
