import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =========================
# Repo root bootstrap
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEABORN_ROOT = THIS_FILE.parents[1]
if str(SEABORN_ROOT) not in sys.path:
    sys.path.insert(0, str(SEABORN_ROOT))

# =========================
# Imports
# =========================
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

from helpers.lang_profiles import get_langs
from scripts.csv_helpers import bump_field_limit
from helpers.output_writer import write_df

# ========================
# Parameters
# ========================
LABELS = [0, 1, 2, 3]
LANG_PROFILE = "lang"
LANGS: List[str] = get_langs(LANG_PROFILE)

# Choose metric:
#   mean_diff     -> signed difference (LLM - NIST)
#   mae_4pt       -> absolute error
#   disagree_rate -> binary disagreement
METRIC = "mean_diff"

# Positions to include in the ANOVA
POSITIONS = ["first", "last", "random"]

# If True, clamp negative omega squared values to 0 for reporting
CLIP_NEGATIVE_OMEGA = False

# =========================
# Config
# =========================
TREC_DL_YEAR = "2021"
LABEL_ROOT = Path("outputs/llm_label") / f"trec_dl_{TREC_DL_YEAR}"

OUT_DIR = Path("figures") / TREC_DL_YEAR / "anova" / f"all_models_all_{LANG_PROFILE}_position"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SAMPLES = OUT_DIR / "anova_samples_long_model_lang_position.csv"
OUT_ANOVA_CSV = OUT_DIR / "anova_model_language_position.csv"
OUT_ANOVA_TEX = OUT_DIR / f"anova_model_language_position_{TREC_DL_YEAR}.tex"

INVALID_CSV = Path(__file__).resolve().parent / f"invalid_{TREC_DL_YEAR}.csv"
KEY_COLS = ["qid", "pid"]

# =========================
# Discovery / loading
# =========================
def find_llm_files() -> Dict[str, List[Path]]:
    if not LABEL_ROOT.exists():
        raise FileNotFoundError(f"LABEL_ROOT not found: {LABEL_ROOT}")

    model_files: Dict[str, List[Path]] = {}
    for model_dir in LABEL_ROOT.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        csv_files = list(model_dir.glob(f"{model_name}_trecdl_{TREC_DL_YEAR}_*_labels.csv"))
        if csv_files:
            model_files[model_name] = csv_files

    return model_files


def get_lang_from_filename(file_path: Path, model: str) -> Optional[str]:
    """
    Extract full language token from filename, e.g.
      qwen3-32b-v1_trecdl_2021_eng_first_labels.csv -> eng_first
      qwen3-32b-v1_trecdl_2021_eng_labels.csv -> eng
    """
    fname = file_path.name
    pattern = rf"^{re.escape(model)}_trecdl_\d{{4}}_(.+?)_labels\.csv$"
    m = re.search(pattern, fname)
    return m.group(1) if m else None


def split_lang_and_position(lang_token: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Split a filename token like:
      eng_first  -> ("eng", "first")
      vi_last    -> ("vi", "last")
      ru_random  -> ("ru", "random")
      eng        -> ("eng", None)

    Returns:
      (base_lang, position)
    """
    if not lang_token:
        return None, None

    for pos in POSITIONS:
        suffix = f"_{pos}"
        if lang_token.endswith(suffix):
            base_lang = lang_token[: -len(suffix)]
            return base_lang, pos

    return lang_token, None


def load_invalid_keys(path: Path) -> set[tuple[int, str]]:
    if not path.exists():
        print(f"[INFO] No invalid file found: {path}")
        return set()

    inv = pd.read_csv(path)
    if not set(KEY_COLS).issubset(inv.columns):
        raise ValueError(f"{path} must contain columns {KEY_COLS}")

    inv = inv.dropna(subset=KEY_COLS).copy()
    inv["qid"] = pd.to_numeric(inv["qid"], errors="coerce")
    inv = inv.dropna(subset=["qid"]).copy()
    inv["qid"] = inv["qid"].astype(int)
    inv["pid"] = inv["pid"].astype(str)

    keys = set(inv[KEY_COLS].drop_duplicates().itertuples(index=False, name=None))
    print(f"[INFO] Loaded {len(keys)} invalid keys from {path}")
    return keys


def load_labels(file_path: Path, invalid_keys: set[tuple[int, str]]) -> pd.DataFrame:
    bump_field_limit()
    df = pd.read_csv(file_path)

    required = {"qid", "pid", "relevance", "llm_relevance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {file_path}")

    df["NIST"] = pd.to_numeric(df["relevance"], errors="coerce")
    df["LLM"] = pd.to_numeric(df["llm_relevance"], errors="coerce")

    df = df.dropna(subset=["qid", "pid", "NIST", "LLM"]).copy()
    df["qid"] = pd.to_numeric(df["qid"], errors="coerce")
    df = df.dropna(subset=["qid"]).copy()

    df["qid"] = df["qid"].astype(int)
    df["pid"] = df["pid"].astype(str)
    df["NIST"] = df["NIST"].astype(int)
    df["LLM"] = df["LLM"].astype(int)

    df = df[df["NIST"].isin(LABELS) & df["LLM"].isin(LABELS)].copy()

    if invalid_keys:
        keys = pd.Index(list(zip(df["qid"].to_numpy(), df["pid"].to_numpy())))
        df = df[~keys.isin(invalid_keys)].copy()

    return df


def per_row_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    base = df.drop_duplicates(subset=["qid", "pid"]).copy()

    if metric == "mean_diff":
        base["value"] = base["LLM"] - base["NIST"]
    elif metric == "mae_4pt":
        base["value"] = (base["LLM"] - base["NIST"]).abs()
    elif metric == "disagree_rate":
        base["value"] = (base["LLM"] != base["NIST"]).astype(float)
    else:
        raise ValueError("Unknown METRIC. Use 'mean_diff', 'mae_4pt', or 'disagree_rate'.")

    return base[["qid", "pid", "value"]]


# =========================
# Output helpers
# =========================
def safe_slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    fmt = df.copy()

    for col in fmt.columns:
        if pd.api.types.is_numeric_dtype(fmt[col]):
            fmt[col] = fmt[col].map(lambda x: f"{x:.6g}" if pd.notnull(x) else "")

    col_format = "l" + " r" * (len(fmt.columns) - 1)

    return fmt.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format=col_format,
    )


# =========================
# Effect size helpers
# =========================
def add_omega_squared(anova_df: pd.DataFrame, clip_negative: bool = False) -> pd.DataFrame:
    """
    Add omega squared to an ANOVA table returned by statsmodels.

    Formula:
        omega^2 = (SS_effect - df_effect * MS_error) / (SS_total + MS_error)
    """
    out = anova_df.copy()

    if "term" not in out.columns:
        raise ValueError("ANOVA dataframe must contain a 'term' column.")

    residual_mask = out["term"] == "Residual"
    if not residual_mask.any():
        raise ValueError("ANOVA dataframe must contain a Residual row.")

    if "mean_sq" not in out.columns:
        out["mean_sq"] = out["sum_sq"] / out["df"]

    ms_error = out.loc[residual_mask, "mean_sq"].iloc[0]
    ss_total = out["sum_sq"].sum()

    def calc(row: pd.Series) -> float:
        if row["term"] == "Residual":
            return np.nan
        value = (row["sum_sq"] - row["df"] * ms_error) / (ss_total + ms_error)
        if clip_negative:
            value = max(0.0, value)
        return value

    out["omega_sq"] = out.apply(calc, axis=1)
    return out


# =========================
# ANOVA
# =========================
def run_anova(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Three-way ANOVA:
        value ~ C(model) * C(lang) * C(position)

    This tests:
      - main effect of model
      - main effect of language
      - main effect of position
      - all two-way interactions
      - the three-way interaction
    """
    df = long_df.dropna(subset=["value", "model", "lang", "position"]).copy()

    df["model"] = df["model"].astype("category")
    df["lang"] = df["lang"].astype("category")
    df["position"] = df["position"].astype("category")

    fitted = ols("value ~ C(model) * C(lang) * C(position)", data=df).fit()

    # Type III is usually better once you care about higher-order interactions
    anova_df = sm.stats.anova_lm(fitted, typ=3).reset_index()
    anova_df = anova_df.rename(columns={"index": "term"})

    if "mean_sq" not in anova_df.columns:
        anova_df["mean_sq"] = anova_df["sum_sq"] / anova_df["df"]

    anova_df = add_omega_squared(anova_df, clip_negative=CLIP_NEGATIVE_OMEGA)
    return anova_df


# =========================
# Main
# =========================
def main() -> None:
    model_files = find_llm_files()
    invalid_keys = load_invalid_keys(INVALID_CSV)

    print(f"Found {len(model_files)} models under: {LABEL_ROOT}")

    rows: List[pd.DataFrame] = []
    skipped = 0

    for model, files in model_files.items():
        for f in files:
            lang_token = get_lang_from_filename(f, model)
            base_lang, position = split_lang_and_position(lang_token)

            # Only keep requested position-based files
            if base_lang not in LANGS:
                continue
            if position not in POSITIONS:
                continue

            try:
                df = load_labels(f, invalid_keys)
                perrow = per_row_metric(df, METRIC)
                if perrow.empty:
                    continue

                perrow["model"] = model
                perrow["lang"] = base_lang
                perrow["position"] = position

                rows.append(perrow[["model", "lang", "position", "qid", "pid", "value"]])

                print(f"[INFO] Loaded {model} | {base_lang} | {position}: {len(perrow)} samples")

            except Exception as e:
                skipped += 1
                print(f"[SKIP] {model} {lang_token} ({f.name}): {e}")

    if not rows:
        raise RuntimeError("No samples produced.")

    long_df = pd.concat(rows, ignore_index=True)

    # Keep only cells with at least 2 rows
    counts = (
        long_df.groupby(["model", "lang", "position"])
        .size()
        .reset_index(name="n")
    )
    keep = counts[counts["n"] >= 2][["model", "lang", "position"]]
    long_df = long_df.merge(keep, on=["model", "lang", "position"], how="inner")

    if long_df["model"].nunique() < 2:
        raise RuntimeError("Need at least 2 models for ANOVA.")
    if long_df["lang"].nunique() < 2:
        raise RuntimeError("Need at least 2 languages for ANOVA.")
    if long_df["position"].nunique() < 2:
        raise RuntimeError("Need at least 2 positions for ANOVA.")

    write_df(long_df, OUT_SAMPLES)

    anova_df = run_anova(long_df)
    write_df(anova_df, OUT_ANOVA_CSV)

    latex = to_latex_table(
        anova_df,
        caption=(
            f"Three-way ANOVA for {METRIC} with model, language, and injection position "
            f"as factors on TREC-DL {TREC_DL_YEAR}. Omega squared is reported as an "
            f"effect size estimate."
        ),
        label=f"tab:anova_model_language_position_{safe_slug(METRIC)}_{TREC_DL_YEAR}",
    )
    OUT_ANOVA_TEX.write_text(latex, encoding="utf-8")

    print("\n[ANOVA RESULT]")
    print(anova_df.to_string(index=False))

    pos_row = anova_df[anova_df["term"] == "C(position)"]
    if not pos_row.empty:
        pval = pos_row["PR(>F)"].iloc[0]
        omega = pos_row["omega_sq"].iloc[0]
        print(f"\n[POSITION EFFECT] p-value = {pval:.6g}, omega_sq = {omega:.6g}")

    print(f"\n[OK] Wrote samples: {OUT_SAMPLES}")
    print(f"[OK] Wrote ANOVA CSV: {OUT_ANOVA_CSV}")
    print(f"[OK] Wrote ANOVA TeX: {OUT_ANOVA_TEX}")

    if skipped:
        print(f"[INFO] Skipped {skipped} files due to errors.")


if __name__ == "__main__":
    main()