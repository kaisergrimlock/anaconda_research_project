#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.stats.multicomp import pairwise_tukeyhsd


# =========================
# Config (demo)
# =========================
OUT_DIR = Path("figures") / "demo" / "tukey_hsd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TUKEY_CSV = OUT_DIR / "tukey_hsd_table.csv"
OUT_TUKEY_TEX = OUT_DIR / "tukey_hsd_table.tex"
OUT_SIMUL_SVG = OUT_DIR / "tukey_hsd_plot_simultaneous.svg"

ALPHA = 0.05


def build_demo_df(seed: int = 42) -> pd.DataFrame:
    """
    Demo dataset:
      - group: your condition (e.g., raw/eng/vi/ru, or model, or prompt variant)
      - value: numeric metric you compare (e.g., MAE, FP rate, kappa, etc.)
    """
    rng = np.random.default_rng(seed)
    groups = ["raw", "eng", "vi", "ru"]

    means = {"raw": 0.80, "eng": 0.78, "vi": 0.86, "ru": 0.83}
    std = 0.03
    n = 60

    rows = []
    for g in groups:
        vals = rng.normal(means[g], std, size=n)
        rows.extend([{"group": g, "value": float(v)} for v in vals])

    return pd.DataFrame(rows)


def tukey_to_df(tukey) -> pd.DataFrame:
    """
    Convert TukeyHSDResults.summary() into a clean DataFrame.
    Columns typically: group1, group2, meandiff, p-adj, lower, upper, reject
    """
    table = tukey.summary().data
    header = table[0]
    body = table[1:]
    df = pd.DataFrame(body, columns=header)

    # coerce numeric fields
    for c in ["meandiff", "p-adj", "lower", "upper"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "reject" in df.columns:
        df["reject"] = df["reject"].astype(str).str.lower().map({"true": True, "false": False})

    return df


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    fmt = df.copy()
    for c in ["meandiff", "p-adj", "lower", "upper"]:
        if c in fmt.columns:
            fmt[c] = fmt[c].map(lambda x: f"{x:.6g}" if pd.notnull(x) else "")

    return fmt.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format="l l r r r r l",
    )


def main():
    # 1) Build demo data (swap for your real data later)
    df = build_demo_df()

    # 2) Tukey HSD
    tukey = pairwise_tukeyhsd(endog=df["value"], groups=df["group"], alpha=ALPHA)

    # 3) Table outputs
    tukey_df = tukey_to_df(tukey)
    write_df(tukey_df, OUT_TUKEY_CSV)

    latex = to_latex_table(
        tukey_df,
        caption=f"Tukey HSD pairwise comparisons (FWER={ALPHA}).",
        label="tab:tukey_hsd_demo",
    )
    write_text(latex, OUT_TUKEY_TEX)

    print(tukey.summary())
    print(f"\nWrote CSV:   {OUT_TUKEY_CSV}")
    print(f"Wrote LaTeX: {OUT_TUKEY_TEX}")

    # 4) plot_simultaneous
    tukey.plot_simultaneous()
    plt.tight_layout()
    plt.savefig(OUT_SIMUL_SVG, format="svg")
    plt.close()
    print(f"Wrote plot:  {OUT_SIMUL_SVG}")


if __name__ == "__main__":
    main()
