import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt

# ===== choose settings here =====
MODEL = "gpt-oss-20b"
PROFILE = "cwb_instruct"
YEAR = "2021"

# ===== import lang_profiles =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HELPERS_DIR = os.path.join(CURRENT_DIR, "helpers")
sys.path.append(HELPERS_DIR)

from lang_profiles import get_langs


def extract_model(filename):
    match = re.search(r'^(.*?)_trecdl_', filename)
    return match.group(1) if match else None


def extract_year(filename):
    match = re.search(r'trecdl_(202\d)_', filename)
    return match.group(1) if match else None


def extract_lang(filename):
    match = re.search(r'trecdl_202\d_(.*?)_labels\.csv', filename)
    return match.group(1) if match else None


def add_segment_labels(ax, bars, values, min_width=3):
    for bar, value in zip(bars, values):
        width = bar.get_width()
        if width >= min_width:
            x = bar.get_x() + width / 2
            y = bar.get_y() + bar.get_height() / 2
            ax.text(
                x, y, f"{value:.1f}",
                ha="center", va="center", fontsize=8
            )


def main():
    csv_path = r'd:\Work\Research_Project\anaconda_research_project\outputs\sanitation_checker\sanitation_scores_totals.csv'

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    df["model"] = df["filename"].apply(extract_model)
    df["year"] = df["filename"].apply(extract_year)
    df["lang"] = df["filename"].apply(extract_lang)

    # Filter
    df = df[df["model"] == MODEL].copy()
    df = df[df["year"] == YEAR].copy()

    valid_langs = get_langs(PROFILE)
    df = df[df["lang"].isin(valid_langs)].copy()

    # Optional: for cwb_instruct, keep only attacked forms
    if PROFILE == "cwb_instruct":
        df = df[df["lang"].str.endswith("cwb_instruct")].copy()

    if df.empty:
        print(f"No rows found for model='{MODEL}', profile='{PROFILE}', year='{YEAR}'")
        return

    # Calculate invalid
    df["total_invalid"] = df["total_rows"] - df["total_yes"] - df["total_no"]

    # Percentages
    df["yes_pct"] = df["total_yes"] / df["total_rows"] * 100
    df["no_pct"] = df["total_no"] / df["total_rows"] * 100
    df["invalid_pct"] = df["total_invalid"] / df["total_rows"] * 100

    df["label"] = df["lang"]
    df = df.sort_values("label").reset_index(drop=True)

    # FIX: always start with a completely fresh figure
    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.8)))

    bars_yes = ax.barh(df["label"], df["yes_pct"], label="Yes")
    bars_no = ax.barh(df["label"], df["no_pct"], left=df["yes_pct"], label="No")
    bars_invalid = ax.barh(
        df["label"],
        df["invalid_pct"],
        left=df["yes_pct"] + df["no_pct"],
        label="Invalid"
    )

    # Label only current bars
    add_segment_labels(ax, bars_yes, df["yes_pct"], min_width=3)
    add_segment_labels(ax, bars_no, df["no_pct"], min_width=3)
    add_segment_labels(ax, bars_invalid, df["invalid_pct"], min_width=3)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of Total Rows")
    ax.set_ylabel("Language")
    ax.set_title(f"Sanitation Response Percentages - {MODEL} - {PROFILE} - {YEAR}")
    ax.legend()

    # Optional: cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    output_file = os.path.join(
        os.path.dirname(csv_path),
        f"sanitation_percentages_{MODEL}_{PROFILE}_{YEAR}.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved to: {output_file}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()