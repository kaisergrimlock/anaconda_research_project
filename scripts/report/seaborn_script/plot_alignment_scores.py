import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt

# ===== choose settings here =====
MODEL = "gpt-oss-20b"
PROFILE = "default"
YEAR = "2022"

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


def add_bar_labels(ax, bars, values, fmt="{:.2f}", min_value=0):
    x_max = ax.get_xlim()[1]
    for bar, value in zip(bars, values):
        if value > min_value:
            x = bar.get_width() + 0.01 * x_max
            y = bar.get_y() + bar.get_height() / 2
            ax.text(x, y, fmt.format(value), va="center", fontsize=8)


def main():
    csv_path = r'd:\Work\Research_Project\anaconda_research_project\outputs\alignment_checker\alignment_scores_totals.csv'

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    df["model"] = df["filename"].apply(extract_model)
    df["year"] = df["filename"].apply(extract_year)
    df["lang"] = df["filename"].apply(extract_lang)

    # Filter by model and year
    df = df[df["model"] == MODEL].copy()
    df = df[df["year"] == YEAR].copy()

    # Filter by profile from lang_profiles
    valid_langs = get_langs(PROFILE)
    df = df[df["lang"].isin(valid_langs)].copy()

    if df.empty:
        print(f"No rows found for model='{MODEL}', profile='{PROFILE}', year='{YEAR}'")
        return

    # Calculate average alignment score per valid scored row
    df["avg_alignment_score"] = df["total_alignment_score"] / df["valid_score_rows"]

    # Optional: score coverage percentage
    df["coverage_pct"] = df["valid_score_rows"] / df["total_rows"] * 100

    df["label"] = df["lang"]
    df = df.sort_values("label").reset_index(drop=True)

    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.8)))

    bars = ax.barh(df["label"], df["avg_alignment_score"])

    ax.set_xlabel("Average Alignment Score")
    ax.set_ylabel("Language")
    ax.set_title(f"Average Alignment Score - {MODEL} - {PROFILE} - {YEAR}")

    add_bar_labels(ax, bars, df["avg_alignment_score"], fmt="{:.2f}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    output_file = os.path.join(
        os.path.dirname(csv_path),
        f"alignment_avg_scores_{MODEL}_{PROFILE}_{YEAR}.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved to: {output_file}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()