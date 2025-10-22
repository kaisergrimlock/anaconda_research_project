# seaborn scatter with custom legend: per-LLM markers, per-company colors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe

# =========================
# Data (example rows)
# =========================
rows = [
    {"llm": "anthropic.claude-3-haiku-20240307-v1:0",  "kappa_quadratic": 0.17635588056063378, "cost": "$0.12"},
    {"llm": "mistral.mixtral-8x7b-instruct-v0:1",       "kappa_quadratic": 0.26067415730337073, "cost": "$0.06"},
    {"llm": "openai.gpt-oss-20b-1:0",                   "kappa_quadratic": 0.41959476788920236, "cost": "$0.07"},
    {"llm": "openai.gpt-oss-120b-1:0",                  "kappa_quadratic": 0.4384688389750079,  "cost": "$0.27"},
    {"llm": "anthropic.claude-3-5-haiku-20241022-v1:0", "kappa_quadratic": 0.3480502152950452,  "cost": "$0.89"},
    {"llm": "us.amazon.nova-lite-v1:0",                 "kappa_quadratic": 0.2701173894531962,  "cost": "$0.06"},
]
df = pd.DataFrame(rows)
df["cost_usd"] = df["cost"].str.replace("$", "", regex=False).astype(float)

# =========================
# Provider family mapping
# =========================
def family(name: str) -> str:
    if name.startswith("openai."):     return "OpenAI GPT-OSS"
    if name.startswith("anthropic."):  return "Anthropic"
    if name.startswith("mistral."):    return "Mistral"
    if name.startswith("us.amazon."):  return "Amazon Nova"
    return "Other"
df["family"] = df["llm"].map(family)

# =========================
# Plot config (edit freely)
# =========================
# Axis limits (y fixed, x padded from data)
Y_MIN, Y_MAX = 0.20, 0.90
X_PAD_LEFT, X_PAD_RIGHT = 0.85, 1.15  # multiply min/max by these for padding

# Thresholds + labels + styling
# Demassie et al.
LOWER_THRESH = 0.391
UPPER_THRESH = 0.599
LOWER_LABEL  = "Demassie et al. crowd-source lower-bound"
UPPER_LABEL  = "Demassie et al. crowd-source upper-bound"
BAND_COLORS  = {"low": "red", "mid": "gold", "high": "green"}   # fill colors
BAND_ALPHAS  = {"low": 0.12, "mid": 0.10, "high": 0.10}         # opacities
SHOW_DASHED_LINES = True

# Style
sns.set_theme(style="whitegrid", font_scale=1.0)

# =========================
# Color by family, marker by LLM
# =========================
families = df["family"].unique().tolist()
palette = sns.color_palette("deep", n_colors=len(families))
family_color = dict(zip(families, palette))

markers_cycle = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]
llms = df["llm"].tolist()
llm_marker = {llm: markers_cycle[i % len(markers_cycle)] for i, llm in enumerate(llms)}

# =========================
# Figure
# =========================
fig, ax = plt.subplots(figsize=(8, 5.2))

# x-limits with padding derived from data
xmin_data, xmax_data = df["cost_usd"].min(), df["cost_usd"].max()
xmin = xmin_data * X_PAD_LEFT
xmax = xmax_data * X_PAD_RIGHT
ymin, ymax = Y_MIN, Y_MAX
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# =========================
# Background bands (keep fills, but no band labels)
# =========================
def _add_band(y0, y1, key):
    if y1 > y0:
        ax.add_patch(Rectangle(
            (xmin, y0), xmax - xmin, y1 - y0,
            alpha=BAND_ALPHAS[key], color=BAND_COLORS[key]
        ))

# Clamp to axis limits
low_bot,  low_top  = ymin, min(LOWER_THRESH, ymax)
mid_bot,  mid_top  = max(LOWER_THRESH, ymin), min(UPPER_THRESH, ymax)
high_bot, high_top = max(UPPER_THRESH, ymin), ymax

_add_band(low_bot,  low_top,  "low")
_add_band(mid_bot,  mid_top,  "mid")
_add_band(high_bot, high_top, "high")

# =========================
# Threshold lines + labels ON the lines
# =========================
def _label_line(y, text, pos="right", color="gray", pad_frac=0.02):
    """Write a label on the threshold line."""
    if not (ymin < y < ymax):
        return
    x = xmax - (xmax - xmin) * pad_frac if pos == "right" else xmin + (xmax - xmin) * pad_frac
    ha = "right" if pos == "right" else "left"
    ax.text(
        x, y, text,
        fontsize=10, color=color, va="bottom", ha=ha,
        path_effects=[pe.withStroke(linewidth=3, foreground="white")]
    )

if SHOW_DASHED_LINES:
    if ymin < LOWER_THRESH < ymax:
        ax.axhline(LOWER_THRESH, linestyle="--", linewidth=0.8, color="gray")
        _label_line(LOWER_THRESH, f"{LOWER_LABEL} ({LOWER_THRESH:.3f})", pos="right")
    if ymin < UPPER_THRESH < ymax:
        ax.axhline(UPPER_THRESH, linestyle="--", linewidth=0.8, color="gray")
        _label_line(UPPER_THRESH, f"{UPPER_LABEL} ({UPPER_THRESH:.3f})", pos="right")

# =========================
# Points (manual to set both color & marker)
# =========================
for _, r in df.iterrows():
    ax.scatter(
        r["cost_usd"], r["kappa_quadratic"],
        s=120,
        marker=llm_marker[r["llm"]],
        color=family_color[r["family"]],
        edgecolor="black", linewidths=0.4
    )

# =========================
# Labels & title
# =========================
ax.set_xlabel("Cost per 1K tokens (USD)")
ax.set_ylabel("Binarized Cohen’s κ")
ax.set_title("LLM Cost vs. Binarized Cohen’s κ")

# =========================
# Custom legend (provider colors)
# =========================
family_handles = [
    Line2D([0], [0], marker="o", linestyle="", markersize=8,
           markerfacecolor=family_color[fam], markeredgecolor="black", label=fam)
    for fam in families
]
leg2 = ax.legend(handles=family_handles, title="Provider (color)", loc="upper right", frameon=True)

plt.tight_layout()
plt.show()
