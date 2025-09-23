# seaborn scatter with custom legend: per-LLM markers, per-company colors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

rows = [
    {"llm": "anthropic.claude-3-haiku-20240307-v1:0", "kappa_quadratic": 0.24716640085061137, "cost": "$0.12"},
    {"llm": "mistral.mixtral-8x7b-instruct-v0:1",      "kappa_quadratic": 0.37597724699412294, "cost": "$0.06"},
    {"llm": "openai.gpt-oss-20b-1:0",                  "kappa_quadratic": 0.5789947293996658,  "cost": "$0.07"},
    {"llm": "openai.gpt-oss-120b-1:0",                 "kappa_quadratic": 0.529605758309728,   "cost": "$0.27"},
    {"llm": "anthropic.claude-3-5-haiku-20241022-v1:0","kappa_quadratic": 0.45082998883849057, "cost": "$0.89"},
    {"llm": "us.amazon.nova-lite-v1:0",                "kappa_quadratic": 0.40943553285788215, "cost": "$0.06"},
]

df = pd.DataFrame(rows)
df["cost_usd"] = df["cost"].str.replace("$", "", regex=False).astype(float)

def family(name: str) -> str:
    if name.startswith("openai."): return "OpenAI GPT-OSS"
    if name.startswith("anthropic."): return "Anthropic"
    if name.startswith("mistral."): return "Mistral"
    if name.startswith("us.amazon."): return "Amazon Nova"
    return "Other"

df["family"] = df["llm"].map(family)

# --- color by family (company), marker by LLM ---
families = df["family"].unique().tolist()
palette = sns.color_palette("deep", n_colors=len(families))
family_color = dict(zip(families, palette))

markers_cycle = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]
llms = df["llm"].tolist()
llm_marker = {llm: markers_cycle[i % len(markers_cycle)] for i, llm in enumerate(llms)}

sns.set_theme(style="whitegrid", font_scale=1.0)
fig, ax = plt.subplots(figsize=(8, 5.2))

# Background zones (Bad/Fair/Good by kappa thresholds)
ymin, ymax = 0.2, 0.9
xmin, xmax = df["cost_usd"].min()*0.85, df["cost_usd"].max()*1.15
ax.add_patch(Rectangle((xmin, 0.0), xmax-xmin, 0.4-0.0, alpha=0.12, color="red"))
ax.add_patch(Rectangle((xmin, 0.4), xmax-xmin, 0.6-0.4, alpha=0.10, color="gold"))
ax.add_patch(Rectangle((xmin, 0.6), xmax-xmin, 1.0-0.6, alpha=0.10, color="green"))

# Plot points manually so we can set color/marker per row
for _, r in df.iterrows():
    ax.scatter(
        r["cost_usd"], r["kappa_quadratic"],
        s=120,
        marker=llm_marker[r["llm"]],
        color=family_color[r["family"]],
        edgecolor="black", linewidths=0.4
    )

ax.set_xlabel("Cost per 1K tokens (USD)")
ax.set_ylabel("Quadratic Cohen’s κ")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_title("LLM Cost vs. Quadratic Cohen’s κ")

ax.text(xmin + (xmax-xmin)*0.02, 0.62, "Good", fontsize=10, color="green", fontstyle="italic")
ax.text(xmin + (xmax-xmin)*0.02, 0.50, "Fair", fontsize=10, color="darkgoldenrod", fontstyle="italic")
ax.text(xmin + (xmax-xmin)*0.02, 0.30, "Bad", fontsize=10, color="darkred", fontstyle="italic")

# ----- Custom legends -----
# Legend 1: LLM names (marker varies per LLM, color = family color)
llm_handles = [
    Line2D([0], [0],
           marker=llm_marker[name],
           linestyle="",
           markersize=8,
           markerfacecolor=family_color[df.loc[df["llm"] == name, "family"].iloc[0]],
           markeredgecolor="black",
           label=name)
    for name in llms
]
leg1 = ax.legend(handles=llm_handles, title="LLMs", loc="lower right", frameon=True)

# Legend 2: Company colors (same color for all models from the company)
family_handles = [
    Line2D([0], [0], marker="o", linestyle="", markersize=8,
           markerfacecolor=family_color[fam], markeredgecolor="black", label=fam)
    for fam in families
]
leg2 = ax.legend(handles=family_handles, title="Provider (color)", loc="upper left", frameon=True)

ax.add_artist(leg1)  # keep both legends

plt.tight_layout()
plt.show()
