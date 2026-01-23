from __future__ import annotations

import matplotlib as mpl

paper_fmt = {
    "font.family": "serif",
    "font.serif": [
        "Linux Libertine",
        "Linux Libertine O",
        "Linux Biolinum",
        "Linux Biolinum O",
    ],
    "font.size": 9,
    "font.sans-serif": [
        "Linux Libertine",
        "Linux Libertine O",
        "Linux Biolinum O",
    ],
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.figsize": (4, 2.25),
    "legend.borderaxespad": 0.5,
    "legend.fontsize": "small",
    "legend.title_fontsize": "small",
    "axes.labelpad": 5.0,
    "axes.titlesize": "medium",
    "axes.labelsize": "medium",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_paper_fmt() -> None:
    mpl.rcParams.update(paper_fmt)
