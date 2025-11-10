# metrics_llm_vs_nist.py
from __future__ import annotations

from typing import Sequence, Tuple, Dict, Any
import numpy as np
import pandas as pd


def compute_mae(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Mean Absolute Error on integer / ordinal labels."""
    s_true = pd.to_numeric(pd.Series(y_true), errors="coerce")
    s_pred = pd.to_numeric(pd.Series(y_pred), errors="coerce")
    mask = s_true.notna() & s_pred.notna()
    if not mask.any():
        return float("nan")
    return float((s_true[mask] - s_pred[mask]).abs().mean())


def compute_weighted_kappa_ordinal(cm: pd.DataFrame) -> float:
    """
    Quadratic-weighted Cohen's kappa for ordinal labels.
    Assumes cm is square and rows/cols in the same order (e.g. 0..3).
    """
    n = cm.to_numpy().sum()
    if n == 0:
        return float("nan")

    labels = list(cm.index)
    k = len(labels)

    # quadratic weight matrix
    W = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            W[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)

    O = cm.to_numpy() / n

    row_marg = cm.sum(axis=1).to_numpy() / n
    col_marg = cm.sum(axis=0).to_numpy() / n
    E = np.outer(row_marg, col_marg)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 1.0
    return 1.0 - num / den


def compute_unweighted_kappa(cm: pd.DataFrame) -> float:
    """
    Standard Cohen's kappa from a confusion matrix.
    Works for 2x2 (binary) or kxk.
    """
    n = cm.to_numpy().sum()
    if n == 0:
        return float("nan")

    po = np.trace(cm.to_numpy()) / n
    row_marg = cm.sum(axis=1).to_numpy()
    col_marg = cm.sum(axis=0).to_numpy()
    pe = (row_marg * col_marg).sum() / (n * n)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def binarize_labels(s: pd.Series, threshold: int = 2) -> pd.Series:
    """
    Map 0..(threshold-1) -> 0  and threshold..max -> 1
    Default: 0-1 -> 0, 2-3 -> 1
    """
    s = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    return (s >= threshold).astype(int)


def eval_all_metrics(
    paired: pd.DataFrame,
    cm_4pt: pd.DataFrame,
    out_dir: str | None = None,
) -> Dict[str, Any]:
    """
    Given:
      - paired: DF with columns ['NIST', 'LLM']
      - cm_4pt: 4x4 confusion matrix on 0..3
    compute:
      - mae
      - weighted kappa (4pt)
      - binary kappa (0-1 vs 2-3)
      - binary confusion matrix
    Optionally write to out_dir.
    """
    mae = compute_mae(paired["NIST"], paired["LLM"])
    kappa_weighted = compute_weighted_kappa_ordinal(cm_4pt)

    # binary
    nist_bin = binarize_labels(paired["NIST"])
    llm_bin = binarize_labels(paired["LLM"])

    cm_bin = pd.crosstab(
        index=pd.Categorical(nist_bin, categories=[0, 1], ordered=True),
        columns=pd.Categorical(llm_bin,  categories=[0, 1], ordered=True),
        dropna=False,
    )
    cm_bin.index.name = "NIST_bin"
    cm_bin.columns.name = "LLM_bin"

    kappa_binary = compute_unweighted_kappa(cm_bin)

    metrics = {
        "mae": mae,
        "kappa_weighted_4pt": kappa_weighted,
        "kappa_binary_2pt": kappa_binary,
        "pairs": float(len(paired)),
    }

    if out_dir:
        out_path = pd.Path(out_dir) if hasattr(pd, "Path") else None  # harmless iff missing
        # you can just let the caller handle writing
        pass

    return {
        "metrics": metrics,
        "cm_bin": cm_bin,
    }
