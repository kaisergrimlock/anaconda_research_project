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


# =========================
#   Krippendorff's alpha
# =========================

def _krippendorff_alpha(
    ratings: np.ndarray,
    level: str = "ordinal",
) -> float:
    """
    Krippendorff's alpha for 2+ raters, any number of units.
    ratings: 2D array of shape (n_raters, n_units), with np.nan for missing.
    level: "nominal" or "ordinal".
    """
    arr = np.asarray(ratings, dtype=float)

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    # Collect all valid values to determine categories
    valid = ~np.isnan(arr)
    if not valid.any():
        return float("nan")

    values = np.unique(arr[valid])
    k = len(values)
    if k <= 1:
        # All ratings identical – perfect agreement, but alpha formula is degenerate.
        return 1.0

    # Map category value -> index 0..k-1 (sorted by value)
    value_to_idx = {v: i for i, v in enumerate(values)}

    # Coincidence matrix O (k x k)
    O = np.zeros((k, k), dtype=float)

    n_raters, n_units = arr.shape

    # Build coincidence matrix per Krippendorff:
    # For each unit, count how often each category was used, then update O.
    for u in range(n_units):
        col = arr[:, u]
        mask = ~np.isnan(col)
        if mask.sum() <= 1:
            # Need at least two valid ratings to contribute
            continue
        # indices of categories for this unit
        idxs = [value_to_idx[v] for v in col[mask]]
        counts = np.bincount(idxs, minlength=k)

        # Add to coincidence matrix
        # Diagonal: n_c * (n_c - 1)
        # Off-diagonal: n_c * n_c'
        for i in range(k):
            if counts[i] == 0:
                continue
            O[i, i] += counts[i] * (counts[i] - 1)
            for j in range(i + 1, k):
                if counts[j] == 0:
                    continue
                increment = counts[i] * counts[j]
                O[i, j] += increment
                O[j, i] += increment

    N = O.sum()
    if N == 0:
        return float("nan")

    # Distance matrix D
    D = np.zeros((k, k), dtype=float)
    if level == "nominal":
        for i in range(k):
            for j in range(k):
                D[i, j] = 0.0 if i == j else 1.0
    elif level == "ordinal":
        # Use positions (0..k-1) as the ordinal ranks of the sorted categories
        positions = np.arange(k, dtype=float)
        max_dist = (k - 1) ** 2
        if max_dist == 0:
            # Only one category – already handled above, but just in case
            return 1.0
        for i in range(k):
            for j in range(k):
                D[i, j] = ((positions[i] - positions[j]) ** 2) / max_dist
    else:
        raise ValueError(f"Unknown level='{level}', expected 'nominal' or 'ordinal'.")

    # Observed disagreement
    Do = float((O * D).sum() / N)

    # Expected disagreement
    row_marg = O.sum(axis=1)  # n_c
    De = float(
        (
            (row_marg[:, None] * row_marg[None, :] * D).sum()
        ) / (N - 1.0)
    )
    if De == 0:
        # No expected disagreement (degenerate); define as perfect agreement
        return 1.0

    return 1.0 - Do / De


def compute_krippendorff_alpha_paired(
    s_true: pd.Series,
    s_pred: pd.Series,
    level: str = "ordinal",
) -> float:
    """
    Convenience wrapper for two-coder data:
    s_true: Series of NIST labels
    s_pred: Series of LLM labels
    level: 'ordinal' or 'nominal'
    """
    a_true = pd.to_numeric(s_true, errors="coerce").to_numpy()
    a_pred = pd.to_numeric(s_pred, errors="coerce").to_numpy()
    ratings = np.vstack([a_true, a_pred])
    return _krippendorff_alpha(ratings, level=level)


# =========================
#   Full metric bundle
# =========================

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
      - Krippendorff's alpha (4pt ordinal + 2pt binary nominal)
    Optionally write to out_dir (caller can implement).
    """
    # MAE
    mae = compute_mae(paired["NIST"], paired["LLM"])

    # Kappa (4-point ordinal, from confusion matrix)
    kappa_weighted = compute_weighted_kappa_ordinal(cm_4pt)

    # Krippendorff's alpha (4-point, ordinal)
    alpha_4pt = compute_krippendorff_alpha_paired(
        paired["NIST"], paired["LLM"], level="ordinal"
    )

    # Binary
    nist_bin = binarize_labels(paired["NIST"])
    llm_bin = binarize_labels(paired["LLM"])

    cm_bin = pd.crosstab(
        index=pd.Categorical(nist_bin, categories=[0, 1], ordered=True),
        columns=pd.Categorical(llm_bin,  categories=[0, 1], ordered=True),
        dropna=False,
    )
    cm_bin.index.name = "NIST_bin"
    cm_bin.columns.name = "LLM_bin"

    # Binary kappa (unweighted / nominal)
    kappa_binary = compute_unweighted_kappa(cm_bin)

    # Binary Krippendorff's alpha (nominal)
    alpha_bin = compute_krippendorff_alpha_paired(
        nist_bin, llm_bin, level="nominal"
    )

    metrics = {
        "mae": mae,
        "kappa_weighted_4pt": kappa_weighted,
        "kappa_binary_2pt": kappa_binary,
        "alpha_4pt_ordinal": alpha_4pt,
        "alpha_binary_nominal": alpha_bin,
        "alpha_alpha_ordinal"
        "pairs": float(len(paired)),
    }

    if out_dir:
        # You can add writing logic here if needed
        pass

    return {
        "metrics": metrics,
        "cm_bin": cm_bin,
    }
