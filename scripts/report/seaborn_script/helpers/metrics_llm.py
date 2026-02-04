# metrics_llm_vs_nist.py
from __future__ import annotations

from typing import Sequence, Tuple, Dict, Any
import numpy as np
import pandas as pd
import krippendorff
from statsmodels.stats.inter_rater import cohens_kappa
from sklearn.metrics import mean_absolute_error



def compute_mae(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Mean Absolute Error on integer / ordinal labels."""
    s_true = pd.to_numeric(pd.Series(y_true), errors="coerce")
    s_pred = pd.to_numeric(pd.Series(y_pred), errors="coerce")
    mask = s_true.notna() & s_pred.notna()
    if not mask.any():
        return float("nan")

    return float(mean_absolute_error(s_true[mask], s_pred[mask]))



def compute_weighted_kappa_ordinal(cm: pd.DataFrame) -> float:
    """
    Quadratic-weighted Cohen's kappa for ordinal labels.
    Assumes cm is square and rows/cols in the same order (e.g. 0..3).
    """
    table = cm.to_numpy(dtype=float)
    if table.sum() == 0:
        return float("nan")
    
    k = table.shape[0]
    scores = np.arange(k)
    res = cohens_kappa(table, wt="quadratic", weights=scores, return_results=True)
    return float(res.kappa)


def compute_unweighted_kappa(cm: pd.DataFrame) -> float:
    table = cm.to_numpy(dtype=float) #Turn the confusion matrix into a numpy array
    if table.sum() == 0:
        return float("nan") #Handle edge case of all-zero confusion matrix
    # Compute unweighted Cohen's kappa
    res = cohens_kappa(table, wt=None, weights=None, return_results=True)
    return float(res.kappa)


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

def krippendorff_alpha(
    ratings: np.ndarray,
    *,
    level: str,
    value_domain: list[int] | None,
) -> float:
    """
    Core alpha computation: accepts ratings shaped (n_raters, n_units) with np.nan for missing,
    then calls krippendorff.alpha().
    """
    arr = np.asarray(ratings, dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    # If everything is missing, krippendorff may error / return nan depending on version
    if np.isnan(arr).all():
        return float("nan")

    # Optional: keep your "degenerate single-category => 1.0" behavior
    valid_vals = arr[~np.isnan(arr)]
    if valid_vals.size > 0 and np.unique(valid_vals).size <= 1:
        return 1.0

    # Lock domain for ordinal for stable comparisons
    if value_domain is None and level == "ordinal":
        value_domain = [0, 1, 2, 3]

    try:
        return float(
            krippendorff.alpha(
                reliability_data=arr,
                level_of_measurement=level,
                value_domain=value_domain,
            )
        )
    except Exception:
        return float("nan")

def compute_krippendorff_alpha_paired(
    s_true: pd.Series,
    s_pred: pd.Series,
    level: str = "ordinal",
    *,
    value_domain: list[int] | None = None,
) -> float:
    a_true = pd.to_numeric(s_true, errors="coerce").to_numpy(dtype=float)
    a_pred = pd.to_numeric(s_pred, errors="coerce").to_numpy(dtype=float)

    ratings = np.vstack([a_true, a_pred])  # shape (2, n_units)

    return krippendorff_alpha(
        ratings,
        level=level,
        value_domain=value_domain,
    )

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
        "pairs": float(len(paired)),
    }

    if out_dir:
        # You can add writing logic here if needed
        pass

    return {
        "metrics": metrics,
        "cm_bin": cm_bin,
    }
