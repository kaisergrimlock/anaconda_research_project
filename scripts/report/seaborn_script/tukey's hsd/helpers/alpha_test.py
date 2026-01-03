#!/usr/bin/env python3
import numpy as np
import pandas as pd

from metrics_llm import compute_krippendorff_alpha_paired

try:
    import krippendorff
    HAS_REF = True
except ImportError:
    HAS_REF = False
    print("⚠️  pip install krippendorff to enable reference comparisons")


def alpha(true, pred, level="ordinal"):
    return compute_krippendorff_alpha_paired(
        pd.Series(true), pd.Series(pred), level=level
    )


def section(name: str):
    print("\n" + "=" * 60)
    print(">>>", name)
    print("=" * 60)


# -------------------------------------------------------
# 1️⃣ PERFECT AGREEMENT TESTS
# -------------------------------------------------------
section("PERFECT AGREEMENT")
true = [0, 1, 2, 3]
pred = [0, 1, 2, 3]
print("Ordinal:", alpha(true, pred, "ordinal"))
print("Nominal:", alpha(true, pred, "nominal"))

# -------------------------------------------------------
# 2️⃣ TOTAL DISAGREEMENT (opposite ratings)
# -------------------------------------------------------
section("TOTAL DISAGREEMENT")
true = [0, 0, 1, 1, 2, 3]
pred = [3, 3, 2, 2, 1, 0]
print("Ordinal:", alpha(true, pred, "ordinal"))
print("Nominal:", alpha(true, pred, "nominal"))

# -------------------------------------------------------
# 3️⃣ SMALL MANUAL CASE – good to inspect coincidence matrix logic
# -------------------------------------------------------
section("SMALL SAMPLE CASE")
true = [0, 0, 1, 1]
pred = [1, 1, 0, 0]
print("Ordinal:", alpha(true, pred, "ordinal"))
print("Nominal:", alpha(true, pred, "nominal"))

# -------------------------------------------------------
# 4️⃣ RANDOM SANITY TEST WITH REFERENCE IMPLEMENTATION
# -------------------------------------------------------
if HAS_REF:
    section("COMPARE AGAINST REFERENCE IMPLEMENTATION (krippendorff)")
    rng = np.random.default_rng(0)
    for i in range(5):
        data = rng.integers(0, 4, size=(2, 25)).astype(float)
        a_ref = krippendorff.alpha(reliability_data=data, level_of_measurement='ordinal')
        a_me  = alpha(data[0], data[1], level="ordinal")
        print(f"Test {i}: ref={a_ref:.6f}, mine={a_me:.6f}, diff={abs(a_ref-a_me):.8f}")

print("\nDone.\n")
