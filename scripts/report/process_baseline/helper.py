# common/huge_csv.py
import csv, sys

DEFAULT_TARGET = 2_000_000_000  # try ~2e9; will back off if too large

def allow_huge_csv_fields(target: int = DEFAULT_TARGET) -> int:
    """
    Raise Python's csv field size limit high enough for giant cells (e.g., long passages).
    Safe on Windows/Mac/Linux: it backs off if the platform can't take the requested size.
    Returns the final limit that was set.
    """
    limit = min(int(target), int(getattr(sys, "maxsize", target)))
    while limit >= 131_072:
        try:
            csv.field_size_limit(limit)
            return csv.field_size_limit()
        except OverflowError:
            limit //= 2
    # Fallback to whatever the current limit is if we couldn't raise above the default cap
    return csv.field_size_limit()
