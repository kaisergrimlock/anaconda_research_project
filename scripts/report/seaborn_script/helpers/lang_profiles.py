from __future__ import annotations
from typing import Dict, List

LANG_PROFILES: Dict[str, List[str]] = {
    "brackets": [
        "raw", "eng", "ru", "vi", "th", "sw", "ga",
        "eng_brackets", "ru_brackets", "vi_brackets", "sw_brackets", "ga_brackets", "th_brackets",
    ],
    "last": [
        "raw", "eng_last", "fr_last", "ru_last", "ar_last", "he_last", "vi_last", "th_last", "sw_last", "ga_last",
    ],
    "first": [
        "raw", "eng_first", "fr_first", "ru_first", "ar_first", "he_first", "vi_first", "th_first", "sw_first", "ga_first", "zh_first",
    ],
    "word": [
        "raw", "eng", "ru", "ar", "vi", "th", "sw",
        "eng_word", "ru_word", "ar_word", "vi_word", "th_word", "sw_word",
    ],
    "crit": [
        "raw", "raw_crit",
        "eng", "eng_crit",
        "ar", "ar_crit",
        "fr", "fr_crit",
        "ru", "ru_crit",
        "zh", "zh_crit",
        "vi", "vi_crit",
        "hi", "hi_crit",
        "he", "he_crit",
        "th", "th_crit",
        "sw", "sw_crit",
        "ga", "ga_crit",
    ],
    "crit_subset": ["vi", "vi_crit", "th", "th_crit", "sw", "sw_crit", "ga", "ga_crit", "zh", "zh_crit"],
    "lang": ["raw", "eng", "fr", "ru", "vi", "he", "ar", "th", "sw", "ga", "zh", "hi"],
    "corrected": ["vi", "vi_corrected", "th", "th_corrected", "he", "he_corrected", "ar", "ar_corrected"],
    "mult": ["eng", "eng_mult", "eng_mult_2", "eng_mult_3"],
    "low": ["eng", "th", "uk", "el", "ha", "sw", "ga", "hi", "raw"],
    "mid": ["eng", "eng_mid", "vi", "vi_mid"],
    "script": ["eng", "fr", "ru", "uk", "zh", "ja", "ar", "ur"],
    "cwb": ["eng", "engcwb", "raw", "vi", "vicwb", "th", "thcwb", "fr", "frcwb", "ru", "rucwb", "he", "hecwb", "sw", "swcwb", "ga", "gacwb", "hi", "hicwb", "zh", "zhcwb"]
}

def get_langs(profile: str) -> List[str]:
    if profile not in LANG_PROFILES:
        valid = ", ".join(sorted(LANG_PROFILES.keys()))
        raise KeyError(f"Unknown LANG profile '{profile}'. Valid profiles are: {valid}")
    return LANG_PROFILES[profile]
