from __future__ import annotations
from typing import Dict, List

LANG_PROFILES: Dict[str, List[str]] = {
    "brackets": [
        "raw", "eng", "ru", "vi", "th", "sw", "ga",
        "eng_brackets", "ru_brackets", "vi_brackets", "sw_brackets", "ga_brackets", "th_brackets",
    ],
    "last": [
        "raw", "eng", "ru", "ar", "vi", "th", "sw", "he", "zh", "fr", "hi", "ga",
        "eng_last", "fr_last", "ru_last", "ar_last", "he_last", "vi_last", "th_last", "sw_last", "ga_last", "zh_last", "he_last", "hi_last",
    ],
    "first": [
        "raw", "eng", "ru", "ar", "vi", "th", "sw", "he", "zh", "fr", "hi", "ga",
        "raw", "eng_first", "fr_first", "ru_first", "ar_first", "he_first", "vi_first", "th_first", "sw_first", "ga_first", "zh_first", "hi_first",
    ],
    "word": [
        "raw", "eng", "ru", "ar", "vi", "th", "sw", "he", "zh", "fr", "hi",
        "eng_word", "ru_word", "ar_word", "vi_word", "th_word", "sw_word", "he_word", "zh_word", "fr_word", "hi_word",
    ],
    "crit": [
        "raw", "eng", "ar", "fr", "ru", "zh", "vi", "hi", "he", "th", "sw", "ga",
        "raw_crit", "eng_crit", "ar_crit", "fr_crit", "ru_crit", "zh_crit", "vi_crit",
        "hi_crit", "he_crit", "th_crit", "sw_crit", "ga_crit",
    ],
    "crit_subset": ["vi", "vi_crit", "th", "th_crit", "sw", "sw_crit", "ga", "ga_crit", "zh", "zh_crit"],
    "lang": ["raw", "eng", "fr", "ru", "vi", "he", "ar", "th", "sw", "ga", "zh", "hi"],
    "mult": ["raw", "eng", "eng_mult_2", "eng_mult_3"],
    "script": ["eng", "fr", "ru", "uk", "zh", "ja", "ar", "ur"],
    "cwb": [
        "eng", "raw", "vi", "th", "fr", "ru", "he", "sw", "ga", "hi", "zh",
        "engcwb", "vicwb", "thcwb", "frcwb", "rucwb", "hecwb", "swcwb", "gacwb", "hicwb", "zhcwb",
    ],
    
    "cwb_instruct": [
        "eng", "raw", "vi", "ar" "th", "fr", "ru", "he", "sw", "ga", "hi", "zh",
        "engcwb_instruct", "arcwb_instruct", "vicwb_instruct", "thcwb_instruct", "frcwb_instruct", "rucwb_instruct",
        "hecwb_instruct", "swcwb_instruct", "gacwb_instruct", "hicwb_instruct", "zhcwb_instruct",
    ],
    
    "instruct": [
        "eng", "raw", "ar", "vi", "th", "fr", "ru", "he", "sw", "ga", "hi", "zh",
        "eng_instruct", "vi_instruct", "ar_instruct", "fr_instruct", "th_instruct", "ru_instruct", "he_instruct", "sw_instruct", "ga_instruct", "hi_instruct", "zh_instruct"
    ],
    
    "instruct_defended":[
        "eng", "eng_instruct_defended", "eng_instruct", "raw"
    ]

}

def get_langs(profile: str) -> List[str]:
    if profile not in LANG_PROFILES:
        valid = ", ".join(sorted(LANG_PROFILES.keys()))
        raise KeyError(f"Unknown LANG profile '{profile}'. Valid profiles are: {valid}")
    return LANG_PROFILES[profile]
