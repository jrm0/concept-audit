"""
venues_config.py — Shared Stratum 1 (Core GPC) journal list for the full-text pulls.

DRAFT list from Conceptual_Coherence_Corpus_Protocol_v1.md §3 — pending Jonathan's
sign-off. Edit here once; 05 (S2ORC) and 06 (PMC) both import from this file.

Each entry: canonical name -> {"s2_names": [...variant venue strings in Semantic
Scholar...], "pmc_names": [...journal name/abbrev as indexed in PubMed...]}.
Venue strings in Semantic Scholar are inconsistent; add variants as you discover
them in pull logs rather than editing the scripts.
"""

YEAR_START = 1980
YEAR_END = 2025

STRATUM1_JOURNALS = {
    # --- RP:P sampling frame (must-include) ---
    "Psychological Science": {
        "s2_names": ["Psychological Science"],
        "pmc_names": ["Psychol Sci", "Psychological Science"],
    },
    "Journal of Personality and Social Psychology": {
        "s2_names": ["Journal of Personality and Social Psychology"],
        "pmc_names": ["J Pers Soc Psychol"],
    },
    "Journal of Experimental Psychology: Learning, Memory, and Cognition": {
        "s2_names": ["Journal of Experimental Psychology: Learning, Memory and Cognition",
                     "Journal of Experimental Psychology: Learning, Memory, and Cognition"],
        "pmc_names": ["J Exp Psychol Learn Mem Cogn"],
    },
    # --- Breadth additions (DRAFT) ---
    "Journal of Experimental Psychology: General": {
        "s2_names": ["Journal of Experimental Psychology: General"],
        "pmc_names": ["J Exp Psychol Gen"],
    },
    "Psychological Bulletin": {
        "s2_names": ["Psychological Bulletin"],
        "pmc_names": ["Psychol Bull"],
    },
    "Psychological Review": {
        "s2_names": ["Psychological Review"],
        "pmc_names": ["Psychol Rev"],
    },
    "Cognition": {
        "s2_names": ["Cognition"],
        "pmc_names": ["Cognition"],
    },
    "Journal of Experimental Social Psychology": {
        "s2_names": ["Journal of Experimental Social Psychology"],
        "pmc_names": ["J Exp Soc Psychol"],
    },
    "Personality and Social Psychology Bulletin": {
        "s2_names": ["Personality and Social Psychology Bulletin"],
        "pmc_names": ["Pers Soc Psychol Bull"],
    },
    "Developmental Psychology": {
        "s2_names": ["Developmental Psychology"],
        "pmc_names": ["Dev Psychol"],
    },
    "Journal of Abnormal Psychology": {
        # Renamed Journal of Psychopathology and Clinical Science in 2022
        "s2_names": ["Journal of Abnormal Psychology",
                     "Journal of Psychopathology and Clinical Science"],
        "pmc_names": ["J Abnorm Psychol", "J Psychopathol Clin Sci"],
    },
    "Cognitive Psychology": {
        "s2_names": ["Cognitive Psychology"],
        "pmc_names": ["Cogn Psychol"],
    },
    "Memory & Cognition": {
        "s2_names": ["Memory & Cognition", "Memory and Cognition"],
        "pmc_names": ["Mem Cognit"],
    },
    "Psychonomic Bulletin & Review": {
        "s2_names": ["Psychonomic Bulletin & Review", "Psychonomic bulletin & review"],
        "pmc_names": ["Psychon Bull Rev"],
    },
    "Emotion": {
        "s2_names": ["Emotion"],
        "pmc_names": ["Emotion"],
    },
    "Health Psychology": {
        "s2_names": ["Health Psychology"],
        "pmc_names": ["Health Psychol"],
    },
    "Clinical Psychological Science": {
        "s2_names": ["Clinical Psychological Science"],
        "pmc_names": ["Clin Psychol Sci"],
    },
    "Social Psychological and Personality Science": {
        "s2_names": ["Social Psychological and Personality Science"],
        "pmc_names": ["Soc Psychol Personal Sci"],
    },
    "Perspectives on Psychological Science": {
        "s2_names": ["Perspectives on Psychological Science"],
        "pmc_names": ["Perspect Psychol Sci"],
    },
    "British Journal of Psychology": {
        "s2_names": ["British Journal of Psychology"],
        "pmc_names": ["Br J Psychol"],
    },
}


def all_s2_names():
    return [v for j in STRATUM1_JOURNALS.values() for v in j["s2_names"]]


def all_pmc_names():
    return [v for j in STRATUM1_JOURNALS.values() for v in j["pmc_names"]]
