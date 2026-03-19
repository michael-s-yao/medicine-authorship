#!/usr/bin/python3
"""
Data loading functions.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2025.
"""
import json
from .entrez import search_entrez, fetch_entrez, save_nlm_query
from .journal import (
    Journal, get_journals, main as journal_main, JOURNAL_SAVEPATH
)
from .article import (
    Author,
    Award,
    Article,
    get_articles,
    main as article_main,
    ARTICLE_SAVEPATH
)
from .benchmark import (
    BenchmarkDataset,
    SSABenchmarkDataset,
    PinyinBenchmarkDataset,
    GlobalBenchmarkDataset
)


__all__ = [
    "search_entrez",
    "fetch_entrez",
    "save_nlm_query",
    "get_journals",
    "get_articles",
    "main",
    "Journal",
    "Author",
    "Award",
    "Article",
    "BenchmarkDataset",
    "SSABenchmarkDataset",
    "PinyinBenchmarkDataset",
    "GlobalBenchmarkDataset"
]


def main() -> int:
    """
    Construct the medicine article dataset.
    Input:
        None.
    Returns:
        Exit code.
    """
    journal_main(JOURNAL_SAVEPATH)

    with open(JOURNAL_SAVEPATH) as f:
        journals = {
            broad_subject: [Journal(**j) for j in journals["results"]]
            for broad_subject, journals in json.load(f).items()
        }

    article_main(ARTICLE_SAVEPATH, journals)

    return 0
