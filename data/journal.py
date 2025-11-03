#!/usr/bin/python3
"""
Search the NLM API for medicine-related journals.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2025.
"""
import json
import logging
import os
import pandas as pd
import time
from thefuzz import process  # type: ignore
from pathlib import Path
from typing import (
    Any, Dict, Final, Iterator, List, NamedTuple, Optional, Union
)

from .entrez import search_entrez, fetch_entrez, save_nlm_query


logger = logging.getLogger(__name__)


JOURNAL_SAVEPATH: Final[str] = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "raw", "journals.json"
)


class Journal(NamedTuple):
    NlmUniqueID: str
    Title: str
    MedlineTA: str
    PublicationFirstYear: int
    PublicationEndYear: int
    Subject: str
    SCIMagoMatch: Optional[str]
    SCIMagoScore: Optional[int]
    SJR: Optional[float]
    HIndex: Optional[int]
    IsOpenAccess: Optional[bool]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Journal):
            return False
        return self.NlmUniqueID == other.NlmUniqueID

    def __hash__(self) -> int:
        return hash(self.NlmUniqueID)


def get_journals(
    parent_subject: str,
    broad_subjects: List[str],
    journal_ranking: pd.DataFrame,
    language: Optional[str] = "English",
    time_sleep: Optional[float] = 1.0,
    retmax: Optional[int] = 100000
) -> List[NamedTuple]:
    """
    Retrieve journals in the NLM catalog that are relevant to the input query.
    Input:
        parent_subject: the parent subject term query.
        broad_subjects: the subject term query strings.
        journal_ranking: the journal ranking data from Scimago. See
            https://www.scimagojr.com/journalrank.php for additional details.
        language: an optional language filter. Default English.
        time_sleep: an optional amount of seconds to sleep between API calls.
        retmax: the maximum number of search results. Default 100000.
    Returns:
        A list of journals related to the subject term.
    """
    db: Final[str] = "nlmcatalog"
    retmode: Final[str] = "xml"
    titles: Final[List[str]] = journal_ranking["Title"].map(str.lower).tolist()

    for subject_term in broad_subjects:
        query = f"{subject_term}[st]"
        if language is not None:
            query += f" AND {language.lower()}[la]"
        root = search_entrez(db, query, retmode=retmode, retmax=retmax)
        if root is None:
            logger.warning(
                f"Journal retrieval failed for broad subject {subject_term}"
            )
            continue

        ids = sum(
            [
                [str(item.text) for item in lst.findall("Id")]
                for lst in root.findall("IdList")
            ],
            []
        )

        def id_chunks(n: int = 64) -> Iterator[List[str]]:
            for i in range(0, len(ids), n):
                yield ids[i:(i + n)]

        results = []
        for chunk_ids in id_chunks():
            root = fetch_entrez(db, chunk_ids, retmode=retmode)
            if root is None:
                continue
            for record in root.findall("NLMCatalogRecord"):
                if (nlm_id := record.find("NlmUniqueID")) is None:
                    continue
                elif (title := record.find("TitleMain")) is None:
                    continue
                elif (title := title.find("Title")) is None:
                    continue
                elif (medline_ta := record.find("MedlineTA")) is None:
                    continue

                if (info := record.find("PublicationInfo")) is None:
                    continue
                elif (pub_year := info.find("PublicationFirstYear")) is None:
                    continue
                elif not str(pub_year.text).isdigit():
                    continue
                start = int(str(pub_year.text))

                end_year = info.find("PublicationEndYear")
                if end_year is None or not str(end_year.text).isdigit():
                    end = 9999
                else:
                    end = int(str(end_year.text))

                match, score, sjr, h_index = None, None, None, None
                is_open_access = None
                if end >= 2024:
                    search_key = str(title.text).lower().split(":")[0]
                    search_key = search_key.split("=")[0].replace("&", "and")
                    search_key = "".join(
                        filter(lambda c: c.isalpha() or c in " ,-", search_key)
                    )
                    search_key = " ".join(search_key.split()).strip()
                    if ". supplement" in str(title.text).lower():
                        search_key = search_key.replace(
                            " supplement", ". supplement"
                        )
                    match, score = process.extractOne(search_key, titles)
                    row = journal_ranking.iloc[titles.index(match)]
                    sjr = float(str(row["SJR"]).replace(",", ".", 1))
                    h_index = int(row["H index"])
                    is_open_access = bool(
                        str(row["Open Access"]).lower() == "yes"
                    )
                results.append(
                    Journal(
                        NlmUniqueID=str(nlm_id.text),
                        Title=str(title.text),
                        MedlineTA=str(medline_ta.text),
                        PublicationFirstYear=start,
                        PublicationEndYear=end,
                        Subject=parent_subject,
                        SCIMagoMatch=match,
                        SCIMagoScore=score,
                        SJR=sjr,
                        HIndex=h_index,
                        IsOpenAccess=is_open_access
                    )
                )

            if time_sleep:
                time.sleep(time_sleep)

    return sorted(list(set(results)))


def main(
    save_fn: Union[Path, str],
    journal_ranking_url: str = (
        "https://www.scimagojr.com/journalrank.php?out=xls&year=2024"
    ),
    time_sleep: Optional[float] = 1.0
) -> int:
    """
    Find medicine journals in the NLM Catalog.
    Input:
        save_fn: the local JSON path to save the search results to.
        journal_ranking_url: the URL of the CSV file with the journal ranking
            data from Scimago. See https://www.scimagojr.com/journalrank.php
            for additional details.
        time_sleep: an optional amount of seconds to sleep between API calls.
    Returns:
        Exit code.
    """
    assert str(save_fn).lower().endswith(".json")

    with open("raw/medicine_broad_subjects.json") as f:
        search_terms: Dict[str, List[str]] = {
            key: list(val.keys()) for key, val in json.load(f)["data"].items()
        }

    journal_rankings = pd.read_csv(journal_ranking_url, sep=";")

    results: Dict[str, List[NamedTuple]] = {}
    for term, broad_subjects in search_terms.items():
        results[term] = get_journals(
            term, broad_subjects, journal_rankings, time_sleep=time_sleep
        )

    save_nlm_query(save_fn, results, search_terms)

    return 0
