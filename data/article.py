#!/usr/bin/python3
"""
Search the NLM API for medicine-related articles.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2025.
"""
import logging
import os
import time
from pathlib import Path
from statistics import mode
from typing import Any, Dict, Final, List, NamedTuple, Optional, Tuple, Union

from .entrez import search_entrez, fetch_entrez, save_nlm_query
from .journal import Journal, JOURNAL_SAVEPATH


logger = logging.getLogger(__name__)


ARTICLE_SAVEPATH: Final[str] = os.path.join(
    os.path.dirname(JOURNAL_SAVEPATH), "articles.json"
)


class Author(NamedTuple):
    first: str
    last: str
    orcid: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.first} {self.last}"

    def __repr__(self) -> str:
        return f"{str(self)} (ORCID ID: {self.orcid})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Author):
            return str(other) == str(self)
        elif self.orcid is not None and other.orcid is not None:
            return self.orcid == other.orcid
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(repr(self))


class Award(NamedTuple):
    source: str
    ids: Tuple[str, ...]

    def __str__(self) -> str:
        val = self.source
        if len(self.ids):
            val += f" {' | '.join(sorted(self.ids))}"
        return val

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Award):
            return any(str(other) in x for x in self.ids)
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(repr(self))


class Article(NamedTuple):
    title: str
    authors: Tuple[Author, ...]
    funding: Tuple[Award, ...]
    journal: Journal
    year: int
    doi: Optional[str]


def get_articles(
    journal: Journal,
    retmax: Optional[int] = 100000,
    start_year: Optional[int] = 2015,
    end_year: Optional[int] = 2025,
    batch_size: int = 32,
    max_retries: int = 6
) -> List[Article]:
    """
    Retrieve articles in the NLM catalog that are published in a journal.
    Input:
        journal: the journal to query by.
        retmax: the maximum number of search results. Default 100000.
        start_year: an optional start year to filter by. Default 2015.
        end_year: an optional end year to filter by. Default 2025.
        batch_size: the batch size to use for API requests. Default 32.
        max_retries: maximum number of API retries. Default 6.
    Returns:
        A list of articles published in the input journal.
    """
    db: Final[str] = "pmc"
    retmode: Final[str] = "xml"

    query = f"{journal.MedlineTA}[journal]"
    start = f'"{start_year}/01/01"' if start_year else '"1900"'
    end = f'"{end_year}/12/31"' if end_year else '"9999"'
    query += f" AND {start}[PDAT]:{end}[PDAT]"
    if (root := search_entrez("pmc", query, retmax=retmax)) is not None:
        ids = sum(
            [
                [str(item.text) for item in lst.findall("Id")]
                for lst in root.findall("IdList")
            ],
            []
        )
    else:
        logger.warning(f"Article search failed for journal {str(journal)}")
        ids = []

    results = []
    for batch_ids in [
        ids[pos:(pos + batch_size)] for pos in range(0, len(ids), batch_size)
    ]:
        if (root := fetch_entrez(db, batch_ids, retmode=retmode)) is None:
            continue

        for article in root.findall(".//article"):
            if (title := article.find(".//article-title")) is None:
                continue
            doi_elem = article.find(".//article-id[@pub-id-type='doi']")
            doi = None if doi_elem is None else doi_elem.text

            authors = []
            for pers in article.findall(".//contrib[@contrib-type='author']"):
                name = pers.find("name[@name-style='western']")
                if name is None:
                    continue
                surname = str(name.findtext("surname"))
                given_names = str(name.findtext("given-names"))
                orcid_elem = pers.find("contrib-id[@contrib-id-type='orcid']")
                orcid = None
                if orcid_elem is not None and orcid_elem.text:
                    orcid = orcid_elem.text.strip().split("/")[-1]
                authors.append(Author(given_names, surname, orcid))

            allyears = [
                str(elem.text)
                for elem in article.iter("year") if str(elem.text).isdigit()
            ]
            year = mode([int(x) for x in allyears]) if len(allyears) else 9999

            funding = []
            for award in article.findall(".//award-group"):
                if (source := award.find("funding-source")) is None:
                    continue
                ids = [str(_id.text) for _id in award.findall("award-id")]
                funding.append(Award(str(source.text), tuple(ids)))

            results.append(
                Article(
                    str(title.text),
                    tuple(authors),
                    tuple(funding),
                    journal,
                    year,
                    doi
                )
            )

    return results


def main(
    save_fn: Union[Path, str],
    journals: Dict[str, List[Journal]],
    time_sleep: Optional[float] = 1.0
) -> int:
    """
    Find medicine articles in the NLM Catalog.
    Input:
        save_fn: the local JSON path to save the search results to.
        journals: a map of medical broad subjects to corresponding lists of
            medical journals in the NLM catalog.
        time_sleep: an optional amount of seconds to sleep between API calls.
    Returns:
        Exit code.
    """
    assert str(save_fn).lower().endswith(".json")
    start_year: int = 2015
    end_year: int = 2025

    results: Dict[str, List[NamedTuple]] = {}
    for broad_subject, subject_journals in journals.items():
        results[broad_subject] = []
        for idx, journal in enumerate(subject_journals):
            n_journals = len(subject_journals)
            logger.info(f"Finding Articles in Journal {idx + 1}/{n_journals}")
            results[broad_subject].extend(
                get_articles(journal, start_year=start_year, end_year=end_year)
            )
            if time_sleep:
                time.sleep(time_sleep)

    save_nlm_query(
        save_fn,
        {key: list(set(val)) for key, val in results.items()},
        {
            subject: [str(start_year), str(end_year)]
            for subject in journals.keys()
        }
    )
    return 0
