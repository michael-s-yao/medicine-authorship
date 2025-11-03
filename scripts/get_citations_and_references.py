#!/usr/bin/python3
"""
Extract references and citations of papers using OpenCitations API.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2025.
"""
import json
import logging
import random
import requests
import os
import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, Final, List, Optional, Tuple, Union


logger = logging.getLogger(__name__)


__BASE_URL: Final[str] = "https://api.opencitations.net/index/v1"


def get_num_references_and_citations(
    doi: Optional[str],
    base_url: str = __BASE_URL,
    max_retries: int = 6,
    timeout: Tuple[int, int] = (15, 60)
) -> Dict[str, Union[Optional[str], int]]:
    """
    Return the number of citations and references for a paper.
    Input:
        doi: the DOI of the paper.
        base_url: the base URL of the OpenCitations API.
        max_retries: maximum number of API query retries. Default 6.
        timeout: a tuple of connection and read timeouts (in seconds).
    Returns:
        A dict with the references and citations metadata.
    """
    default = {
        "doi": doi,
        "num_total_citations": -1,
        "num_journal_self_citations": -1,
        "num_author_self_citations": -1,
        "num_total_references": -1,
        "num_journal_self_references": -1,
        "num_author_self_references": -1
    }
    if doi is None:
        return default

    min_wait: Final[float] = 2.0
    max_wait: Final[float] = 5.0
    references_url: Final[str] = f"{base_url}/references/{doi}"
    citations_url: Final[str] = f"{base_url}/citations/{doi}"

    headers = {}
    if (token := os.getenv("OPENCITATIONS_ACCESS_TOKEN", None)):
        headers["authorization"] = token

    for retry_idx in range(max_retries):
        try:
            references_response = requests.get(
                references_url, headers=headers, timeout=timeout
            )
            if references_response.status_code in [404, 503]:
                return default
            references_response.raise_for_status()
            break
        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out for DOI {doi}")
            return default
        except Exception as e:
            if retry_idx == max_retries - 1:
                logger.warning(str(e))
                return default
            time.sleep(random.uniform(min_wait, max_wait))
            continue

    try:
        references: List[Dict[str, str]] = json.loads(references_response.text)
    except json.decoder.JSONDecodeError:
        return default

    for retry_idx in range(max_retries):
        try:
            citations_response = requests.get(
                citations_url, headers=headers, timeout=timeout
            )
            if citations_response.status_code in [404, 503]:
                return default
            citations_response.raise_for_status()
            break
        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out for DOI {doi}")
            return default
        except Exception as e:
            if retry_idx == max_retries - 1:
                logger.warning(str(e))
                return default
            time.sleep(random.uniform(min_wait, max_wait))
            continue

    try:
        citations: List[Dict[str, str]] = json.loads(citations_response.text)
    except json.decoder.JSONDecodeError:
        return default

    return {
        "doi": doi,
        "num_total_citations": len(citations),
        "num_journal_self_citations": len([
            cite for cite in citations
            if str(cite.get("journal_sc", "")).lower().strip() == "yes"
        ]),
        "num_author_self_citations": len([
            cite for cite in citations
            if str(cite.get("author_sc", "")).lower().strip() == "yes"
        ]),
        "num_total_references": len(references),
        "num_journal_self_references": len([
            ref for ref in references
            if str(ref.get("journal_sc", "")).lower().strip() == "yes"
        ]),
        "num_author_self_references": len([
            ref for ref in references
            if str(ref.get("author_sc", "")).lower().strip() == "yes"
        ])
    }


def main(
    articles_save_fn: Union[Path, str] = os.path.join("raw", "articles.json"),
    references_citations_savedir: Union[Path, str] = "raw"
):
    with open(articles_save_fn) as f:
        data = json.load(f)
    os.makedirs(str(references_citations_savedir), exist_ok=True)

    for broad_subject, val in data.items():
        savepath = os.path.join(
            str(references_citations_savedir), f"{broad_subject}.parquet"
        )
        results: List[Dict[Any, Any]] = []
        if os.path.isfile(savepath):
            results.extend(pd.read_parquet(savepath).to_dict("records"))
        if not len(dois := [x["doi"] for x in val["results"]]):
            continue
        assert all(x is None or str(x).startswith("10.") for x in dois)

        years = [x["year"] for x in val["results"]]
        assert all(isinstance(x, int) for x in years)
        assert len(years) == len(dois)

        print("Total Number of Articles:", len(dois))
        print(
            "Number of Articles with a DOI:", sum(x is not None for x in dois)
        )
        print(
            "Number of Articles without a DOI:", sum(x is None for x in dois)
        )

        already_read = [x["doi"] for x in results]

        for i, (doi, year) in enumerate(
            tqdm(zip(dois, years), total=len(dois))
        ):
            if doi in already_read:
                continue
            record = get_num_references_and_citations(doi)
            record["year"] = year
            results.append(record)

        pd.DataFrame.from_records(results).to_parquet(savepath, index=False)


if __name__ == "__main__":
    main()
