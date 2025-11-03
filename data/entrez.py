#!/usr/bin/python3
"""
Base function implementations for interacting with the Entrez API.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2025.
"""
import json
import logging
import os
import random
import requests
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Final, List, NamedTuple, Optional, Tuple, Union
from xml.etree import ElementTree as ET


logger = logging.getLogger(__name__)


__BASE_URL: Final[str] = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


__min_wait: Final[float] = 2.0


__max_wait: Final[float] = 5.0


def search_entrez(
    db: str,
    query: str,
    base_url: str = __BASE_URL,
    retmode: Optional[str] = "xml",
    retmax: Optional[int] = 100000,
    max_retries: int = 6,
    timeout: Tuple[int, int] = (15, 1200)
) -> Optional[ET.Element]:
    """
    Query the search Entrez API.
    Input:
        db: the database to query.
        query: the search query.
        base_url: the base API URL.
        retmode: the return type.
        retmax: the maximum number of entries to return.
        max_retries: maximum number of API query retries. Default 6.
        timeout: a tuple of connection and read timeouts (in seconds).
    Returns:
        The root of an XML Element Tree with the query results.
    """
    search_url: Final[str] = f"{base_url}/esearch.fcgi"
    params: Dict[str, str] = {"db": db, "term": query}
    if retmode is not None:
        params["retmode"] = retmode
    if retmax is not None:
        params["retmax"] = str(retmax)

    for retry_idx in range(max_retries):
        try:
            response = requests.get(search_url, params=params, timeout=timeout)
            if response.status_code in [404, 503]:
                return None
            response.raise_for_status()
            break
        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out for search query {params}")
            return None
        except Exception as e:
            if retry_idx == max_retries - 1:
                logger.warning(str(e))
                return None
            time.sleep(random.uniform(__min_wait, __max_wait))
            continue

    return ET.fromstring(response.text)


def fetch_entrez(
    db: str,
    ids: List[str],
    base_url: str = __BASE_URL,
    retmode: Optional[str] = "xml",
    max_retries: int = 6,
    timeout: Tuple[int, int] = (15, 1200)
) -> Optional[ET.Element]:
    """
    Query the fetch Entrez API.
    Input:
        db: the database to query.
        ids: the IDs of the objects to fetch.
        base_url: the base API URL.
        retmode: the return type.
        max_retries: maximum number of API query retries. Default 6.
        timeout: a tuple of connection and read timeouts (in seconds).
    Returns:
        The root of an XML Element Tree with the query results.
    """
    fetch_url: Final[str] = f"{base_url}/efetch.fcgi"
    params = {"db": db, "id": ",".join(map(str, ids))}
    if retmode is not None:
        params["retmode"] = retmode

    for retry_idx in range(max_retries):
        try:
            response = requests.get(fetch_url, params=params, timeout=timeout)
            if response.status_code in [404, 503]:
                return None
            response.raise_for_status()
            break
        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out for fetch query {params}")
            return None
        except Exception as e:
            if retry_idx == max_retries - 1:
                logger.warning(str(e))
                return None
            time.sleep(random.uniform(__min_wait, __max_wait))
            continue

    return ET.fromstring(response.text)


def save_nlm_query(
    save_fn: Union[Path, str],
    results: Dict[str, List[NamedTuple]],
    search_terms: Dict[str, List[str]]
) -> None:
    """
    Locally cache the results of an NLM query.
    Input:
        save_fn: the local path to save the results to.
        results: the query results by parent subject to locally cache.
        search_terms: the search term(s) by parent subject used to generate
            the results.
    Returns:
        None.
    """
    now = datetime.now(timezone.utc).isoformat()
    data: Dict[str, Dict[str, Any]] = {}
    for key in results.keys():
        serialized_results: List[Dict[str, str]] = [
            entry._asdict() for entry in sorted(
                list(set(results[key])),
                key=lambda journal: getattr(journal, "Title", "None")
            )
        ]
        data[key] = {
            "time": now,
            "search_terms": search_terms[key],
            "results": serialized_results
        }

    if len(parentdir := os.path.dirname(str(save_fn))):
        os.makedirs(parentdir, exist_ok=True)
    with open(save_fn, "w") as f:
        json.dump(data, f, indent=2)
