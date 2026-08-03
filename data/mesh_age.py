#!/usr/bin/python3
"""
Classify each manuscript's study population as pediatric, adult, or mixed
using MEDLINE's MeSH age-group Check Tags.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2026.
"""
import logging
import time
from tqdm import tqdm
from typing import Dict, FrozenSet, List, Optional, Set

from .entrez import fetch_entrez


logger = logging.getLogger(__name__)


PEDIATRIC_CHECK_TAGS: FrozenSet[str] = frozenset({
    "Infant, Newborn",
    "Infant",
    "Child, Preschool",
    "Child",
    "Adolescent"
})


ADULT_CHECK_TAGS: FrozenSet[str] = frozenset({
    "Young Adult",
    "Adult",
    "Middle Aged",
    "Aged",
    "Aged, 80 and over",
})


def classify_age_group(mesh_terms: Set[str]) -> str:
    """
    Classify a manuscript's study population from its MeSH descriptors.
    Input:
        mesh_terms: the set of MeSH DescriptorName strings attached to the
            article's MEDLINE record (not just check tags - any MeSH term
            set works, since this only intersects the two tag sets).
    Returns:
        One of "pediatric", "adult", "mixed", or "unknown"
    """
    is_pediatric = bool(mesh_terms & PEDIATRIC_CHECK_TAGS)
    is_adult = bool(mesh_terms & ADULT_CHECK_TAGS)
    if is_pediatric and is_adult:
        return "mixed"
    elif is_pediatric:
        return "pediatric"
    elif is_adult:
        return "adult"
    return "unknown"


def get_age_groups(
    pmids: List[Optional[str]],
    batch_size: int = 100,
    time_sleep: Optional[float] = 0.5,
    max_retries: int = 6,
    quiet: bool = False,
    desc: Optional[str] = None
) -> Dict[str, str]:
    """
    Fetch MeSH age-group classifications for a batch of PubMed IDs.
    Input:
        pmids: the PMIDs to query.
        batch_size: number of PMIDs per efetch call.
        time_sleep: optional seconds to sleep between batches, to stay
            within NCBI's rate limits.
        max_retries: maximum number of API retries.
        quiet: whether to turn off progress bars.
        desc: optional description of the set of PMIDs for the progress bar.
    Returns:
        A dictionary mapping input PMIDs to target patient populations.
    """
    results: Dict[str, str] = {}
    unique_pmids = sorted({p for p in pmids if p})

    for start in tqdm(
        range(0, len(unique_pmids), batch_size), disable=quiet, desc=desc
    ):
        batch = unique_pmids[start:(start + batch_size)]
        root = fetch_entrez(
            "pubmed", batch, retmode="xml", max_retries=max_retries
        )
        if root is None:
            logger.warning(
                f"MeSH fetch failed for PMID batch starting at index {start}"
            )
            continue

        for record in root.findall(".//PubmedArticle"):
            pmid_elem = record.find(".//MedlineCitation/PMID")
            if pmid_elem is None or not pmid_elem.text:
                continue
            pmid = str(pmid_elem.text)

            mesh_terms = {
                str(desc.text)
                for desc in record.findall(
                    ".//MeshHeadingList/MeshHeading/DescriptorName"
                )
                if desc.text
            }
            results[pmid] = classify_age_group(mesh_terms)

        if time_sleep:
            time.sleep(time_sleep)

    return results
