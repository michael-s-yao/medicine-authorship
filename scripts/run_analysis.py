#!/usr/bin/python3
"""
Main experimental analysis script.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2025.
"""
import click
import json
import logging
import numpy as np
import os
import pandas as pd
import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union
from umap import UMAP  # type: ignore[import-untyped]

import sys
sys.path.append(".")
from core import ce_diff_test_umap  # noqa: E402


METHOD2COLS: Dict[str, List[str]] = {
    "pip": ["gender_guesser", "global_gender_predictor"],
    "llm": [f"Llama-3.1-8B_{i + 1}" for i in range(5)],
}
METHOD2COLS["all"] = sum(METHOD2COLS.values(), [])


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


@click.command()
@click.option(
    "-m",
    "--method",
    type=click.Choice(list(METHOD2COLS.keys())),
    required=True,
    help="Gender prediction method."
)
@click.option(
    "-s",
    "--broad-subject",
    type=click.Choice([
        "allergy",
        "cardiology",
        "criticalcare",
        "endocrinology",
        "gastroenterology",
        "geriatrics",
        "infectiousdisease",
        "medicine",
        "nephrology",
        "oncology",
        "primarycare",
        "pulmonology",
        "rheumatology"
    ]),
    required=True,
    help="NIH broad subject to analyze."
)
@click.option(
    "--thresh",
    type=float,
    default=0.8,
    show_default=True,
    help="Threshold value for gender prediction."
)
@click.option(
    "--savedir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path
    ),
    default=Path.cwd() / "out",
    show_default=True,
    help="Directory to save the analysis results to."
)
@click.option(
    "--hf-dataset",
    type=str,
    default="michaelsyao/MedicineAuthorship",
    show_default=True,
    help="Huggingface dataset ID."
)
@click.option(
    "--overwrite/--skip-existing",
    default=False,
    show_default=True,
    help="Whether to overwrite or skip existing analysis files."
)
def main(
    method: str,
    broad_subject: str,
    thresh: float,
    savedir: Union[Path, str],
    hf_dataset: str,
    overwrite: bool
):
    """Medicine authorship analysis script."""
    article_ds = load_dataset(
        hf_dataset,
        data_files=f"articles/{broad_subject}.parquet",
        split="train"
    )
    gender_ds = load_dataset(
        hf_dataset,
        data_files=f"genders/{broad_subject}.parquet",
        split="train"
    )
    journal_ds = load_dataset(
        hf_dataset,
        data_files=f"journals/{broad_subject}.csv",
        split="train",
        encoding="latin_1"
    )
    embedding_ds = load_dataset(
        hf_dataset,
        data_files=f"embeddings/{broad_subject}.parquet",
        split="train"
    )
    citation_ds = load_dataset(
        hf_dataset,
        data_files=f"citations/{broad_subject}.parquet",
        split="train"
    )

    article_df = article_ds.to_pandas()
    gender_df = gender_ds.to_pandas()
    journal_df = journal_ds.to_pandas()
    citation_df = citation_ds.to_pandas()
    embedding_df: Dict[str, List[Any]] = {
        "title": [], "doi": [], "embedding": []
    }
    embedding_cols = [
        col for col in sorted(embedding_ds[0].keys())
        if col.startswith("embedding")
    ]
    for row in tqdm(embedding_ds):
        embedding_df["title"].append(row["title"])
        embedding_df["doi"].append(row["doi"])
        embedding_df["embedding"].append(
            np.array([row[col] for col in embedding_cols]).astype(np.float32)
        )
    del embedding_ds

    # Q: How often do we see female vs male authors?
    author_analysis_all_fn = os.path.join(
        str(savedir), f"all_authors_analysis_{broad_subject}.parquet"
    )
    if not os.path.isfile(author_analysis_all_fn) or overwrite:
        author_analysis(
            author_analysis_all_fn,
            article_df,
            gender_df,
            journal_df,
            citation_df,
            None,
            method,
            thresh=thresh
        )
    else:
        logger.info(f"{author_analysis_all_fn} already exists. Skipping...")

    # Q: How often do we see female vs male first authors?
    author_analysis_first_fn = os.path.join(
        str(savedir), f"first_author_analysis_{broad_subject}.parquet"
    )
    if not os.path.isfile(author_analysis_first_fn) or overwrite:
        author_analysis(
            author_analysis_first_fn,
            article_df,
            gender_df,
            journal_df,
            citation_df,
            1,
            method,
            thresh=thresh
        )
    else:
        logger.info(f"{author_analysis_first_fn} already exists. Skipping...")

    # Q: How often do we see female vs male last authors?
    author_analysis_last_fn = os.path.join(
        str(savedir), f"last_author_analysis_{broad_subject}.parquet"
    )
    if not os.path.isfile(author_analysis_last_fn) or overwrite:
        author_analysis(
            author_analysis_last_fn,
            article_df,
            gender_df,
            journal_df,
            citation_df,
            -1,
            method,
            thresh=thresh
        )
    else:
        logger.info(f"{author_analysis_last_fn} already exists. Skipping...")

    # Q: What is the proportion of male versus female authors for each paper?
    frac_gender_analysis_fn = os.path.join(
        str(savedir), f"fractional_gender_analysis_{broad_subject}.csv"
    )
    if not os.path.isfile(frac_gender_analysis_fn) or overwrite:
        fractional_gender_analysis(
            frac_gender_analysis_fn,
            article_df,
            gender_df,
            journal_df,
            citation_df,
            method,
            thresh=thresh
        )
    else:
        logger.info(f"{frac_gender_analysis_fn} already exists. Skipping...")

    # Q: How are studies funded based on the senior author's gender?
    funding_analysis_fn = os.path.join(
        str(savedir), f"funding_analysis_{broad_subject}.csv"
    )
    if not os.path.isfile(funding_analysis_fn) or overwrite:
        funding_analysis(
            funding_analysis_fn,
            article_df,
            gender_df,
            method,
            thresh=thresh
        )
    else:
        logger.info(f"{funding_analysis_fn} already exists. Skipping...")

    # Q: What is the topic of different studies based on the author gender?
    title_analysis_fn = os.path.join(
        str(savedir), f"title_analysis_{broad_subject}.csv"
    )
    if not os.path.isfile(title_analysis_fn) or overwrite:
        title_analysis(
            title_analysis_fn,
            article_df,
            gender_df,
            method,
            thresh=thresh
        )
    else:
        logger.info(f"{title_analysis_fn} already exists. Skipping...")

    # Q: What do the embeddings of the paper titles look like?
    title_embedding_analysis_fn = os.path.join(
        str(savedir), f"title_embedding_analysis_{broad_subject}.csv"
    )
    if not os.path.isfile(title_embedding_analysis_fn) or overwrite:
        title_embedding_analysis(
            embedding_df,
            title_embedding_analysis_fn,
            title_analysis_fn,
            article_df
        )
    else:
        logger.info(
            f"{title_embedding_analysis_fn} already exists. Skipping..."
        )

    title_embedding_stat_analysis(embedding_df, title_embedding_analysis_fn)


def author_analysis(
    save_fn: Union[Path, str],
    article_df: pd.DataFrame,
    gender_df: pd.DataFrame,
    journal_df: pd.DataFrame,
    citation_df: pd.DataFrame,
    author_idx: Optional[int],
    method: str,
    thresh: Optional[float] = 0.8,
    min_year: int = 2015,
    max_year: int = 2025
) -> Optional[str]:
    """
    Run analysis on the gender distributions of paper authors.
    Input:
        save_fn: the Parquet filename to save the analysis results to.
        article_df: the dataframe containing the article data.
        gender_df: the dataframe containing the gender data.
        journal_df: the dataframe containing the journal data.
        citation_df: the dataframe containing the citation and reference data.
        author_idx: an optional author index to focus the analysis on.
        method: the method of determining author gender.
        thresh: an optional threshold value (required if method is `llm` or
            `all`).
        min_year: the minimum publication year to include. Default 2015.
        max_year: the maximum publication year to include. Default 2025.
    Returns:
        None.
    """
    results = []
    for _, paper in tqdm(article_df.iterrows(), total=article_df.shape[0]):
        if paper.year > max_year or paper.year < min_year:
            continue
        citation_data = citation_df[citation_df.doi == paper.doi]
        num_total_citations = -1
        num_self_citations = -1
        num_total_references = -1
        num_self_references = -1
        if not citation_data.empty:
            num_total_citations = int(
                citation_data.iloc[0].num_total_citations
            )
            num_self_citations = int(
                citation_data.iloc[0].num_author_self_citations
            )
            num_total_references = int(
                citation_data.iloc[0].num_total_references
            )
            num_self_references = int(
                citation_data.iloc[0].num_author_self_references.item()
            )
        authors = gender_df[gender_df.title == paper.title]
        if author_idx is not None:
            authors = authors[authors.author_idx == (
                author_idx if author_idx > 0 else authors.author_idx.max()
            )]

        journal = json.loads(paper.journal)[1]
        journal_df_subset = journal_df[journal_df.Title == journal]
        if journal_df_subset.empty:
            continue
        journal_metadata = journal_df_subset.iloc[0]
        for _, single_author in authors.iterrows():
            gender: Optional[str] = None
            yp = single_author[METHOD2COLS[method]].to_numpy().squeeze()
            if yp.size == 0:
                continue

            if method.lower() in ["llm", "all"]:
                if float(np.count_nonzero(yp == "female")) / yp.size >= thresh:
                    gender = "female"
                elif float(np.count_nonzero(yp == "male")) / yp.size >= thresh:
                    gender = "male"
            elif method.lower() == "api":
                if str(yp.item()) in ["female", "male"]:
                    gender = str(yp.item())
            elif method.lower() == "pip":
                yp = np.unique(yp)
                if len(yp) == 1:
                    gender = str(yp.item())

            if gender is not None:
                results.append({
                    "gender": gender,
                    "year": paper.year,
                    "journal_is_open_access": "yes" == str.lower(
                        str(journal_metadata[f"Open Access {paper.year - 1}"])
                    ),
                    "journal_sjr": float(
                        journal_metadata[f"SJR {paper.year - 1}"]
                    ),
                    "journal_h_index": float(
                        journal_metadata[f"H index {paper.year - 1}"]
                    ),
                    "num_total_citations": num_total_citations,
                    "num_self_citations": num_self_citations,
                    "num_total_references": num_total_references,
                    "num_self_references": num_self_references
                })

    df = pd.DataFrame.from_records(results)
    return df.to_parquet(save_fn, index=False)


def fractional_gender_analysis(
    save_fn: Union[Path, str],
    article_df: pd.DataFrame,
    gender_df: pd.DataFrame,
    journal_df: pd.DataFrame,
    citation_df: pd.DataFrame,
    method: str,
    thresh: Optional[float] = 0.8,
    min_year: int = 2015,
    max_year: int = 2025
) -> Optional[str]:
    """
    Run analysis on the gender distributions of authors within each paper.
    Input:
        save_fn: the CSV filename to save the analysis results to.
        article_df: the dataframe containing the article data.
        gender_df: the dataframe containing the gender data.
        journal_df: the dataframe containing the journal data.
        citation_df: the dataframe containing the citation and reference data.
        method: the method of determining author gender.
        thresh: an optional threshold value (required if method is `llm` or
            `all`).
        min_year: the minimum publication year to include. Default 2015.
        max_year: the maximum publication year to include. Default 2025.
    Returns:
        None.
    """
    results = []
    for _, paper in tqdm(article_df.iterrows(), total=article_df.shape[0]):
        if paper.year > max_year or paper.year < min_year:
            continue
        citation_data = citation_df[citation_df.doi == paper.doi]
        num_total_citations = -1
        num_self_citations = -1
        num_total_references = -1
        num_self_references = -1
        if not citation_data.empty:
            num_total_citations = int(
                citation_data.iloc[0].num_total_citations
            )
            num_self_citations = int(
                citation_data.iloc[0].num_author_self_citations
            )
            num_total_references = int(
                citation_data.iloc[0].num_total_references
            )
            num_self_references = int(
                citation_data.iloc[0].num_author_self_references.item()
            )
        authors = gender_df[gender_df.title == paper.title]

        if authors.empty:
            continue

        journal = json.loads(paper.journal)[1]
        journal_df_subset = journal_df[journal_df.Title == journal]
        if journal_df_subset.empty:
            continue
        journal_metadata = journal_df_subset.iloc[0]

        count_female, count_male = 0, 0
        for _, single_author in authors.iterrows():
            yp = single_author[METHOD2COLS[method]].to_numpy().squeeze()
            if yp.size == 0:
                continue

            if method in ["llm", "all"]:
                if float(np.count_nonzero(yp == "female")) / yp.size >= thresh:
                    count_female += 1
                elif float(np.count_nonzero(yp == "male")) / yp.size >= thresh:
                    count_male += 1
            elif method.lower() == "api":
                if str(yp.item()) == "female":
                    count_female += 1
                elif str(yp.item()) == "male":
                    count_male += 1
            elif method.lower() == "pip":
                yp = np.unique(yp)
                if len(yp) == 1 and str(yp.item()) == "female":
                    count_female += 1
                elif len(yp) == 1 and str(yp.item()) == "male":
                    count_male += 1

        results.append({
            "freq_female": float(count_female) / authors.shape[0],
            "freq_male": float(count_male) / authors.shape[0],
            "year": paper.year,
            "journal_is_open_access": "yes" == (
                str(journal_metadata[f"Open Access {paper.year - 1}"]).lower()
            ),
            "journal_sjr": float(journal_metadata[f"SJR {paper.year - 1}"]),
            "journal_h_index": float(
                journal_metadata[f"H index {paper.year - 1}"]
            ),
            "num_total_citations": num_total_citations,
            "num_self_citations": num_self_citations,
            "num_total_references": num_total_references,
            "num_self_references": num_self_references,
            "num_total_authors": authors.shape[0]
        })

    df = pd.DataFrame.from_records(results)
    return df.to_csv(save_fn, index=False)


def funding_analysis(
    save_fn: Union[Path, str],
    article_df: pd.DataFrame,
    gender_df: pd.DataFrame,
    method: str,
    thresh: Optional[float] = 0.8,
    min_year: int = 2015,
    max_year: int = 2025
) -> Optional[str]:
    """
    Run analysis on the gender distributions of funding sources in each paper.
    Input:
        save_fn: the CSV filename to save the analysis results to.
        article_df: the dataframe containing the article data.
        gender_df: the dataframe containing the gender data.
        method: the method of determining author gender.
        thresh: an optional threshold value (required if method is `llm` or
            `all`).
        min_year: the minimum publication year to include. Default 2015.
        max_year: the maximum publication year to include. Default 2025.
    Returns:
        None.
    """
    results = []
    for _, paper in tqdm(article_df.iterrows(), total=article_df.shape[0]):
        if paper.year > max_year or paper.year < min_year:
            continue
        funding = json.loads(paper.funding)
        if len(funding) == 0 or any("None" in x for x in funding):
            continue

        authors = gender_df[gender_df.title == paper.title]
        if authors.empty:
            continue
        senior_author_idx = authors.author_idx.max()
        _, senior_author = next(
            authors[authors.author_idx == senior_author_idx].iterrows()
        )

        senior_author_gender = "unknown"
        yp = senior_author[METHOD2COLS[method]].to_numpy().squeeze()
        if yp.size == 0:
            continue

        if method in ["llm", "all"]:
            if float(np.count_nonzero(yp == "female")) / yp.size >= thresh:
                senior_author_gender = "female"
            elif float(np.count_nonzero(yp == "male")) / yp.size >= thresh:
                senior_author_gender = "male"
        elif method.lower() == "api":
            senior_author_gender = str(yp.item())
        elif method.lower() == "pip":
            yp = np.unique(yp)
            if len(yp) == 1 and str(yp.item()) == "female":
                senior_author_gender = "female"
            elif len(yp) == 1 and str(yp.item()) == "male":
                senior_author_gender = "male"

        results.extend([
            {
                "senior_author_gender": senior_author_gender,
                "funding_source": funding[i][0],
                "weight": 1.0 / len(funding)
            }
            for i in range(len(funding))
        ])

    df = pd.DataFrame.from_records(results)
    return df.to_csv(save_fn, index=False)


def title_analysis(
    save_fn: Union[Path, str],
    article_df: pd.DataFrame,
    gender_df: pd.DataFrame,
    method: str,
    thresh: Optional[float] = 0.8,
    min_year: int = 2015,
    max_year: int = 2025
) -> Optional[str]:
    """
    Run analysis on the gender distributions of article titles.
    Input:
        save_fn: the CSV filename to save the analysis results to.
        article_df: the dataframe containing the article data.
        gender_df: the dataframe containing the gender data.
        method: the method of determining author gender.
        thresh: an optional threshold value (required if method is `llm` or
            `all`).
        min_year: the minimum publication year to include. Default 2015.
        max_year: the maximum publication year to include. Default 2025.
    Returns:
        None.
    """
    results = []
    for _, paper in tqdm(article_df.iterrows(), total=article_df.shape[0]):
        if paper.year > max_year or paper.year < min_year:
            continue
        authors = gender_df[gender_df.title == paper.title]
        if authors.empty:
            continue
        last_author_idx, first_author_idx = 1, authors.author_idx.max()
        _, last_author = next(
            authors[authors.author_idx == last_author_idx].iterrows()
        )
        _, first_author = next(
            authors[authors.author_idx == first_author_idx].iterrows()
        )

        last_author_gender = "unknown"
        yp = last_author[METHOD2COLS[method]].to_numpy().squeeze()
        if yp.size == 0:
            continue

        if method in ["llm", "all"]:
            if float(np.count_nonzero(yp == "female")) / yp.size >= thresh:
                last_author_gender = "female"
            elif float(np.count_nonzero(yp == "male")) / yp.size >= thresh:
                last_author_gender = "male"
        elif method.lower() == "api":
            last_author_gender = str(yp.item())
        elif method.lower() == "pip":
            yp = np.unique(yp)
            if len(yp) == 1 and str(yp.item()) == "female":
                last_author_gender = "female"
            elif len(yp) == 1 and str(yp.item()) == "male":
                last_author_gender = "male"

        first_author_gender = "unknown"
        yp = first_author[METHOD2COLS[method]].to_numpy().squeeze()
        if yp.size == 0:
            continue
        if method.lower() in ["llm", "all"]:
            if float(np.count_nonzero(yp == "female")) / yp.size >= thresh:
                first_author_gender = "female"
            elif float(np.count_nonzero(yp == "male")) / yp.size >= thresh:
                first_author_gender = "male"
        elif method.lower() == "api":
            first_author_gender = str(yp.item())
        elif method.lower() == "pip":
            yp = np.unique(yp)
            if len(yp) == 1 and str(yp.item()) == "female":
                first_author_gender = "female"
            elif len(yp) == 1 and str(yp.item()) == "male":
                first_author_gender = "male"

        if all(
            x == "unknown" for x in [last_author_gender, first_author_gender]
        ):
            continue

        results.append({
            "first_author_gender": first_author_gender,
            "last_author_gender": last_author_gender,
            "title": paper.title
        })

    df = pd.DataFrame.from_records(results)
    return df.to_csv(save_fn, index=False)


def title_embedding_analysis(
    embedding_df: Dict[str, List[Any]],
    save_fn: Union[Path, str],
    title_analysis_fn: Union[Path, str],
    article_df: pd.DataFrame,
    seed: Optional[int] = 2025,
    **kwargs: Dict[str, Any]
) -> None:
    """
    Run UMAP to compress the embeddings of paper titles.
    Input:
        embedding_df: the dataset of raw paper embeddings
            generated from running `python scripts/embed_titles.py`.
        save_fn: the CSV filename to save the analysis results to.
        title_analysis_fn: the file path of the predicted genders of authors
            by title of the manuscript.
        article_df: the dataframe containing the article data.
        seed: optional random seed.
    Returns:
        None.
    """
    assert os.path.isfile(title_analysis_fn)
    title_analysis = pd.read_csv(title_analysis_fn)

    embeddings = np.vstack(embedding_df["embedding"])
    embeddings = embeddings / np.linalg.norm(
        embeddings, axis=-1, keepdims=True
    )

    model = UMAP(
        n_components=2,
        metric="cosine",
        random_state=seed,
        transform_seed=seed,
        init="pca",
        n_jobs=1,
        **kwargs
    )
    embeddings = model.fit_transform(embeddings)

    first_author_genders, last_author_genders, titles = [], [], []
    for title in tqdm(embedding_df["title"]):
        titles.append(title)
        authors = title_analysis[title_analysis.title == title]
        if authors.empty:
            first_author_genders.append("unknown")
            last_author_genders.append("unknown")
        else:
            first_author_genders.append(authors.first_author_gender.iloc[0])
            last_author_genders.append(authors.last_author_gender.iloc[0])

    df = pd.DataFrame({
        "title": titles,
        "first_author_gender": first_author_genders,
        "last_author_gender": last_author_genders,
        "embedding_x1": embeddings[:, 0],
        "embedding_x2": embeddings[:, 1]
    })
    return df.to_csv(save_fn, index=False)


def title_embedding_stat_analysis(
    embeddings_df: Dict[str, List[Any]],
    embedding_analysis_fn: Union[Path, str]
) -> None:
    """
    Run appropriate statistical test(s) to compare the compressed distributions
    of embeddings across genders.
    Input:
        embeddings_data_df: the dataset of raw paper embeddings generated from
            running `python scripts/embed_titles.py`.
        embedding_analysis_fn: the file path of the compressed paper embeddings
            generated from running the `title_embedding_analysis()` function.
    Returns:
        None.
    """
    full_embeddings = np.vstack(embeddings_df["embedding"])

    embedding_analysis = pd.read_csv(embedding_analysis_fn)

    umap_embeddings = (
        embedding_analysis[["embedding_x1", "embedding_x2"]].to_numpy()
    )

    first_female_idxs = np.where(
        embedding_analysis.first_author_gender == "female"
    )
    first_female_z = torch.from_numpy(full_embeddings[first_female_idxs])
    first_female_umap_z = torch.from_numpy(umap_embeddings[first_female_idxs])

    first_male_idxs = np.where(
        embedding_analysis.first_author_gender == "male"
    )
    first_male_z = torch.from_numpy(full_embeddings[first_male_idxs])
    first_male_umap_z = torch.from_numpy(umap_embeddings[first_male_idxs])

    last_female_idxs = np.where(
        embedding_analysis.last_author_gender == "female"
    )
    last_female_z = torch.from_numpy(full_embeddings[last_female_idxs])
    last_female_umap_z = torch.from_numpy(umap_embeddings[last_female_idxs])

    last_male_idxs = np.where(
        embedding_analysis.last_author_gender == "male"
    )
    last_male_z = torch.from_numpy(full_embeddings[last_male_idxs])
    last_male_umap_z = torch.from_numpy(umap_embeddings[last_male_idxs])

    first_author_result = ce_diff_test_umap(
        first_female_z, first_female_umap_z, first_male_z, first_male_umap_z
    )
    print(first_author_result.statistic, first_author_result.pvalue)

    last_author_result = ce_diff_test_umap(
        last_female_z, last_female_umap_z, last_male_z, last_male_umap_z
    )
    print(last_author_result.statistic, last_author_result.pvalue)

    female_author_result = ce_diff_test_umap(
        first_female_z, first_female_umap_z, last_female_z, last_female_umap_z
    )
    print(female_author_result.statistic, female_author_result.pvalue)

    male_author_result = ce_diff_test_umap(
        first_male_z, first_male_umap_z, last_male_z, last_male_umap_z
    )
    print(male_author_result.statistic, male_author_result.pvalue)


if __name__ == "__main__":
    main()
