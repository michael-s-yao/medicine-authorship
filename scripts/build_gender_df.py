#!/usr/bin/python3
"""
Processes the raw gender predictions into the expected final data format.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2025.
"""
import json
import numpy as np
import os
import pandas as pd
from datasets import load_dataset  # type: ignore
from math import ceil
from pathlib import Path
from tqdm import tqdm
from typing import Final, List, Union


def main(
    thresh: float = 0.8,
    raw_datadir: Union[Path, str] = "raw/genders",
    hf_dataset: str = "michaelsyao/MedicineAuthorship",
    min_year: int = 2015,
    max_year: int = 2025
):
    """Process the gender predictions into the expected final data format."""
    journal_metadata = load_dataset(
        hf_dataset,
        data_files="journals/*.csv",
        split="train",
        encoding="latin1"
    )
    journal_metadata_df = journal_metadata.to_pandas()

    final_cols: Final[List[str]] = [
        "gender",
        "year",
        "journal_is_open_access",
        "journal_sjr",
        "journal_h_index",
        "num_total_citations",
        "num_self_citations",
        "num_total_references",
        "num_self_references"
    ]

    for fn in os.listdir(raw_datadir):
        df = pd.read_parquet(os.path.join(raw_datadir, fn))

        llm_output_cols = [
            col for col in df.columns if col.startswith("Llama-3.1-8B_")
        ]
        thresh = ceil(thresh * len(llm_output_cols))
        M, F, U = "male", "female", "unknown"
        m_counts = (df[llm_output_cols] == M).sum(axis=1)
        f_counts = (df[llm_output_cols] == F).sum(axis=1)
        df["gender"] = np.where(
            m_counts >= thresh, M, np.where(f_counts >= thresh, F, U)
        )

        citation_ds = load_dataset(
            hf_dataset, data_files=f"citations/{fn}", split="train"
        )
        citation_df = citation_ds.to_pandas().drop_duplicates(subset=["doi"])
        df = pd.merge(df, citation_df, on="doi", how="left")

        df = df[df.year <= max_year]
        df = df[df.year >= min_year]

        journal_ds = load_dataset(
            hf_dataset, data_files=f"articles/{fn}", split="train"
        )
        df["journal"] = df["title"].map({
            x["title"]: json.loads(x["journal"])[1] for x in journal_ds
        })

        is_open_access, sjr, h_index = [], [], []
        for _, entry in tqdm(df.iterrows(), total=len(df)):
            metadata_df = journal_metadata_df[
                journal_metadata_df.Title == entry.journal
            ]
            if metadata_df.empty:
                is_open_access.append(np.nan)
                sjr.append(np.nan)
                h_index.append(np.nan)
                continue
            metadata = metadata_df.iloc[0]
            is_open_access.append(metadata[f"Open Access {entry.year - 1}"])
            sjr.append(metadata[f"SJR {entry.year - 1}"])
            h_index.append(metadata[f"H index {entry.year - 1}"])
        df["journal_is_open_access"] = is_open_access
        df["journal_sjr"] = sjr
        df["journal_h_index"] = h_index
        df = df.rename(
            columns={
                "num_author_self_citations": "num_self_citations",
                "num_author_self_references": "num_self_references"
            }
        )

        df[final_cols].to_parquet(f"all_authors_{fn}", index=False)
        df.drop_duplicates(subset=["doi"])[final_cols].to_parquet(
            f"first_author_{fn}", index=False
        )
        df.drop_duplicates(subset=["doi"], keep="last")[final_cols].to_parquet(
            f"last_author_{fn}", index=False
        )


if __name__ == "__main__":
    main()
