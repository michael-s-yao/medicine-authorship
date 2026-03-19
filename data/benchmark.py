#!/usr/bin/python3
"""
Name-gender benchmarking datasets.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2025.
"""
import abc
import os
import pandas as pd
import requests
import unicodedata
import zipfile
from torch.utils.data import Dataset
from typing import Dict, Final, Tuple, Union
from pathlib import Path


_headers: Final[Dict[str, str]] = {
    "User-Agent": "Wget/1.21",
    "Accept": "*/*",
    "Accept-Encoding": "identity",
    "Connection": "Keep-Alive"
}


class BenchmarkDataset(Dataset, abc.ABC):
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: the benchmarking dataset.
        """
        self._data = df

    def __len__(self) -> int:
        """
        Returns the number of unique names in the dataset.
        Input:
            None.
        Returns:
            The number of unique names in the dataset.
        """
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        """
        Returns a name-gender pair from the dataset.
        Input:
            idx: the index of the pair to retrieve from the dataset.
        Returns:
            The specified name-gender-country tuple.
        """
        data = self._data.iloc[idx].to_dict()
        return data["name"].title(), data["gender"], data["country"]


class SSABenchmarkDataset(BenchmarkDataset):
    """
    A dataset of US Social Security Administration baby names and
    their associated genders.
    References:
        [1] Popular baby names. US Social Security Administration. Accessed 18
            March 2026. URL: https://www.ssa.gov/oact/babynames/limits.html
    """

    src_url: Final[Union[Path, str]] = (
        "https://www.ssa.gov/oact/babynames/names.zip"
    )

    def __init__(
        self,
        local_dir: Union[Path, str] = Path.home() / ".cache" / "ssa_dataset",
        min_birth_year: int = 1940,
        max_birth_year: int = 2024
    ):
        """
        Args:
            local_dir: the local directory to save the dataset to.
            min_birth_year: the minimum birth year to include data for.
            max_birth_year: the maximum birth year to include data for.
        """
        os.makedirs(os.path.dirname(str(local_dir)), exist_ok=True)
        if not os.path.isdir(str(local_dir)):
            filename = str(self.src_url).split("/")[-1]

            with requests.get(
                str(self.src_url), headers=_headers, stream=True
            ) as response:
                response.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            with zipfile.ZipFile(filename, "r") as zref:
                zref.extractall(str(local_dir))

            os.remove(filename)

        assert min_birth_year < max_birth_year
        df = pd.concat([
            pd.read_csv(
                os.path.join(str(local_dir), f"yob{year}.txt"),
                names=["name", "gender", "count"],
                header=None
            )
            for year in range(min_birth_year, max_birth_year + 1)
        ])
        ds = df.groupby("name")["gender"].agg(lambda x: x.mode()[0])
        df = ds.reset_index()
        df["country"] = "US"

        super(SSABenchmarkDataset, self).__init__(df)


class PinyinBenchmarkDataset(BenchmarkDataset):
    """
    A dataset of Chinese Pinyin names and their associated genders.
    References:
        [1] Shi D, Tong ST. An open dataset of Chinese name-to gender
            associations for gender prediction in broad scientific research.
            Sci Data 12(1962). (2025). doi: 10.1038/s41597-025-06276-y
    """

    src_url: Final[Union[Path, str]] = (
        "https://dataverse.harvard.edu/api/access/datafile/10803451"
    )

    def __init__(self, local_dir: Union[Path, str] = Path.home() / ".cache"):
        """
        Args:
            local_dir: the local directory to save the dataset to.
        """
        os.makedirs(str(local_dir), exist_ok=True)
        fn = os.path.join(str(local_dir), "pinyin_dataset.txt")
        if not os.path.isfile(fn):
            with requests.get(str(self.src_url), headers=_headers) as response:
                response.raise_for_status()
                with open(fn, "w") as f:
                    f.write(response.text)

        df = pd.read_csv(fn, sep="\t").rename(columns={"PinyinName": "name"})
        df["gender"] = (df["Male"] > df["Female"]).map({False: "F", True: "M"})
        df = df[["name", "gender"]]
        df["country"] = "CN"

        super(PinyinBenchmarkDataset, self).__init__(df)


class GlobalBenchmarkDataset(BenchmarkDataset):
    """
    A dataset of global forenames from different countries and their associated
    genders.
    References:
        [1] Martinez GL, Saenz da Juana-i-Ribes H, Yin D, et al. Expanding the
            World Gender-Name Dictionary: WGND 2.0. World Intellectual Property
            Organization: Economics Working Papers. (2021). doi:
            10.34667/tind.43980
    """

    src_url: Final[Union[Path, str]] = (
        "https://dataverse.harvard.edu/api/access/datafile/4750348"
    )

    def __init__(self, local_dir: Union[Path, str] = Path.home() / ".cache"):
        """
        Args:
            local_dir: the local directory to save the dataset to.
        """
        os.makedirs(str(local_dir), exist_ok=True)
        fn = os.path.join(str(local_dir), "global_dataset.txt")
        if not os.path.isfile(fn):
            with requests.get(str(self.src_url), headers=_headers) as response:
                response.raise_for_status()
                with open(fn, "w") as f:
                    f.write(response.text)

        df = pd.read_csv(fn, sep="\t")
        df["name"] = df["name"].map(
            lambda x: "".join(
                c for c in str(x).encode("latin-1").decode("utf-8")
                if not unicodedata.category(c).startswith("P")
            )
        )
        df = df.rename(columns={"code": "country"})
        df = df.sort_values("wgt", ascending=False)
        df = df.drop_duplicates(subset=["name", "country"])
        df = df[["name", "gender", "country"]]
        df = df[df["gender"].isin(["M", "F"])]

        super(GlobalBenchmarkDataset, self).__init__(df)
