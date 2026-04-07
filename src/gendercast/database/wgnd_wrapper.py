#!/usr/bin/python3
"""
Wrapper around the genderit gender prediction method to work nicely with
our package.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Martinez GL, Saenz da Juana-i-Ribes H, Yin D, et al. Expanding the
        World Gender-Name Dictionary: WGND 2.0. World Intellectual Property
        Organization: Economics Working Papers. (2021). doi:
        10.34667/tind.43980

Note(s):
    The `genderit` package is available from the r4r_gender repository by
    @IES-platform at https://github.com/IES-platform/r4r_gender. Note that
    this database method uses a database of name-gender pairs that is
    exactly the same as the GlobalBenchmarkDataset dataset in this package.

Licensed under the MIT License. Copyright University of Pennsylvania 2026.
"""
import os
import requests
import pandas as pd
from io import StringIO
from typing import Any, Dict, Final, Set, Union
from pathlib import Path


class Genderit:
    src_url: Final[str] = (
        "https://dataverse.harvard.edu/api/access/datafile/4750351"
    )

    cachedir: Final[Union[Path, str]] = Path.home() / ".cache" / "wgnd"

    def __init__(self, **kwargs: Dict[str, Any]):
        del kwargs

        os.makedirs(str(self.cachedir), exist_ok=True)
        cachepath = os.path.join(str(self.cachedir), "d3.csv.gz")
        if not os.path.isfile(cachepath):
            with requests.Session() as session:
                response = session.get(self.src_url, timeout=60)
                response.raise_for_status()
                df = pd.read_csv(
                    StringIO(response.content.decode("utf-8")), sep="\t"
                )
                df.to_csv(cachepath, index=False, compression="gzip")
        else:
            df = pd.read_csv(cachepath, compression="gzip")

        df["name"] = df["name"].str.lower()
        self.male_names: Set[str] = set(df.loc[df["gender"] == "M", "name"])
        self.female_names: Set[str] = set(df.loc[df["gender"] == "F", "name"])

    def predict_gender(self, name: str) -> str:
        """
        Predicts the gender of a first name using the genderit tool.
        Input:
            name: the first name to predict the gender of.
        Returns:
            The predicted gender of the first name.
        """
        name = name.lower()
        if name in self.male_names and name not in self.female_names:
            return "male"
        elif name in self.female_names and name not in self.male_names:
            return "female"
        return "unknown"
