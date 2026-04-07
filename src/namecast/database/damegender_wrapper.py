#!/usr/bin/python3
"""
DameGender gender prediction tool implementation, using the international
name data from the original tool at https://github.com/davidam/damegender

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2026.
"""
import pandas as pd
from typing import Final, List, Set


class DameGender:
    src_url: Final[str] = (
        "https://raw.githubusercontent.com/davidam/damegender/"
        "b0b9e142d1f9f0f396ff3cce61d07b1c17f7410a/src/damegender/"
        "files/names/names_inter/interall.csv"
    )

    columns: Final[List[str]] = [
        "name_upper", "n", "percent_male", "percent_female"
    ]

    # Using the same percentage threshold as in the original `damegnder`
    # package from @davidam.
    percent_thresh: Final[float] = 70.0

    def __init__(self):
        """
        Args:
            None.
        """
        df = pd.read_csv(self.src_url, header=None, names=self.columns)
        self.male_names: Final[Set[str]] = set(
            df.loc[df["percent_male"] >= self.percent_thresh, "name_upper"]
        )
        self.female_names: Final[Set[str]] = set(
            df.loc[df["percent_female"] >= self.percent_thresh, "name_upper"]
        )

    def predict_gender(self, name: str) -> str:
        """
        Predicts the gender of a first name using the DameGender tool.
        Input:
            name: the first name to predict the gender of.
        Returns:
            The predicted gender of the first name.
        """
        name = name.upper()
        if name in self.male_names and name not in self.female_names:
            return "male"
        elif name in self.female_names and name not in self.male_names:
            return "female"
        return "unknown"
