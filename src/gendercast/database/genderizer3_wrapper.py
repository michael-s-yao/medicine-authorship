#!/usr/bin/python3
"""
Wrapper around the genderizer3 gender prediction method to work nicely with
our package.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2026.
"""
from genderizer3.genderizer3 import Genderizer  # type: ignore
from typing import Any, Dict


class Genderizer3:
    def __init__(self, **kwargs: Dict[str, Any]):
        """
        Args:
            None.
        """
        del kwargs

    def predict_gender(self, name: str) -> str:
        """
        Predicts the gender of an individual based on their first name.
        Input:
            name: the first name to predict the gender for.
        Returns:
            The predicted gender associated with the name.
        """
        return Genderizer.detect(firstName=name)
