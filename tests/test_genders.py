#!/usr/bin/python3
"""
Test cases for gender prediction methods.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2026.
"""
import pytest
from typing import Dict, Final

import gendercast


test_cases: Final[Dict[str, str]] = {
    "Annie": "female",
    "Robert": "male",
    "caroline": "female",
    "DAVID": "male",
    "NotAName": "unknown"
}


@pytest.fixture(params=gendercast.list_registered_methods())
def model(request: pytest.FixtureRequest) -> gendercast.GenderCast:
    """
    Instantiates a `gendercast` gender prediction method.
    Input:
        request: a `pytest` FixtureRequest object.
    Returns:
        The specified `gendercast` gender prediction method.
    """
    return gendercast.make(request.param)


def test_predict_returns_valid_gender(model: gendercast.GenderCast) -> None:
    """
    Tests the `gendercast` gender prediction method.
    Input:
        model: the instantiated `gendercast` prediction model.
    Returns:
        None.
    """
    for name, gender in test_cases.items():
        assert model.predict(name) == gender, name


def test_predict_batch_returns_valid_genders(
    model: gendercast.GenderCast
) -> None:
    """
    Tests the batched `gendercast` gender prediction method.
    Input:
        model: the instantiated `gendercast` prediction model.
    Returns:
        None.
    """
    predictions = model.predict_batch(test_cases.keys())
    for pred, (name, gender) in zip(predictions, test_cases.items()):
        assert pred == gender, name
