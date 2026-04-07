#!/usr/bin/python3
"""
Gender prediction API based on first names.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2026.
"""
from typing import Any, Dict, List

from .registry import Registry
from .gendercast import GenderCast
from .spec import Gender
from . import llm, database, benchmarks


__all__ = [
    "register",
    "make",
    "list_registered_methods",
    "llm",
    "database",
    "benchmarks",
    "GenderCast",
    "Gender"
]


__registry = Registry()


def register(
    id_: str,
    model: str,
    entry_point: str,
    init_kwargs: Dict[str, Any],
    **kwargs: Dict[str, Any]
) -> None:
    """
    Registers a gender prediction method.
    Input:
        id_: the unique ID for the gender prediction method.
        model: the gender prediction engine object that will be
            instantiated and called for all gender predictions.
        entry_point: the function name of the model to call for gender
            predictions.
        init_kwargs: a dictionary of model initialization arguments.
        kwargs: optional keyword arguments to pass with every gender
            prediction function call.
    Returns:
        None.
    """
    __registry.register(id_, model, entry_point, init_kwargs, **kwargs)
    return


def list_registered_methods() -> List[str]:
    """
    Returns a list of the available registered methods.
    Input:
        None.
    Returns:
        A list of the available registered methods.
    """
    return __registry.registered_methods


def make(id_: str, **kwargs: Dict[str, Any]) -> GenderCast:
    """
    Instantiates a specified gender prediction method.
    Input:
        id_: the unique ID of the gender prediction method to instantiate.
    Returns:
        The specified GenderCast gender prediction object.
    """
    return __registry.make(id_, **kwargs)


register(
    id_="genderizer3",
    model="gendercast.database:Genderizer3",
    entry_point="predict_gender",
    init_kwargs={}
)


register(
    id_="gender-extractor",
    model="gender_extractor:GenderExtractor",
    entry_point="extract_gender",
    init_kwargs={}
)


register(
    id_="gender-guesser",
    model="gender_guesser.detector:Detector",
    entry_point="get_gender",
    init_kwargs={}
)


register(
    id_="global-gender-predictor",
    model="global_gender_predictor:GlobalGenderPredictor",
    entry_point="predict_gender",
    init_kwargs={}
)


register(
    id_="damegender",
    model="gendercast.database:DameGender",
    entry_point="predict_gender",
    init_kwargs={}
)


register(
    id_="genderit",
    model="gendercast.database:Genderit",
    entry_point="predict_gender",
    init_kwargs={}
)


register(
    id_="meta-llama/Llama-3.1-8B",
    model="gendercast.llm:HFModelPredictor",
    entry_point="predict_gender",
    init_kwargs={"hf_model": "meta-llama/Llama-3.1-8B"}
)


register(
    id_="openai/gpt-oss-20b",
    model="gendercast.llm:GPTOSSModelPredictor",
    entry_point="predict_gender",
    init_kwargs={"model_id": "openai/gpt-oss-20b"}
)


register(
    id_="openai/gpt-oss-120b",
    model="gendercast.llm:GPTOSSModelPredictor",
    entry_point="predict_gender",
    init_kwargs={"model_id": "openai/gpt-oss-120b"}
)
