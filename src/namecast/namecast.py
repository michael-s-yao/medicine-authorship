#!/usr/bin/python3
"""
Predict genders of individuals using large language models (and other methods).

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2026.
"""
from collections.abc import Iterable
from importlib import import_module
from typing import Any, Dict, Final, List

from .spec import Gender, ModelSpec


class NameCast:
    def __init__(self, model_spec: ModelSpec, **kwargs: Dict[str, Any]):
        """
        Args:
            model_spec: the NameCast model specification.
        """
        self._model_spec: Final[ModelSpec] = model_spec
        self._kwargs: Final[Dict[str, Any]] = kwargs

        model_mod, model_cls = self._model_spec.model.split(":", 1)
        self._engine = getattr(import_module(model_mod), model_cls)(
            **self._model_spec.init_kwargs
        )
        self._entry_point: Final[str] = self._model_spec.entry_point

    def predict(self, name: str) -> Gender:
        """
        Predicts the gender based on the input first name.
        Input:
            name: the first name to predict the gender of.
        Returns:
            The predicted gender.
        """
        pred = getattr(self._engine, self._entry_point)(
            name.title(), **self._kwargs
        )
        return self.as_gender(pred)

    def predict_batch(self, name_batch: Iterable[str]) -> List[Gender]:
        """
        Batch gender prediction helper method.
        Input:
            name_batch: a batch of names to predict the gender for.
        Returns:
            A list of the predicted genders.
        """
        return [self.predict(name) for name in name_batch]

    def as_gender(self, gender: str) -> Gender:
        """
        Casts a gender prediction into a Gender.
        Input:
            gender: the gender prediction to cast.
        Returns:
            The casted Gender.
        """
        if gender.lower() == "m" or gender.lower() == "male":
            return "male"
        elif gender.lower() == "f" or gender.lower() == "female":
            return "female"
        return "unknown"
