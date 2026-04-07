#!/usr/bin/python3
"""
Predict genders of individuals using large language models (and other methods).

Author(s):
    Michael Yao @michael-s-yao

Portions of this code were adapted from the r4r_gender repository by
@IES-platform at https://github.com/IES-platform/r4r_gender/blob/main/
genderit/python/gender_it_functions.py

Licensed under the MIT License. Copyright University of Pennsylvania 2026.
"""
import re
import string
import unicodedata as ud
from collections.abc import Iterable
from importlib import import_module
from typing import Any, Dict, Final, List
from unidecode import unidecode

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

        self.__latin_letters: Dict[str, bool] = {}

    def predict(self, name: str) -> Gender:
        """
        Predicts the gender based on the input first name.
        Input:
            name: the first name to predict the gender of.
        Returns:
            The predicted gender.
        """
        pred = getattr(self._engine, self._entry_point)(
            self.clean_name(name), **self._kwargs
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

    def is_latin(self, uchr: str) -> bool:
        """
        Returns whether a character is in Latin-1 encoding.
        Input:
            uchr: the character to check.
        Returns:
            Whether the character is Latin-1 encoded.
        """
        try:
            return self.__latin_letters[uchr]
        except KeyError:
            return self.__latin_letters.setdefault(
                uchr, "LATIN" in ud.name(uchr)
            )

    def only_roman_chars(self, unistr: str) -> bool:
        """
        Returns whether a string includes only Latin/Roman alphabet characters.
        Input:
            unistr: the string to check.
        Returns:
            Whether the string includes only Latin/Roman alphabet characters.
        """
        return all(self.is_latin(uchr) for uchr in unistr if uchr.isalpha())

    def clean_name(self, name: str) -> str:
        """
        Cleans up an input name prior to gender determination.
        Input:
            name: the name to clean.
        Returns:
            The cleaned up name.
        """
        name = re.sub(" +", " ", str(name).lower()).strip()
        if self.only_roman_chars(name):
            return unidecode(name).strip(string.punctuation).title()
        return name.title()
