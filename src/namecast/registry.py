#!/usr/bin/python3
"""
Gender prediction method registry.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2026.
"""
from typing import Any, Dict, List

from .namecast import NameCast
from .spec import ModelSpec


class Registry:
    def __init__(self):
        """
        Args:
            None.
        """
        self._specs: Dict[str, ModelSpec] = {}

    def register(
        self,
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
        if id_ in self._specs:
            raise ValueError(f"Model '{id}' is already registered.")
        if model.count(":") != 1 or model[0] == ":" or model[-1] == ":":
            raise ValueError(f"Invalid model '{model}' specificied.")
        self._specs[id_] = ModelSpec(
            id_=id_,
            model=model,
            entry_point=entry_point,
            init_kwargs=init_kwargs,
            kwargs=kwargs
        )

    @property
    def registered_methods(self) -> List[str]:
        """
        Returns a list of the available registered methods.
        Input:
            None.
        Returns:
            A list of the available registered methods.
        """
        return list(self._specs.keys())

    def make(self, id_: str, **calltime_kwargs: Dict[str, Any]) -> NameCast:
        """
        Instantiates a specified gender prediction method.
        Input:
            id_: the unique ID of the gender prediction method to instantiate.
        Returns:
            The specified NameCast gender prediction object.
        """
        if id_ not in self._specs:
            raise ValueError(
                f"No model registered with id '{id_}'. "
                f"Registered options: {self.registered_methods}"
            )
        spec = self._specs[id_]
        merged_kwargs = {**spec.kwargs, **calltime_kwargs}
        return NameCast(spec, **merged_kwargs)
