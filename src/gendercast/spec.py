#!/usr/bin/python3
"""
Gender prediction method specification.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2026.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Literal


@dataclass
class ModelSpec:
    id_: str
    model: str
    entry_point: str
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    kwargs: Dict[str, Any] = field(default_factory=dict)


Gender = Literal["male", "female", "unknown"]
