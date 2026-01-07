#!/usr/bin/python3
"""
Data plotting utility functions.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2025.
"""
import math
from typing import Dict, Final


BROAD_SUBJECTS: Final[Dict[str, str]] = {
    "Medicine": "medicine.parquet",
    "Allergy": "allergy.parquet",
    "Cardiology": "cardiology.parquet",
    "Critical Care": "criticalcare.parquet",
    "Endocrinology": "endocrinology.parquet",
    "Gastroenterology": "gastroenterology.parquet",
    "Geriatrics": "geriatrics.parquet",
    "Infectious Disease": "infectiousdisease.parquet",
    "Nephrology": "nephrology.parquet",
    "Oncology": "oncology.parquet",
    "Primary Care": "primarycare.parquet",
    "Pulmonology": "pulmonology.parquet",
    "Rheumatology": "rheumatology.parquet"
}


def fmt_pval(x: float) -> str:
    if x >= 1 or x < 0:
        raise ValueError
    if x == 0:
        return r"$0.0$"
    if x >= 0.01:
        return rf"${x:.2f}$"
    if x >= 0.001:
        return rf"${x:.3f}$"

    exp = math.floor(math.log10(x))
    mantissa = x / (10 ** exp)
    return rf"${mantissa:.2f} \times 10^{{{exp}}}$"
