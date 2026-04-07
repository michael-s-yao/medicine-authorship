#!/src/bin/python3
"""
LLM-based gender prediction methods.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2026.
"""
from .gpt_oss import GPTOSSModelPredictor
from .hf import HFModelPredictor


__all__ = ["GPTOSSModelPredictor", "HFModelPredictor"]
