#!/usr/bin/python3
"""
Gender prediction using open-weight HuggingFace models.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2026.
"""
import numpy as np
import torch
import transformers
from transformers.pipelines.text_generation import TextGenerationPipeline
from typing import Any, Dict, Final, Optional


class HFModelPredictor:
    def __init__(
        self,
        hf_model: str,
        seed: Optional[int] = None,
        **kwargs: Dict[str, Any]
    ):
        """
        Args:
            hf_model: the HuggingFace model to use.
            seed: optional random seed.
        """
        del kwargs
        self.model_id: Final[str] = hf_model
        self.model: Final[TextGenerationPipeline] = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"dtype": torch.bfloat16},
            device_map="auto"
        )
        self.seed: Final[Optional[int]] = seed
        self._rng = np.random.default_rng(seed=self.seed)

    def predict_gender(self, name: str) -> str:
        """
        Predicts the gender of an individual based on their first name.
        Input:
            name: the first name to predict the gender for.
        Returns:
            The predicted gender associated with the name.
        """
        gender = (
            "male or female" if self._rng.choice([True, False])
            else "female or male"
        )
        prefix = f"Given a name, respond with whether the person is {gender}."
        examples = "Alice: female\n\nBob: male"
        if self._rng.choice([True, False]):
            examples = "Adam: male\n\nBella: female"
        prompt = f"{prefix}\n\n{examples}\n\n{name}: "

        pred = self.model([prompt], max_new_tokens=1, temperature=0.01)
        label = pred[0][0]["generated_text"].replace(prompt, "")
        label = label.strip().lower()
        if label.startswith("f"):
            return "female"
        elif label.startswith("m"):
            return "male"
        return "unknown"
