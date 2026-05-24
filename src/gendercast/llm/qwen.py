#!/usr/bin/python3
"""
Gender prediction using Qwen 3.6 27B LLM.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2026.
"""
import numpy as np
import os
import random
import time
from openai import OpenAI
from typing import Any, Dict, Final, Optional


class Qwen3ModelPredictor:
    def __init__(
        self,
        seed: Optional[int] = None,
        max_retries: int = 6,
        **kwargs: Dict[str, Any]
    ):
        """
        Args:
            seed: optional random seed.
            max_retries: maximum number of API retries. Default 6.
        """
        self.model_id: Final[str] = "Qwen/Qwen3.6-27B"
        self.client = OpenAI(
            base_url="http://localhost:8000/v1", api_key=""
        )
        self.seed: Final[Optional[int]] = seed
        self.max_retries: Final[int] = 6
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
        prefix = f"Given a name, respond with whether the person is {gender}. "
        prefix += "You must respond with one of ['male', 'female', 'unknown']."
        examples = "Alice: female\n\nBob: male"
        if self._rng.choice([True, False]):
            examples = "Adam: male\n\nBella: female"
        prompt = f"{prefix}\n\n{examples}\n\n{name}: "

        for retry_idx in range(self.max_retries):
            response = self.client.responses.create(
                model=self.model_id,
                input=prompt,
                max_output_tokens=16,
                temperature=0.01,
                truncation="auto"
            )
            try:
                pred = response.output[-1].content[0].text  # type: ignore
                print(pred)
                break
            except IndexError:
                return "unknown"
                continue
            except AttributeError as e:
                if retry_idx == self.max_retries - 1:
                    raise e
                time.sleep(random.uniform(5, 10))
        if "female" in pred.lower().strip():
            return "female"
        elif "male" in pred.lower().strip():
            return "male"
        return "unknown"
