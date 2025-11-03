#!/usr/bin/python3
"""
Predict genders of individuals using large language models.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2025.
"""
import argparse
import json
import os
import random
import time
import transformers
import torch
from genderizer3.genderizer3 import Genderizer  # type: ignore
from gender_guesser.detector import Detector  # type: ignore
from global_gender_predictor import (  # type: ignore
    GlobalGenderPredictor
)
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
from transformers.pipelines.text_generation import TextGenerationPipeline
from typing import Final, List, Union


def predict_genders(
    first_names: List[str],
    model: Union[str, Detector, GlobalGenderPredictor, TextGenerationPipeline],
    max_new_tokens: int = 1,
    batch_size: int = 1,
    temperature: float = 0.01,
    max_retries: int = 6
) -> List[str]:
    """
    Predict the genders of a list of individuals.
    Input:
        first_names: an author lists of the first names of the authors.
        model: the model for gender prediction to use.
        max_new_tokens: maximum number of new tokens to generate. Default 1.
        batch_size: batch size for model inference. Default 1.
        temperature: LLM temperature parameter. Default 0.01.
        max_retries: maximum number of API retries. Default 6.
    Returns:
        The predicted genders of the authors. One of `male`, `female`, or
        `unknown`.
    """
    if isinstance(model, (Detector, GlobalGenderPredictor)) or (
        model == "genderizer3"
    ):
        results = []
        for name in first_names:
            if hasattr(model, "get_gender"):
                pred = model.get_gender(name)
            elif hasattr(model, "predict_gender"):
                pred = model.predict_gender(name)
            else:
                pred = Genderizer.detect(firstName=name)
            if "andy" in pred.lower() or "unknown" in pred.lower():
                results.append("unknown")
            elif "female" in pred.lower():
                results.append("female")
            else:
                results.append("male")
        return results

    gender = "male or female" if random.randint(0, 1) else "female or male"
    prefix = f"Given a name, respond with whether the person is {gender}."
    examples = "Alice: female\n\nBob: male"
    if random.randint(0, 1):
        examples = "Adam: male\n\nBella: female"
    prompts = [f"{prefix}\n\n{examples}\n\n{name}: " for name in first_names]
    results = []
    if not len(prompts):
        return results

    if any(
        str(model).startswith(x)
        for x in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    ):
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.getenv("HF_TOKEN")
        )
        # Hack to turn off model reasoning. See https://huggingface.co/openai/
        # gpt-oss-120b/discussions/50 for more details.
        suffix: Final[str] = (
            "<|end|><|start|>assistant<|channel|>analysis<|message|><|end|>"
            "<|start|>assistant<|channel|>"
        )
        for inp in prompts:
            for retry_idx in range(max_retries):
                response = client.responses.create(
                    model=str(model),
                    input=(inp + suffix),
                    max_output_tokens=128,
                    temperature=temperature,
                    truncation="auto",
                    reasoning={"effort": "low"}
                )
                try:
                    pred = response.output[-1].content[0].text  # type: ignore
                    break
                except IndexError:
                    results.append("unknown")
                    continue
                except AttributeError as e:
                    if retry_idx == max_retries - 1:
                        raise e
                    time.sleep(random.uniform(5, 10))
            if "female" in pred.lower().strip():
                results.append("female")
            elif "male" in pred.lower().strip():
                results.append("male")
            else:
                results.append("unknown")
        return results

    assert isinstance(model, TextGenerationPipeline)
    for i, prediction in enumerate(
        model(
            prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            temperature=temperature
        )
    ):
        label = prediction[0]["generated_text"].replace(prompts[i], "")
        label = label.strip().lower()
        if label in ["female", "male"]:
            results.append(label)
        else:
            results.append("unknown")

    return results


def main(
    save_fn: Union[Path, str],
    first_names: List[List[str]],
    model_id: Union[str, Detector],
    ckpt_frequency: int
) -> int:
    """
    Predict genders of authors using a large language model.
    Input:
        save_fn: the local JSON path to save the predictions to.
        first_names: a list of author lists of the first names of the authors.
        model_id: the model ID of the LLM to use.
        ckpt_frequency: the frequency to checkpoint the prediction results.
    Returns:
        Exit code.
    """
    assert str(save_fn).lower().endswith(".json")

    labels = []
    num_already_completed = 0
    if os.path.isfile(save_fn):
        with open(save_fn) as f:
            labels = json.load(f)
        num_already_completed = len(labels)

    for i, names in enumerate(tqdm(first_names[num_already_completed:])):
        labels.append({
            name: label
            for name, label in zip(names, predict_genders(names, model_id))
        })

        if (ckpt_frequency > 0 and i % ckpt_frequency == 0) or (
            len(labels) == len(first_names)
        ):
            with open(save_fn, "w") as f:
                json.dump(labels, f, indent=2)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gender prediction using LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--idx",
        type=int,
        default=-1,
        help="An optional run index."
    )
    parser.add_argument(
        "--llm",
        choices=[
            "meta-llama/Llama-3.1-8B",
            "google/gemma-3-270m",
            "microsoft/bitnet-b1.58-2B-4T",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "gender-guesser",
            "global-gender-predictor",
            "genderizer3"
        ],
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="The model ID of the LLM (or deterministic method) to use."
    )
    parser.add_argument(
        "--articles-fn",
        type=str,
        default="raw/articles.json",
        help="The path to the file with article metadata."
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default=os.path.join(os.getcwd(), "raw"),
        help="The directory to save the prediction results to."
    )
    parser.add_argument(
        "--ckpt-frequency",
        type=int,
        default=1000,
        help="The frequency to checkpoint the prediction results to."
    )
    args = parser.parse_args()

    assert args.articles_fn.lower().endswith(".json")
    with open(args.articles_fn) as f:
        data = json.load(f)

    model: Union[
        str, Detector, GlobalGenderPredictor, TextGenerationPipeline
    ] = args.llm
    if str(model).lower() == "gender-guesser":
        model = Detector()
    elif str(model).lower() == "global-gender-predictor":
        model = GlobalGenderPredictor()
    elif not any(
        str(model).startswith(x)
        for x in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    ):
        model = transformers.pipeline(
            "text-generation",
            model=str(model),
            model_kwargs={"dtype": torch.bfloat16},
            device_map="auto"
        )

    if args.savedir:
        os.makedirs(args.savedir, exist_ok=True)

    run_idx = "" if args.idx < 0 else str(args.idx)
    suffix = args.llm.split("/")[-1]
    for broad_subject, val in data.items():
        # The first name is always the 0th element in our implementation.
        first_names = [
            [name[0] for name in x["authors"]] for x in val["results"]
        ]
        if not len(first_names):
            continue
        main(
            os.path.join(
                args.savedir,
                f"genders_{broad_subject}_{run_idx}_{suffix}.json"
            ),
            first_names,
            model,
            args.ckpt_frequency
        )
