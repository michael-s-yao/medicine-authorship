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
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Union

import gendercast


def main(
    save_fn: Union[Path, str],
    first_names: Union[
        gendercast.benchmarks.BenchmarkDataset, List[List[str]]
    ],
    model_id: str,
    ckpt_frequency: int
) -> int:
    """
    Predict genders of authors using a large language model.
    Input:
        save_fn: the local JSON path to save the predictions to.
        first_names: a list of author lists of the first names of the authors.
        model_id: the model ID of the gender prediction method to use.
        ckpt_frequency: the frequency to checkpoint the prediction results.
    Returns:
        Exit code.
    """
    assert str(save_fn).lower().endswith(".json")

    model = gendercast.make(model_id)

    labels = []
    num_already_completed = 0
    if os.path.isfile(save_fn):
        with open(save_fn) as f:
            labels = json.load(f)
        num_already_completed = len(labels)

    for i in tqdm(range(num_already_completed, len(first_names))):
        if isinstance(first_names, gendercast.benchmarks.BenchmarkDataset):
            names = [first_names[i][0]]
        else:
            names = first_names[i]
        labels.append({
            name: label
            for name, label in zip(names, model.predict_batch(names))
        })

        if i >= len(first_names) - 1 or (
            ckpt_frequency > 0 and i % ckpt_frequency == 0
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
        "-m",
        "--method",
        choices=gendercast.list_registered_methods(),
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
        "--benchmark",
        type=str,
        default=None,
        help="An optional benchmark to run."
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

    if args.savedir:
        os.makedirs(args.savedir, exist_ok=True)

    benchmark: Optional[gendercast.benchmarks.BenchmarkDataset] = None
    run_idx = "" if args.idx < 0 else str(args.idx)
    suffix = args.method.split("/")[-1]
    if args.benchmark is not None:
        main(
            os.path.join(
                args.savedir,
                f"genders_{args.benchmark}_{run_idx}_{suffix}.json"
            ),
            getattr(gendercast.benchmarks, args.benchmark)(),
            args.method,
            args.ckpt_frequency
        )
    else:
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
                args.method,
                args.ckpt_frequency
            )
