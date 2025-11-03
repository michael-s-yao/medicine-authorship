#!/usr/bin/python3
"""
Embed titles of articles using a BERT-based embedding model.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2025.
"""
import json
import os
import torch
from math import ceil
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import Any, Dict, Iterator, List, Union
from unicodedata import normalize


def main(
    save_fn: Union[Path, str] = os.path.join(
        os.getcwd(), "raw", "title_embeddings.json"
    ),
    article_fn: Union[Path, str] = os.path.join("raw", "articles.json"),
    model_id: str = "allenai/scibert_scivocab_uncased",
    batch_size: int = 256
):
    with open(article_fn) as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    out: Dict[str, Dict[str, List[Any]]] = {}
    for broad_subject, val in data.items():
        out[broad_subject] = {"title": [], "doi": [], "embedding": []}
        if not len(val["results"]):
            continue

        def article_chunks() -> Iterator[List[str]]:
            for i in range(0, len(val["results"]), batch_size):
                yield val["results"][
                    i:min(i + batch_size, len(val["results"]))
                ]

        for articles in tqdm(
            article_chunks(),
            total=ceil(float(len(val["results"])) / float(batch_size))
        ):
            titles = [
                normalize(
                    "NFC", x["title"].replace("\n", "").strip()  # type: ignore
                )
                for x in articles
            ]
            with torch.no_grad():
                output = model(
                    **tokenizer(titles, return_tensors="pt", padding=True)
                )
                z = output.last_hidden_state.mean(dim=1).squeeze()
            out[broad_subject]["embedding"].extend(
                z.detach().cpu().numpy().tolist()
            )
            out[broad_subject]["title"].extend([
                x["title"] for x in articles  # type: ignore
            ])
            out[broad_subject]["doi"].extend([
                x["doi"] for x in articles  # type: ignore
            ])

    with open(save_fn, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
