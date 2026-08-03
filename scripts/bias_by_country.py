#!/usr/bin/python3
"""
Evaluate gender prediction performance broken down by country of origin.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2026.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support  # type: ignore
from typing import Final, List, Union


GENDER_LABELS: Final[List[str]] = ["male", "female", "unknown"]


def evaluate_by_country(
    answer_key_fn: Union[Path, str],
    predictions_fn: Union[Path, str],
    prediction_col: str,
    factor: float = 100.0
) -> pd.DataFrame:
    """
    Computes gender-prediction performance metrics by country of origin.
    Input:
        answer_key_fn: the path to the DataFrame with the answer key genders.
        predictions_fn: the path to the DataFrame with the predicted genders.
        prediction_col: the column in the DataFrame with the predicted genders.
        factor: multiplication factor for metrics.
    Returns:
        A DataFrame of the performance metrics.
    """
    answer_key_df = pd.read_csv(answer_key_fn)
    required_answer_cols = {"name", "ground_truth_gender", "country_of_origin"}
    assert len(required_answer_cols - set(answer_key_df.columns)) == 0

    predictions_df = pd.read_csv(predictions_fn)
    assert prediction_col in predictions_df.columns
    assert len(answer_key_df) == len(predictions_df)

    idxs = np.where(np.isin(predictions_df[prediction_col], GENDER_LABELS))
    answer_key_df = answer_key_df.iloc[idxs]  # type: ignore
    predictions_df = predictions_df.iloc[idxs]  # type: ignore

    df = answer_key_df.reset_index(drop=True).copy()
    df["predicted"] = predictions_df[prediction_col].reset_index(drop=True)

    def _score(group: pd.DataFrame) -> pd.Series:
        y_true = group["ground_truth_gender"].to_numpy()
        y_pred = group["predicted"].to_numpy()

        accuracy = (y_true == y_pred).mean()

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=GENDER_LABELS, zero_division=0
        )

        metrics = {"n": len(group), "accuracy": accuracy * factor}
        for label, p, r, f, s in zip(
            GENDER_LABELS, precision, recall, f1, support
        ):
            metrics[f"{label}_precision"] = p * factor
            metrics[f"{label}_recall"] = r * factor
            metrics[f"{label}_f1"] = f * factor
            metrics[f"{label}_support"] = s * factor

        present_idx = [i for i, s in enumerate(support) if s > 0]
        if present_idx:
            metrics["macro_precision"] = precision[present_idx].mean() * factor
            metrics["macro_recall"] = recall[present_idx].mean() * factor
            metrics["macro_f1"] = f1[present_idx].mean() * factor
        else:
            metrics["macro_precision"] = float("nan")
            metrics["macro_recall"] = float("nan")
            metrics["macro_f1"] = float("nan")

        return pd.Series(metrics)

    per_country_groups = df.groupby("country_of_origin", dropna=False)
    per_country = per_country_groups.apply(  # type: ignore
        _score, include_groups=False
    )

    overall = _score(df)
    overall.name = "__overall__"

    return pd.concat([per_country, overall.to_frame().T])


def main():
    parser = argparse.ArgumentParser(
        description="Gender analysis by country of origin"
    )
    parser.add_argument(
        "-a",
        "--answer-key",
        type=Path,
        required=True,
        help="Answer key CSV file."
    )
    parser.add_argument(
        "-p",
        "--prediction",
        type=Path,
        required=True,
        help="Gender predictions CSV file."
    )
    parser.add_argument(
        "-c",
        "--column-name",
        type=str,
        required=True,
        help="The name of the column in the CSV with the gender predictions."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output CSV file path."
    )
    args = parser.parse_args()

    assert args.answer_key.exists()
    assert args.prediction.exists()

    results_df = evaluate_by_country(
        args.answer_key.resolve(), args.prediction.resolve(), args.column_name
    )
    results_df.to_csv(str(args.output))


if __name__ == "__main__":
    main()
