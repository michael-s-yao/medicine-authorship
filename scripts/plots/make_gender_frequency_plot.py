#!/usr/bin/python3
"""
Make gender frequency radar plots.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2025.
"""
import click
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import statsmodels.api as sm  # type: ignore
import sys
from pathlib import Path
from scipy.stats import t
from typing import Any, Dict, Final, List, Optional, Union

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
from core.plot_utils import BROAD_SUBJECTS  # noqa


@click.command()
@click.option(
    "--datadir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="Medicine Authorship",
    show_default=True,
    help="Data directory."
)
@click.option(
    "--savedir",
    type=str,
    default=None,
    show_default=True,
    help="Optional figure savepath."
)
def main(datadir: Union[Path, str], savedir: Optional[Union[Path, str]]):
    """Make gender frequency forest plots."""
    plt.rcParams["font.family"] = ["Arial"]

    m_df = get_data("male", datadir=datadir)
    f_df = get_data("female", datadir=datadir)
    m_change_df = get_change_data("male", datadir=datadir)
    f_change_df = get_change_data("female", datadir=datadir)

    pos_label: Final[float] = -100.0
    pos_nauthors: Final[float] = -50.0
    pos_r2: Final[float] = 0.035
    BLUE: Final[str] = "#3170AD"
    RED: Final[str] = "#90312C"

    fig, (ax, change_ax) = plt.subplots(1, 2, figsize=(12, 12))

    ax.set_ylim(len(m_df) + 0.5, -1.0)
    for i in range(len(m_df)):
        if (i % 2):
            continue
        ax.axhspan(
            i - 0.55,
            i + 0.45,
            xmin=-1.05,
            xmax=2.85,
            color="gray",
            alpha=0.1,
            ec=None,
            clip_on=False
        )
    ax.set_yticks([])
    ax.set_yticklabels([])
    change_ax.set_ylim(len(m_df) + 0.5, -1.0)
    change_ax.set_yticks([])
    change_ax.set_yticklabels([])

    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    change_ax.spines["left"].set_visible(False)
    change_ax.spines["right"].set_visible(False)
    change_ax.spines["top"].set_visible(False)

    ax.axvline(50.0, color="grey", linestyle="--", linewidth=1, zorder=0)
    change_ax.axvline(0.0, color="grey", linestyle="--", linewidth=1, zorder=0)

    for i, (_, row) in enumerate(m_df.iterrows()):
        if row["type"] == "data":
            ax.text(pos_label, i, "    " + row["label"], va="center")
            ax.text(
                pos_nauthors,
                i,
                f"{int(row['n_authors']):,}",
                va="center"
            )
            ax.plot(
                row["est"], i, marker="s", markersize=8, color=BLUE, alpha=0.8
            )
        elif row["type"] == "header_main":
            ax.text(
                pos_nauthors,
                i,
                "Total Number of Authors",
                fontweight="bold",
                va="center"
            )
            ax.text(
                pos_label,
                i,
                row["label"],
                fontsize=10,
                fontweight="bold",
                va="center"
            )
            ax.text(
                46.0,
                i,
                "Under Half",
                fontweight="bold",
                ha="right",
                va="center"
            )
            ax.text(
                54.0,
                i,
                "Over Half",
                fontweight="bold",
                ha="left",
                va="center"
            )
        else:
            raise NotImplementedError

    for i, (_, row) in enumerate(f_df.iterrows()):
        if row["type"] == "data":
            ax.plot(
                row["est"], i, marker="o", markersize=8, color=RED, alpha=0.8
            )
        elif row["type"] != "header_main":
            raise NotImplementedError

    for i, (_, row) in enumerate(m_change_df.iterrows()):
        if row["type"] == "data":
            change_ax.hlines(
                i,
                row["lower"],
                row["upper"],
                color=BLUE,
                linewidth=1.5,
                alpha=0.8
            )
            change_ax.plot(
                row["est"], i, marker="s", markersize=8, color=BLUE, alpha=0.8
            )
            change_ax.text(
                pos_r2, i, f"{row['r2']:.2f}", va="center"
            )
        elif row["type"] == "header_main":
            change_ax.text(
                -0.0025,
                i,
                "Decreasing Over Time",
                fontweight="bold",
                ha="right",
                va="center"
            )
            change_ax.text(
                0.0025,
                i,
                "Increasing Over Time",
                fontweight="bold",
                ha="left",
                va="center"
            )
            change_ax.text(
                pos_r2,
                i,
                r"$R^2$ (Male)",
                fontweight="bold",
                color=BLUE,
                va="center"
            )
            change_ax.text(
                pos_r2 + 0.015,
                i,
                r"$R^2$ (Female)",
                fontweight="bold",
                color=RED,
                va="center"
            )
        else:
            raise NotImplementedError

    for i, (_, row) in enumerate(f_change_df.iterrows()):
        if row["type"] == "data":
            change_ax.hlines(
                i,
                row["lower"],
                row["upper"],
                color=RED,
                linewidth=1.5,
                alpha=0.8
            )
            change_ax.plot(
                row["est"], i, marker="o", markersize=8, color=RED, alpha=0.8
            )
            change_ax.text(
                pos_r2 + 0.015, i, f"{row['r2']:.2f}", va="center"
            )
        elif row["type"] != "header_main":
            raise NotImplementedError

    ax.set_xlim(0.0, 100.0)
    ticks = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(t)) * (1 - (int(t) % 20)) for t in ticks])
    ax.set_xlabel("Percentage of Manuscript Authors (%)", fontsize=10)
    ax.xaxis.set_ticks_position("bottom")
    change_ax.set_xlim(-0.03, 0.03)
    change_ax.set_xlabel(
        "Rate of Change in Authorship Percentage (%/year)", fontsize=10
    )

    plt.subplots_adjust(left=0.3, right=0.85, top=0.85, bottom=0.1)

    if savedir is not None:
        plt.savefig(
            os.path.join(str(savedir), "fig2.pdf"),
            transparent=True,
            dpi=600,
            bbox_inches="tight"
        )
    else:
        plt.show()
    plt.close()


def get_data(
    gender: str, datadir: Union[Path, str] = "Medicine Authorship"
) -> pd.DataFrame:
    data: List[Dict[str, Any]] = []
    for i, category in enumerate([
        "All Authors", "First Author", "Last Author"
    ]):
        data.append({
            "type": "header_main", "label": category + (" Only" * (i > 0))
        })
        sub_datadir = os.path.join(str(datadir), category)
        for bs, fn in BROAD_SUBJECTS.items():
            df = pd.read_parquet(os.path.join(sub_datadir, fn))
            total = len(df)
            y = 100.0 * float((df.gender == gender).sum()) / total
            data.append({
                "type": "data",
                "label": bs,
                "n_authors": total,
                "est": y,
                "lower": y,
                "upper": y
            })
    return pd.DataFrame(data)


def get_change_data(
    gender: str,
    datadir: Union[Path, str] = "Medicine Authorship",
    confidence_level: float = 0.99
) -> pd.DataFrame:
    data: List[Dict[str, Any]] = []
    for i, category in enumerate([
        "All Authors", "First Author", "Last Author"
    ]):
        data.append({
            "type": "header_main", "label": category + (" Only" * (i > 0))
        })
        sub_datadir = os.path.join(str(datadir), category)
        for bs, fn in BROAD_SUBJECTS.items():
            df = pd.read_parquet(os.path.join(sub_datadir, fn))
            years = np.sort(df.year.unique())
            data_by_year = []
            for year in years:
                y_df = df[df.year == year]
                total = len(y_df)
                data_by_year.append(
                    float((y_df.gender == gender).sum()) / total
                )
            ols_model = sm.OLS(
                data_by_year, sm.add_constant(years - years.min())
            )
            ols_model = ols_model.fit()
            t_val = t.ppf(1.0 - ((1.0 - confidence_level) / 2), len(years) - 2)
            data.append({
                "type": "data",
                "label": bs,
                "n_authors": len(df),
                "est": ols_model.params[1],
                "lower": ols_model.params[1] - (t_val * ols_model.bse[1]),
                "upper": ols_model.params[1] + (t_val * ols_model.bse[1]),
                "r2": ols_model.rsquared
            })
    return pd.DataFrame(data)


if __name__ == "__main__":
    main()
