#!/usr/bin/python3
"""
Make plots of paper gender proportions.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2025.
"""
import click
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from pathlib import Path
from typing import Any, Final, List, Optional, Union

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
from core.plot_utils import BROAD_SUBJECTS, radar_factory  # noqa


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
    """Make plots of paper gender proportions."""
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["font.family"] = "Arial"
    colors = ["#DD1717"]
    if savedir is not None:
        os.makedirs(os.path.abspath(str(savedir)), exist_ok=True)

    data = get_data(datadir)
    theta = radar_factory(len(data[0]), frame="polygon")

    spoke_labels = data.pop(0)

    fig, axs = plt.subplots(
        figsize=(10.0, 2.5),
        nrows=1,
        ncols=3,
        subplot_kw={"projection": "radar"}
    )

    for i, (ax, (title, case)) in enumerate(zip(axs.flat, data)):
        ax.set_rticks(
            [10.0, 20.0, 30.0, 40.0, 50.0], labels=["", "20", "", "40", ""]
        )
        ax.set_ylim(0.0, 55.0)
        ax.set_title(
            title + "\n",
            weight="bold",
            size="medium",
            position=(0.5, 1.3),
            horizontalalignment="center",
            verticalalignment="center"
        )
        ax.plot(
            theta,
            (100.0 / 3.0) * np.ones_like(theta),
            color="k",
            linestyle=":",
            alpha=0.75
        )
        np_case = 100.0 * np.asarray(case)
        for d, color in zip(np_case, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label="_nolegend_")
        ax.set_varlabels(spoke_labels)
        ax.annotate(
            f"({chr(ord('a') + i)})",
            xy=(-0.1, 1.2),
            xycoords="axes fraction",
            fontweight="bold"
        )

    if savedir is not None:
        plt.savefig(
            os.path.join(str(savedir), "author_lists.pdf"),
            transparent=True,
            dpi=600,
            bbox_inches="tight"
        )
    else:
        plt.show()
    plt.close()


def get_data(datadir: Union[Path, str] = "Medicine Authorship") -> List[Any]:
    data: List[Any] = [list(BROAD_SUBJECTS.keys())]
    thresh: Final[List[float]] = [-1.0 * np.inf, 1.0 / 3.0, 2.0 / 3.0, np.inf]

    sub_datadir = os.path.join(str(datadir), "Fractional Gender Analysis")

    for i, category in enumerate(["Low", "Medium", "High"]):
        female_data = []
        for bs in data[0]:
            df = pd.read_parquet(os.path.join(sub_datadir, BROAD_SUBJECTS[bs]))
            total = len(df)
            f_mask = np.logical_and(
                df.freq_female < thresh[i + 1], df.freq_female >= thresh[i]
            )
            female_data.append(float(f_mask.sum()) / total)
        data.append((category, [female_data]))

    return data


if __name__ == "__main__":
    main()
