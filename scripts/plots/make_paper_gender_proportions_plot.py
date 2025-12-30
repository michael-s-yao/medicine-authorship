#!/usr/bin/python3
"""
Make plots of paper gender proportions.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2025.
"""
import click
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from typing import Final, List, Optional, Union

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
    """Make plots of paper gender proportions."""
    plt.rcParams["font.family"] = ["Arial"]
    BLUE: Final[str] = "#3170AD"
    RED: Final[str] = "#90312C"
    if savedir is not None:
        os.makedirs(os.path.abspath(str(savedir)), exist_ok=True)

    cols: Final[List[str]] = ["fraq_female", "fraq_male"]
    sub_datadir = os.path.join(str(datadir), "Fractional Gender Analysis")
    dfs: List[pd.DataFrame] = []
    for bs, fn in BROAD_SUBJECTS.items():
        df = pd.read_parquet(os.path.join(sub_datadir, fn))[cols]
        df = df.melt(value_vars=cols, value_name="fraq").assign(
            gender=lambda d: d["variable"].map({
                "fraq_female": "Female", "fraq_male": "Male"
            })
        )
        df = df[["fraq", "gender"]]
        df["Broad Subject"] = bs
        df = df.rename(
            columns={
                "fraq": "Proportion of Author List",
                "gender": "Author Gender"
            }
        )
        dfs.append(df)

    _, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        data=pd.concat(dfs, ignore_index=True),
        x="Broad Subject",
        y="Proportion of Author List",
        hue="Author Gender",
        split=True,
        gap=0.1,
        cut=0,
        inner="quart",
        palette=[RED, BLUE],
        alpha=0.8,
        ax=ax
    )
    ax.set_xlabel(ax.get_xlabel(), fontdict={"weight": "bold"})
    ax.set_ylabel(ax.get_ylabel(), fontdict={"weight": "bold"})
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.tick_params(axis="x", labelrotation=30)
    for x in ax.get_xticklabels():
        x.set_horizontalalignment("right")
    sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
    for ell in ax.lines:
        ell.set_linestyle(":")
        ell.set_linewidth(1.5)
        ell.set_color("black")
        ell.set_alpha(0.8)
    for ell in ax.lines[1::3]:
        ell.set_linestyle("-")

    if savedir is not None:
        plt.savefig(
            os.path.join(str(savedir), "fig4.pdf"),
            transparent=True,
            dpi=600,
            bbox_inches="tight"
        )
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
