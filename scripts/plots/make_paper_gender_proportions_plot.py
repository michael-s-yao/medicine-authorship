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
import seaborn as sns
import sys
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from scipy.stats import norm
from typing import Final, List, Optional, Tuple, Union

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
from core.plot_utils import BROAD_SUBJECTS, fmt_pval  # noqa


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
@click.option(
    "--seed",
    type=int,
    default=2025,
    show_default=True,
    help="Optional random seed."
)
def main(
    datadir: Union[Path, str],
    savedir: Optional[Union[Path, str]],
    seed: Optional[int],
    show_pvals: bool = False
):
    """Make plots of paper gender proportions."""
    plt.rcParams["font.family"] = ["Arial"]
    BLUE: Final[str] = "#3170AD"
    RED: Final[str] = "#90312C"
    if savedir is not None:
        os.makedirs(os.path.abspath(str(savedir)), exist_ok=True)

    cols: Final[List[str]] = ["frac_female", "frac_male", "num_total_authors"]
    sub_datadir = os.path.join(str(datadir), "Fractional Gender Analysis")
    dfs: List[pd.DataFrame] = []
    pvals: List[float] = []
    for bs, fn in BROAD_SUBJECTS.items():
        df = pd.read_parquet(os.path.join(sub_datadir, fn))[cols]
        _, p = homophily_test(
            (df["frac_female"] * df["num_total_authors"]).astype(int),
            (df["frac_male"] * df["num_total_authors"]).astype(int)
        )
        pvals.append(p)
        df = df.melt(value_vars=cols, value_name="frac").assign(
            gender=lambda d: d["variable"].map({
                "frac_female": "Female", "frac_male": "Male"
            })
        )
        df = df[["frac", "gender"]]
        df["Broad Subject"] = bs
        df = df.rename(
            columns={
                "frac": "Proportion of Author List",
                "gender": "Author Gender"
            }
        )
        dfs.append(df)
    print(max(pvals))

    _, ax = plt.subplots(figsize=(12 + (4 * show_pvals), 6))
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
    ax.set_xlabel("")
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

    ax.set_ylim(0.0, 1.0 + (0.07 * show_pvals))
    if show_pvals:
        pos_p = 1.03
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        for i, p in enumerate(pvals):
            plt.gca().text(
                i,
                pos_p,
                fmt_pval(p),
                ha="center",
                va="center"
            )
        plt.gca().text(
            12.6,
            pos_p + 0.002,
            "p-value",
            fontsize=10,
            fontweight="bold",
            va="center"
        )

    if savedir is not None:
        plt.savefig(
            os.path.join(str(savedir), "fig3.pdf"),
            transparent=True,
            dpi=600,
            bbox_inches="tight"
        )
    else:
        plt.show()
    plt.close()


def homophily_test(
    count_A: Union[np.ndarray, pd.Series, List[float]],
    count_B: Union[np.ndarray, pd.Series, List[float]]
) -> Tuple[float, float]:
    """
    Homophily test for gender clustering within groups.
    Input:
        count_A: the number of members of group A per group.
        count_B: the number of members of group B per group.
    Returns:
        A tuple of the observed same-gender pair fraction and the
        (right-tailed) p-value.
    """
    A, B = np.asarray(count_A, dtype=int), np.asarray(count_B, dtype=int)
    N = A + B

    mask = N >= 2
    A, B, N = A[mask], B[mask], N[mask]
    p = A.sum() / N.sum()
    w = 1.0 / len(N)

    h_obs = ((A * (A - 1)) + (B * (B - 1))) / (N * (N - 1))
    T_obs = np.sum(w * h_obs)
    mean_T = np.square(p) + np.square(1 - p)

    var_h = (2.0 * p * (1 - p) / (N * (N - 1))) * (
        2.0 * (N - 2) * np.square(1.0 - (2.0 * p)) + 1.0
    )
    z_score = (T_obs - mean_T) / np.sqrt(np.sum(np.square(w) * var_h))

    return z_score, norm.sf(z_score)


if __name__ == "__main__":
    main()
