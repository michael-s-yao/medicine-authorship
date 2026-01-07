#!/usr/bin/python3
"""
Make plots for paper citation data.

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
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
from scipy.stats import t, ttest_ind
from typing import Dict, Final, List, Optional, Tuple, Union

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
def main(
    datadir: Union[Path, str],
    savedir: Optional[Union[Path, str]],
    include_legend: bool = False,
    include_self_references: bool = False
):
    """Make plots for paper citation and reference data."""
    plt.rcParams["font.family"] = ["Arial"]
    strats: Final[List[str]] = ["First Author", "Last Author"]
    metric2title: Dict[str, str] = {
        "total_citations": "Total Citations",
        "self_citations": "Self Citations",
        "total_references": "Total References",
    }
    if include_self_references:
        metric2title["self_references"] = "Self References"
    fig, axes = plt.subplots(
        len(strats), len(metric2title.keys()), figsize=(16, 12)
    )
    axes = axes.flatten()
    for i, strat in enumerate(strats):
        data = get_data(strat, datadir=datadir)
        for j, (metric, title) in enumerate(metric2title.items()):
            cleveland_dotplot(
                axes[(len(metric2title.keys()) * i) + j], metric, title, *data
            )

    for i in range(len(axes)):
        if i % len(metric2title.keys()):
            axes[i].set_yticks([])
        else:
            axes[i].tick_params(axis="y", length=0, pad=90)
            axes[i].text(
                -0.65 + (0.18 * (not include_self_references)),
                1.03,
                strats[int(i / len(metric2title.keys()))] + " Gender",
                fontweight="bold",
                transform=axes[i].transAxes,
                fontsize=11,
                va="center"
            )
            for label in axes[i].get_yticklabels():
                label.set_horizontalalignment("left")
            for j in range(len(BROAD_SUBJECTS)):
                if (j % 2):
                    continue
                axes[i].axhspan(
                    j - 0.5,
                    j + 0.5,
                    xmin=-0.68 + (0.18 * (not include_self_references)),
                    xmax=(6.9 - (2.55 * (not include_self_references))),
                    color="gray",
                    alpha=0.1,
                    ec=None,
                    clip_on=False
                )

    handles, labels = axes[0].get_legend_handles_labels()
    if include_legend:
        axes[-1].legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.55, 1.1),
            fontsize=11
        )
    plt.subplots_adjust(wspace=0.8 - (0.3 * (not include_self_references)))

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


def get_data(
    category: str,
    datadir: Union[Path, str] = "Medicine Authorship",
    confidence_level: float = 0.99
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert category in ["First Author", "Last Author"]
    data = []
    pvals = []
    sub_datadir = os.path.join(str(datadir), category)
    for bs in BROAD_SUBJECTS.keys():
        df = pd.read_parquet(os.path.join(sub_datadir, BROAD_SUBJECTS[bs]))

        stats: Dict[str, List[pd.Series]] = {
            "total_citations": [],
            "self_citations": [],
            "total_references": [],
            "self_references": []
        }
        for gender in ["male", "female"]:
            g_df = df[df.gender == gender]
            total_citations = g_df.num_total_citations
            tot_cits_mu = total_citations.mean()
            tot_cits_sem = total_citations.std(ddof=1) / np.sqrt(len(g_df))
            self_citations = g_df.num_self_citations
            self_cits_mu = self_citations.mean()
            self_cits_sem = self_citations.std(ddof=1) / np.sqrt(len(g_df))
            total_refs = g_df.num_total_references
            tot_refs_mu = total_refs.mean()
            tot_refs_sem = total_refs.std(ddof=1) / np.sqrt(len(g_df))
            self_refs = g_df.num_self_references
            self_refs_mu = self_refs.mean()
            self_refs_sem = self_refs.std(ddof=1) / np.sqrt(len(g_df))
            vt = t.ppf(1.0 - ((1.0 - confidence_level) / 2), len(g_df) - 1)

            stats["total_citations"].append(total_citations)
            stats["self_citations"].append(self_citations)
            stats["total_references"].append(total_refs)
            stats["self_references"].append(self_refs)

            data.append({
                "subject": bs,
                "gender": gender.title(),
                "total_citations_mean": tot_cits_mu,
                "total_citations_lower": tot_cits_mu - (vt * tot_cits_sem),
                "total_citations_upper": tot_cits_mu + (vt * tot_cits_sem),
                "self_citations_mean": self_cits_mu,
                "self_citations_lower": self_cits_mu - (vt * self_cits_sem),
                "self_citations_upper": self_cits_mu + (vt * self_cits_sem),
                "total_references_mean": tot_refs_mu,
                "total_references_lower": tot_refs_mu - (vt * tot_refs_sem),
                "total_references_upper": tot_refs_mu + (vt * tot_refs_sem),
                "self_references_mean": self_refs_mu,
                "self_references_lower": self_refs_mu - (vt * self_refs_sem),
                "self_references_upper": self_refs_mu + (vt * self_refs_sem)
            })

        pvals.append({
            "subject": bs,
            "total_citations_p": ttest_ind(
                stats["total_citations"][0], stats["total_citations"][1]
            ).pvalue,
            "self_citations_p": ttest_ind(
                stats["self_citations"][0], stats["self_citations"][1]
            ).pvalue,
            "total_references_p": ttest_ind(
                stats["total_references"][0], stats["total_references"][1]
            ).pvalue,
            "self_references_p": ttest_ind(
                stats["self_references"][0], stats["self_references"][1]
            ).pvalue,
        })

    return pd.DataFrame(data), pd.DataFrame(pvals)


def cleveland_dotplot(
    ax: Axes, metric: str, title: str, data: pd.DataFrame, pvals: pd.DataFrame
) -> None:
    BLUE: Final[str] = "#3170AD"
    RED: Final[str] = "#90312C"

    subjects = list(BROAD_SUBJECTS.keys())

    for gender in ["Male", "Female"]:
        subset = data[data["gender"] == gender]
        ax.errorbar(
            subset[f"{metric}_mean"],
            subset["subject"],
            xerr=(subset[f"{metric}_upper"] - subset[f"{metric}_mean"]),
            fmt=("o" if gender == "Female" else "s"),
            color=(RED if gender == "Female" else BLUE),
            label=gender,
            capsize=3,
            markersize=8,
            linewidth=1.5,
            alpha=0.8
        )

    ax.text(
        1.05,
        1.03,
        "p-value",
        fontweight="bold",
        transform=ax.transAxes,
        fontsize=11,
        va="center"
    )
    for i, subject in enumerate(subjects):
        ax.text(
            1.05,
            0.96 - (i / len(subjects)),
            fmt_pval(
                float(
                    pvals[pvals.subject == subject][f"{metric}_p"].item()
                )
            ),
            transform=ax.transAxes,
            va="center"
        )

    ax.set_xlabel(f"Number of {title}", fontweight="bold", fontsize=11)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(which="minor", length=3, color="gray", axis="x")
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("")
    ax.set_ylim(-0.5, len(subjects) - 0.5)
    ax.invert_yaxis()
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


if __name__ == "__main__":
    main()
