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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    """Make plots for paper citation data."""
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["font.family"] = "Arial"
    colors = ["#0F4392", "#DD1717"]
    linestyles = ["-.", ":"]
    if savedir is not None:
        os.makedirs(os.path.abspath(str(savedir)), exist_ok=True)

    data = get_data(datadir)
    for key, (mu, sem) in data.items():
        theta = radar_factory(len(mu[0]), frame="polygon")
        spoke_labels = mu.pop(0)
        sem.pop(0)
        data[key] = (mu, sem)

    fig, axs = plt.subplots(
        figsize=(13.5, 6.0),
        nrows=2,
        ncols=4,
        subplot_kw={"projection": "radar"}
    )
    plt.subplots_adjust(wspace=0.1, hspace=0.4)

    for i, (metric, case) in enumerate(data.items()):
        for j, ((title, case_mu), (title_sem, case_sem)) in enumerate(
            zip(*case)
        ):
            assert title == title_sem
            metric = " ".join(
                map(str.title, metric.replace("num_", "").split("_"))
            )
            axs[j, i].set_ylim(0.0, [35, 1.8, 54, 2.7][i])
            if j == 0:
                axs[j, i].set_title(
                    metric + "\n",
                    weight="bold",
                    size="medium",
                    horizontalalignment="center",
                    verticalalignment="center"
                )
            if i == 0:
                axs[j, i].set_ylabel(
                    title + "\n\n\n",
                    weight="bold",
                    size="medium",
                    position=(-1.5, 0.5),
                    horizontalalignment="center",
                    verticalalignment="bottom"
                )
            for d, e, color, lsty in zip(
                case_mu, case_sem, colors, linestyles
            ):
                axs[j, i].plot(theta, d, color=color, linestyle=lsty)
                axs[j, i].errorbar(
                    theta,
                    d,
                    yerr=e,
                    color=color,
                    linestyle=lsty,
                    label="_nolegend_"
                )
                axs[j, i].fill(
                    theta, d, facecolor=color, alpha=0.25, label="_nolegend_"
                )
            axs[j, i].set_varlabels(spoke_labels)

    labels = ("Male", "Female")
    axs[-1, -1].legend(
        labels, loc="right", labelspacing=0.2, bbox_to_anchor=(1.4, 1.2)
    )

    if savedir is not None:
        plt.savefig(
            os.path.join(str(savedir), "citations_and_references.pdf"),
            transparent=True,
            dpi=600
        )
    else:
        plt.show()
    plt.close()


def get_data(
    datadir: Union[Path, str] = "Medicine Authorship"
) -> Dict[str, Tuple[List[Any], List[Any]]]:
    out: Dict[str, Tuple[List[Any], List[Any]]] = {
        "num_total_citations": (
            [list(BROAD_SUBJECTS.keys())], [list(BROAD_SUBJECTS.keys())]
        ),
        "num_self_citations": (
            [list(BROAD_SUBJECTS.keys())], [list(BROAD_SUBJECTS.keys())]
        ),
        "num_total_references": (
            [list(BROAD_SUBJECTS.keys())], [list(BROAD_SUBJECTS.keys())]
        ),
        "num_self_references": (
            [list(BROAD_SUBJECTS.keys())], [list(BROAD_SUBJECTS.keys())]
        )
    }

    for category in ["First Author", "Last Author"]:
        male_mu, female_mu = defaultdict(list), defaultdict(list)
        male_sem, female_sem = defaultdict(list), defaultdict(list)
        sub_datadir = os.path.join(str(datadir), category)
        for bs in out["num_total_citations"][0][0]:
            df = pd.read_parquet(os.path.join(sub_datadir, BROAD_SUBJECTS[bs]))
            m_df, f_df = df[df.gender == "male"], df[df.gender == "female"]
            for key in out.keys():
                male_mu[key].append(m_df[key].mean())
                male_sem[key].append(
                    m_df[key].std(ddof=1) / np.sqrt(len(m_df[key]))
                )
                female_mu[key].append(f_df[key].mean())
                female_sem[key].append(
                    f_df[key].std(ddof=1) / np.sqrt(len(f_df[key]))
                )
        for key in out.keys():
            out[key][0].append((category, [male_mu[key], female_mu[key]]))
            out[key][1].append((category, [male_sem[key], female_sem[key]]))
    return out


if __name__ == "__main__":
    main()
