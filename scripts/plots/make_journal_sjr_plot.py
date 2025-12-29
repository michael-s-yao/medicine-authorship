#!/usr/bin/python3
"""
Make journal SJR analysis plots.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2025.
"""
import click
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm  # type: ignore
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

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
    """Make journal SJR analysis plots."""
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["font.family"] = "Arial"
    colors = ["#0F4392", "#DD1717"]
    linestyles = ["-.", ":"]
    if savedir is not None:
        os.makedirs(os.path.abspath(str(savedir)), exist_ok=True)

    mu_data, sem_data = get_sjr_data(datadir)
    theta = radar_factory(len(mu_data[0]), frame="polygon")

    spoke_labels, _ = mu_data.pop(0), sem_data.pop(0)

    fig, axs = plt.subplots(
        figsize=(10.0, 2.5),
        nrows=1,
        ncols=3,
        subplot_kw={"projection": "radar"}
    )

    for ax, (title, case_mu), (_, case_sem) in zip(
        axs.flat, mu_data, sem_data
    ):
        ax.set_rticks([1.0, 2.0, 3.0], labels=["1", "2", "3"])
        ax.set_ylim(0.0, 4.0)
        ax.set_title(
            title + "\n",
            weight="bold",
            size="medium",
            position=(0.5, 1.3),
            horizontalalignment="center",
            verticalalignment="center"
        )
        ax.plot(theta, 0.0 * theta, color="k", linestyle=":", alpha=0.75)
        for d, e, color, lsty in zip(case_mu, case_sem, colors, linestyles):
            ax.plot(theta, d, color=color, linestyle=lsty)
            ax.errorbar(theta, d, yerr=e, color=color, label="_nolegend_")
            ax.fill(theta, d, facecolor=color, alpha=0.25, label="_nolegend_")
        ax.set_varlabels(spoke_labels)

    if savedir is not None:
        plt.savefig(
            os.path.join(str(savedir), "journal_sjr.pdf"),
            transparent=True,
            dpi=600,
            bbox_inches="tight"
        )
    else:
        plt.show()
    plt.close()


def get_sjr_data(
    datadir: Union[Path, str] = "Medicine Authorship"
) -> Tuple[List[Any], List[Any]]:
    mu: List[Any] = [list(BROAD_SUBJECTS.keys())]
    sem: List[Any] = [list(BROAD_SUBJECTS.keys())]

    for category in ["First Author", "Last Author"]:
        female_mu: List[float] = []
        female_sem: List[float] = []
        male_mu: List[float] = []
        male_sem: List[float] = []
        sub_datadir = os.path.join(str(datadir), category)
        for bs in mu[0]:
            df = pd.read_parquet(os.path.join(sub_datadir, BROAD_SUBJECTS[bs]))

            m_df = df[df.gender == "male"]
            np_m_sjr = m_df.journal_sjr.to_numpy().astype(float)
            np_m_sjr = np_m_sjr[~np.isnan(np_m_sjr)]
            male_mu.append(np_m_sjr.mean())
            male_sem.append(np.std(np_m_sjr, ddof=1) / np.sqrt(np_m_sjr.size))

            f_df = df[df.gender == "female"]
            np_f_sjr = f_df.journal_sjr.to_numpy().astype(float)
            np_f_sjr = np_f_sjr[~np.isnan(np_f_sjr)]
            female_mu.append(np_f_sjr.mean())
            female_sem.append(
                np.std(np_f_sjr, ddof=1) / np.sqrt(np_f_sjr.size)
            )
            stat_test = stats.ttest_ind(np_m_sjr, np_f_sjr, equal_var=False)
            print(
                f"{bs} {category}: Male SJR = {male_mu[-1]} +/- {male_sem[-1]}"
                f" | Female SJR = {female_mu[-1]} +/- {female_sem[-1]}"
                f" | Unpaired t-test: {stat_test}"
            )

        mu.append((category, [male_mu, female_mu]))
        sem.append((category, [male_sem, female_sem]))
    return mu, sem


def get_sjr_citation_data(
    datadir: Union[Path, str] = "Medicine Authorship"
) -> Tuple[List[Any], List[Any]]:
    mu: List[Any] = [list(BROAD_SUBJECTS.keys())]
    sem: List[Any] = [list(BROAD_SUBJECTS.keys())]
    eps = float(np.finfo(np.float32).eps)

    for category in ["First Author", "Last Author"]:
        female_mu: List[float] = []
        female_sem: List[float] = []
        male_mu: List[float] = []
        male_sem: List[float] = []
        sub_datadir = os.path.join(str(datadir), category)
        for bs in mu[0]:
            df = pd.read_parquet(os.path.join(sub_datadir, BROAD_SUBJECTS[bs]))

            m_df = df[df.gender == "male"]
            np_m_sjr = m_df.journal_sjr.to_numpy().astype(float)
            np_m_sjr = np_m_sjr[~np.isnan(np_m_sjr)]
            m_sjrs, m_citations_mu, m_citations_sem = [], [], []
            for sjr in np.unique(np_m_sjr):
                m_data = m_df[m_df.journal_sjr == sjr].num_total_citations
                np_m_data = m_data.to_numpy()
                np_m_data = np_m_data[np_m_data >= 0]
                if np_m_data.size < 2:
                    continue
                m_sjrs.append(sjr)
                m_citations_mu.append(np_m_data.mean())
                m_citations_sem.append(
                    np.std(np_m_data, ddof=1) / np.sqrt(np_m_data.size)
                )
            m_weights = 1.0 / (np.square(np.asarray(m_citations_sem)) + eps)
            m_model = sm.WLS(
                m_citations_mu, sm.add_constant(m_sjrs), weights=m_weights
            )
            m_model = m_model.fit()
            male_mu.append(m_model.params[1])
            male_sem.append(m_model.bse[1])

            f_df = df[df.gender == "female"]
            np_f_sjr = f_df.journal_sjr.to_numpy().astype(float)
            np_f_sjr = np_f_sjr[~np.isnan(np_f_sjr)]
            f_sjrs, f_citations_mu, f_citations_sem = [], [], []
            for sjr in np.unique(np_f_sjr):
                f_data = f_df[f_df.journal_sjr == sjr].num_total_citations
                np_f_data = f_data.to_numpy()
                np_f_data = np_f_data[np_f_data >= 0]
                if np_f_data.size < 2:
                    continue
                f_sjrs.append(sjr)
                f_citations_mu.append(np_f_data.mean())
                f_citations_sem.append(
                    np.std(np_f_data, ddof=1) / np.sqrt(np_f_data.size)
                )
            f_weights = 1.0 / (np.square(np.asarray(f_citations_sem)) + eps)
            f_model = sm.WLS(
                f_citations_mu, sm.add_constant(f_sjrs), weights=f_weights
            )
            f_model = f_model.fit()
            female_mu.append(f_model.params[1])
            female_sem.append(f_model.bse[1])

            print(
                f"{bs} {category}: Female R2 = {f_model.rsquared} | "
                f"Male R2 = {m_model.rsquared}"
            )
        mu.append((category, [male_mu, female_mu]))
        sem.append((category, [male_sem, female_sem]))
    return mu, sem


if __name__ == "__main__":
    main()
