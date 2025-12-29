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
    """Make gender frequency radar plots."""
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

    for ax, (title, case_mu) in zip(axs.flat, data):
        ax.set_rticks(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            labels=["", "20", "", "40", "", "60"]
        )
        ax.set_ylim(0.0, 70.0)
        ax.set_title(
            title + "\n",
            weight="bold",
            size="medium",
            position=(0.5, 1.3),
            horizontalalignment="center",
            verticalalignment="center"
        )
        case_mu = 100.0 * np.asarray(case_mu)
        ax.plot(
            theta,
            50.0 * np.ones_like(theta),
            color="k",
            linestyle=":",
            alpha=0.75
        )
        for d, color in zip(case_mu, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label="_nolegend_")
        ax.set_varlabels(spoke_labels)

    if savedir is not None:
        plt.savefig(
            os.path.join(str(savedir), "frac.pdf"),
            transparent=True,
            dpi=600,
            bbox_inches="tight"
        )
    else:
        plt.show()
    plt.close()

    mu_data, sem_data = get_change_data(datadir)
    theta = radar_factory(len(mu_data[0]), frame="polygon")

    mu_data.pop(0), sem_data.pop(0)

    fig, axs = plt.subplots(
        figsize=(10.0, 2.5),
        nrows=1,
        ncols=3,
        subplot_kw={"projection": "radar"}
    )

    for ax, (title, case_mu), (_, case_sem) in zip(
        axs.flat, mu_data, sem_data
    ):
        ax.set_rticks(
            [-2.0, -1.0, 0.0, 1.0, 2.0], labels=["", "-1", "0", "+1", ""]
        )
        ax.set_ylim(-2.0, 2.0)
        ax.set_title(
            title + "\n",
            weight="bold",
            size="medium",
            position=(0.5, 1.3),
            horizontalalignment="center",
            verticalalignment="center"
        )
        case_mu = 100.0 * np.asarray(case_mu)
        case_sem = 100.0 * np.asarray(case_sem)
        ax.plot(theta, 0.0 * theta, color="k", linestyle=":", alpha=0.75)
        for d, e, color in zip(case_mu, case_sem, colors):
            ax.plot(theta, d, color=color)
            ax.errorbar(theta, d, yerr=e, color=color, label="_nolegend_")
            ax.fill(theta, d, facecolor=color, alpha=0.25, label="_nolegend_")
        ax.set_varlabels(spoke_labels)

    if savedir is not None:
        plt.savefig(
            os.path.join(str(savedir), "frac_change.pdf"),
            transparent=True,
            dpi=600,
            bbox_inches="tight"
        )
    else:
        plt.show()
    plt.close()


def get_data(datadir: Union[Path, str] = "Medicine Authorship") -> List[Any]:
    data: List[Any] = [list(BROAD_SUBJECTS.keys())]

    for category in ["All Authors", "First Author", "Last Author"]:
        female_data = []
        sub_datadir = os.path.join(str(datadir), category)
        for bs in data[0]:
            df = pd.read_parquet(os.path.join(sub_datadir, BROAD_SUBJECTS[bs]))
            total = len(df)
            female_data.append(float((df.gender == "female").sum()) / total)
            male_frac = float((df.gender == "male").sum()) / total
            print(
                f"{bs} {category}: Female Gender Percentage = "
                f"{100.0 * female_data[-1]:.6f}%"
            )
            print(
                f"{bs} {category}: Male Gender Percentage = "
                f"{100.0 * male_frac:.6f}%"
            )
        data.append((category, [female_data]))
    return data


def get_change_data(
    datadir: Union[Path, str] = "Medicine Authorship"
) -> Tuple[List[Any], List[Any]]:
    mu: List[Any] = [list(BROAD_SUBJECTS.keys())]
    sem: List[Any] = [list(BROAD_SUBJECTS.keys())]

    for category in ["All Authors", "First Author", "Last Author"]:
        female_mu, female_sem = [], []
        sub_datadir = os.path.join(str(datadir), category)
        for bs in mu[0]:
            df = pd.read_parquet(os.path.join(sub_datadir, BROAD_SUBJECTS[bs]))
            f_y_data, m_y_data = [], []
            years = np.sort(df.year.unique())
            for year in years:
                y_df = df[df.year == year]
                total = len(y_df)
                f_y_data.append(float((y_df.gender == "female").sum()) / total)
                m_y_data.append(float((y_df.gender == "male").sum()) / total)
            f_model = sm.OLS(f_y_data, sm.add_constant(years - years.min()))
            f_model = f_model.fit()
            female_mu.append(f_model.params[1])
            female_sem.append(f_model.bse[1])
            print(
                f"{bs} {category}: Female Slope = "
                f"{female_mu[-1]} +/- {female_sem[-1]}\n"
                f"{bs} {category}: Female R2 = {f_model.rsquared}"
            )

            m_model = sm.OLS(m_y_data, sm.add_constant(years - years.min()))
            m_model = m_model.fit()
            print(
                f"{bs} {category}: Male Slope = "
                f"{m_model.params[1]} +/- {m_model.bse[1]}"
            )
        mu.append((category, [female_mu]))
        sem.append((category, [female_sem]))
    return mu, sem


if __name__ == "__main__":
    main()
