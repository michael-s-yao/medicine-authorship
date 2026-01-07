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
import statsmodels.formula.api as smf  # type: ignore
import sys
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Union

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
    confidence_level: float = 0.99
):
    """Make journal SJR analysis plots."""
    plt.rcParams["font.family"] = ["Arial"]
    BLUE: Final[str] = "#3170AD"
    RED: Final[str] = "#90312C"

    oa_df = get_open_access_data(datadir)
    sjr_df = get_sjr_data(datadir)

    fig, (oa_ax, change_oa_ax, ax) = plt.subplots(1, 3, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.8)

    oa_ax.set_xlim(0.0, 100.0)
    oa_ax.set_ylim(len(sjr_df) + 0.5, -1.0)
    change_oa_ax.set_xlim(-0.2, 0.2)
    change_oa_ax.set_ylim(len(sjr_df) + 0.5, -1.0)
    ax.set_ylim(len(sjr_df) + 0.5, -1.0)
    oa_ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    change_oa_ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    for i in range(len(sjr_df)):
        if (i % 2):
            continue
        oa_ax.axhspan(
            i - 0.55,
            i + 0.45,
            xmin=-0.8,
            xmax=5.11,
            color="gray",
            alpha=0.1,
            ec=None,
            clip_on=False
        )
    oa_ax.set_yticks([])
    oa_ax.set_yticklabels([])
    change_oa_ax.set_yticks([])
    change_oa_ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])

    oa_ax.spines["left"].set_visible(False)
    oa_ax.spines["right"].set_visible(False)
    oa_ax.spines["top"].set_visible(False)
    change_oa_ax.spines["left"].set_visible(False)
    change_oa_ax.spines["right"].set_visible(False)
    change_oa_ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    change_oa_ax.axvline(
        0.0, color="grey", linestyle="--", linewidth=1, zorder=0
    )

    pos_label: Final[float] = -75.0
    pos_oa_p: Final[float] = 105.0
    pos_change_oa_p: Final[float] = 0.215
    pos_p: Final[float] = 3.5

    zsc = stats.norm.ppf((1.0 + confidence_level) / 2.0)
    for i, ((_, oa_row), (_, row)) in enumerate(
        zip(oa_df.iterrows(), sjr_df.iterrows())
    ):
        if row["type"] == "data":
            oa_ax.text(pos_label, i, "    " + row["label"], va="center")
            oa_ax.plot(
                oa_row["pF"], i, marker="o", color=RED, markersize=8, alpha=0.8
            )
            oa_ax.plot(
                oa_row["pM"],
                i,
                marker="s",
                color=BLUE,
                markersize=8,
                alpha=0.8
            )
            oa_ax.text(
                pos_oa_p, i, f"{fmt_pval(oa_row['MminusF_p'])}", va="center"
            )
            change_oa_ax.errorbar(
                oa_row["Fyear_mean"],
                i,
                xerr=(zsc * oa_row["Fyear_bse"]),
                fmt="o",
                color=RED,
                label="Female",
                capsize=3,
                markersize=8,
                linewidth=1.5,
                alpha=0.8
            )
            change_oa_ax.errorbar(
                oa_row["Myear_mean"],
                i,
                xerr=(zsc * oa_row["Myear_bse"]),
                fmt="s",
                color=BLUE,
                label="Male",
                capsize=3,
                markersize=8,
                linewidth=1.5,
                alpha=0.8
            )
            change_oa_ax.text(
                pos_change_oa_p,
                i,
                f"{fmt_pval(oa_row['interaction_p'])}",
                va="center"
            )
            ax.errorbar(
                row["f_est"],
                i,
                xerr=(row["f_upper"] - row["f_est"]),
                fmt="o",
                color=RED,
                label="Female",
                capsize=3,
                markersize=8,
                linewidth=1.5,
                alpha=0.8
            )
            ax.errorbar(
                row["m_est"],
                i,
                xerr=(row["m_upper"] - row["m_est"]),
                fmt="s",
                color=BLUE,
                label="Male",
                capsize=3,
                markersize=8,
                linewidth=1.5,
                alpha=0.8
            )
            ax.text(
                pos_p, i, f"{fmt_pval(row['journal_sjr_p'])}", va="center"
            )
        elif row["type"] == "header_main":
            change_oa_ax.text(
                -0.02,
                i,
                "Decreasing",
                fontweight="bold",
                ha="right",
                va="center"
            )
            change_oa_ax.text(
                0.02,
                i,
                "Increasing",
                fontweight="bold",
                ha="left",
                va="center"
            )
            change_oa_ax.text(
                pos_change_oa_p,
                i,
                "p-value",
                fontsize=10,
                fontweight="bold",
                va="center"
            )
            oa_ax.text(
                pos_label,
                i,
                row["label"],
                fontsize=10,
                fontweight="bold",
                va="center"
            )
            oa_ax.text(
                pos_oa_p,
                i,
                "p-value",
                fontsize=10,
                fontweight="bold",
                va="center"
            )
            ax.text(
                pos_p,
                i,
                "p-value",
                fontsize=10,
                fontweight="bold",
                va="center"
            )
        else:
            raise NotImplementedError

    oa_ax.set_xlabel(
        "% Manuscripts Published in OA Journals",
        fontsize=10,
        fontweight="bold"
    )
    change_oa_ax.set_xlabel(
        "Change in OA Log Odds Per Year",
        fontsize=10,
        fontweight="bold"
    )
    ax.set_xlabel(
        "SCImago Journal Rank (SJR)",
        fontsize=10,
        fontweight="bold"
    )

    if savedir is not None:
        plt.savefig(
            os.path.join(str(savedir), "fig5.pdf"),
            transparent=True,
            dpi=600,
            bbox_inches="tight"
        )
    else:
        plt.show()
    plt.close()


def get_open_access_data(
    datadir: Union[Path, str] = "Medicine Authorship"
) -> pd.DataFrame:
    data = []
    for strat in ["First Author", "Last Author"]:
        data.append({"type": "header_main", "label": strat + " Gender"})
        raw_data = get_raw_open_access_data(strat, datadir=datadir)
        for bs in BROAD_SUBJECTS.keys():
            subset = raw_data[raw_data["label"] == bs]
            model = smf.glm(
                formula="journal_is_open_access ~ year * gender",
                data=subset,
                family=sm.families.Binomial()
            )
            model = model.fit()

            mu_M = model.params["year"] + model.params["year:gender[T.male]"]
            cov = model.cov_params()
            se_M = np.sqrt(
                cov.loc["year", "year"] + (
                    cov.loc["year:gender[T.male]", "year:gender[T.male]"] + (
                        2 * cov.loc["year", "year:gender[T.male]"]
                    )
                )
            )
            z_M = mu_M / se_M

            margins = model.get_margeff(
                at="overall", method="dydx", dummy=True, atexog=None
            )
            mfx_summary = margins.summary_frame()

            data.append({
                "type": "data",
                "label": bs,
                "strat": strat,
                "pF": 100.0 * np.mean(
                    subset.loc[
                        subset.gender == "female", "journal_is_open_access"
                    ]
                ),
                "pM": 100.0 * np.mean(
                    subset.loc[
                        subset.gender == "male", "journal_is_open_access"
                    ]
                ),
                "MminusF_mean": mfx_summary.loc["gender[T.male]", "dy/dx"],
                "MminusF_bse": mfx_summary.loc["gender[T.male]", "Std. Err."],
                "MminusF_z": mfx_summary.loc["gender[T.male]", "z"],
                "MminusF_p": mfx_summary.loc["gender[T.male]", "Pr(>|z|)"],
                "Fyear_mean": model.params["year"],
                "Fyear_bse": model.bse["year"],
                "Fyear_z": model.params["year"] / model.bse["year"],
                "Fyear_p": model.pvalues["year"],
                "Myear_mean": mu_M,
                "Myear_bse": se_M,
                "Myear_z": z_M,
                "Myear_p": 2.0 * (1.0 - stats.norm.cdf(abs(z_M))),
                "interaction_p": model.pvalues["year:gender[T.male]"]
            })
    return pd.DataFrame(data)


def get_raw_open_access_data(
    category: str, datadir: Union[Path, str] = "Medicine Authorship"
) -> pd.DataFrame:
    sub_datadir = os.path.join(str(datadir), category)
    cols: Final[List[str]] = ["gender", "year", "journal_is_open_access"]
    dfs = []
    for bs, fn in BROAD_SUBJECTS.items():
        df = pd.read_parquet(os.path.join(sub_datadir, fn))
        df = df[cols]
        df = df[df.gender.isin(["male", "female"])]
        df["journal_is_open_access"] = df["journal_is_open_access"] == "Yes"
        df["journal_is_open_access"] = df["journal_is_open_access"].astype(int)
        df["year"] = df["year"] - df["year"].mean()
        df["label"] = bs
        dfs.append(df)
    return pd.concat(dfs)


def get_sjr_data(
    datadir: Union[Path, str] = "Medicine Authorship",
    confidence_level: float = 0.99
) -> pd.DataFrame:
    data: List[Dict[str, Any]] = []
    for strat in ["First Author", "Last Author"]:
        data.append({"type": "header_main", "label": strat + " Gender"})
        raw_data = get_raw_sjr_data(strat, datadir=datadir)
        for bs in BROAD_SUBJECTS.keys():
            subset = raw_data[raw_data["label"] == bs]
            m_y = subset[subset["gender"] == "male"].journal_sjr
            f_y = subset[subset["gender"] == "female"].journal_sjr
            m_vt = stats.t.ppf(
                1.0 - ((1.0 - confidence_level) / 2), len(m_y) - 1
            )
            f_vt = stats.t.ppf(
                1.0 - ((1.0 - confidence_level) / 2), len(f_y) - 1
            )
            m_sem = m_y.std(ddof=1) / np.sqrt(len(m_y))
            f_sem = f_y.std(ddof=1) / np.sqrt(len(f_y))
            data.append({
                "type": "data",
                "label": bs,
                "strat": strat,
                "m_est": m_y.mean(),
                "m_lower": m_y.mean() - (m_vt * m_sem),
                "m_upper": m_y.mean() + (m_vt * m_sem),
                "f_est": f_y.mean(),
                "f_lower": f_y.mean() - (f_vt * f_sem),
                "f_upper": f_y.mean() + (f_vt * f_sem),
                "journal_sjr_p": stats.ttest_ind(m_y, f_y).pvalue
            })
    return pd.DataFrame(data)


def get_raw_sjr_data(
    category: str, datadir: Union[Path, str] = "Medicine Authorship"
) -> pd.DataFrame:
    sub_datadir = os.path.join(str(datadir), category)
    cols: Final[List[str]] = ["gender", "year", "journal_sjr"]
    dfs = []
    for bs, fn in BROAD_SUBJECTS.items():
        df = pd.read_parquet(os.path.join(sub_datadir, fn))
        df = df[cols]
        df = df[df.gender.isin(["male", "female"])]
        df = df[~df.journal_sjr.isna()]
        df["year"] = df["year"] - df["year"].mean()
        df["label"] = bs
        dfs.append(df)
    return pd.concat(dfs)


if __name__ == "__main__":
    main()
