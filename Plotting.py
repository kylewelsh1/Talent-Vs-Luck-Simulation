import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from scipy.stats import norm
from scipy.stats import gaussian_kde


def lognormal(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma*x)*np.exp(-((np.log(x) - mu)**2)/(
        2*sigma**2))


@st.cache
def create_binned_wealth_df(results, talent):
    bin_ranges = list(
        np.linspace(0, np.max(results[:, -1, :]), 300)
    )
    bin_ranges_log = list(
        np.linspace(-10, np.log(np.max(results[:, -1, :])), 30)
    )
    binned_df_list = []
    for i in range(1, len(results[0, 0, :])):
        binned_df = pd.DataFrame(
            {
                "Wealth": results[:, -1, i],
                "Wealth_log": np.log(results[:, -1, i]),
                "Bin": pd.cut(results[:, -1, i], bin_ranges),
                "Bin_log": pd.cut(np.log(results[:, -1, i]), bin_ranges_log),
                "Talent": talent
            }
        )
        binned_df_list.append(binned_df)
    return binned_df_list, bin_ranges, bin_ranges_log


@st.cache
def create_aggregated_df(binned_df_list, bin_ranges, bin_ranges_log):
    aggregated_df = pd.DataFrame({"bin": bin_ranges[1:]})
    aggregated_df_log = pd.DataFrame({"bin_log": bin_ranges_log[1:]})

    for i in range(len(binned_df_list)):
        wealth_count_col = binned_df_list[i].groupby("Bin").count()["Wealth"]
        aggregated_df[f"wealth_count_{i}"] = wealth_count_col.to_list()

        wealth_count_col_log = binned_df_list[i].groupby("Bin_log").count()
        wealth_count_col_log = wealth_count_col_log["Wealth_log"].to_list()

        aggregated_df_log[f"wealth_count_log_{i}"] = wealth_count_col_log

        talent_mean_col = binned_df_list[i].groupby("Bin").mean(
            numeric_only=True
        )
        aggregated_df[f"talent_mean_{i}"] = talent_mean_col["Talent"].to_list()

    talent_col_names = aggregated_df.filter(regex="^talent_mean").columns
    talent_col_names_list = talent_col_names.to_list()

    keep_cols = ["bin", "mean_count"] + talent_col_names_list

    aggregated_df["mean_count"] = aggregated_df.filter(
        regex="^wealth_count"
    ).mean(axis=1)
    aggregated_df = aggregated_df[aggregated_df["mean_count"] > 0]
    aggregated_df = aggregated_df[keep_cols]

    aggregated_df_log["mean_count_log"] = aggregated_df_log.drop(
        columns="bin_log"
    ).mean(axis=1)

    aggregated_df_log = aggregated_df_log[
        aggregated_df_log["mean_count_log"] > 0
    ]
    aggregated_df_log = aggregated_df_log[["bin_log", "mean_count_log"]]
    return aggregated_df, aggregated_df_log


def wealth_dist_plot(results_aggregated, bin_width):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=results_aggregated["bin"],
            y=results_aggregated["mean_count"],
            width=bin_width
            )
        )
    fig.update_layout(
        xaxis_title="Wealth Bin",
        yaxis_title="Wealth Bin Count"
    )
    return fig


def loglog_plot(results):
    Y = np.log(results["mean_count"])
    X = sm.add_constant(np.log(results["bin"]))

    mod = sm.OLS(Y, X).fit()
    intercept = mod.params[0]
    slope = mod.params[1]

    min_x = np.log(results["bin"]).min()
    max_x = np.log(results["bin"]).max()
    num_points = len(results["mean_count"])

    x_lin = np.linspace(min_x, max_x, num_points)
    y_lin = intercept + slope*x_lin

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=np.log(results["mean_count"]),
            x=np.log(results["bin"]),
            mode="markers",
            name="Binned Log-log Wealth"
            )
    )
    fig.add_trace(
        go.Scatter(
            x=x_lin, y=y_lin, mode="lines", name="Fitted Regression Line"
         )
    )
    fig.update_layout(
        xaxis_title="Log Of Wealth Bin", yaxis_title="Log Of Bin Count"
    )
    return fig, slope, intercept


def plot_power_law_estimate(results, slope, intercept):
    x = np.linspace(.1, results["bin"].max(), 2000)
    y = np.exp(intercept) * x**slope

    x_trimmed = x[y < 10000]
    y_trimmed = y[y < 10000]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=results["bin"],
            y=results["mean_count"],
            name="Histogram"
            )
        )

    fig.add_trace(go.Scatter(
        x=x_trimmed, y=y_trimmed, name="Estimated Power Law Distribution")
    )
    fig.update_layout(
        xaxis_title="Wealth Bin",
        yaxis_title="Wealth Bin Count"
    )
    return fig


def log_wealth_plot(results, bin_width):
    n_total_agents = results["mean_count_log"].sum()
    
    results_unbinned = results.reindex(
        results.index.repeat(results["mean_count_log"])
    )
    
    mu = results_unbinned["bin_log"].mean()
    sigma = results_unbinned["bin_log"].std()

    x = np.linspace(-10, 10, 100)
    y = norm.pdf(x, mu, sigma)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=results["bin_log"],
            y=(results["mean_count_log"]/n_total_agents)/bin_width,
            width=bin_width,
            name="Histogram"
            )
       )
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines", name="Estimated Normal Distribution"
            )
        )
    fig.update_layout(
        xaxis_title="Log Wealth Bin",
        yaxis_title="Bin Count Percent"
    )
    return fig


def plot_lognormal_estimate(results, results_log, bin_width):
    n_total_agents = results["mean_count"].sum()
    
    results_unbinned_log = results_log.reindex(
        results_log.index.repeat(results_log["mean_count_log"])
    )

    mu = results_unbinned_log["bin_log"].mean()
    sigma = results_unbinned_log["bin_log"].std()

    x = np.linspace(.1, results["bin"].max(), 2000)
    y = lognormal(x, mu, sigma)
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=results["bin"],
            y=(results["mean_count"]/n_total_agents) / bin_width,
            width=bin_width,
            name="Histogram"
            )
        )
    fig.add_trace(
        go.Scatter(x=x, y=y, name="Estimated Lognormal PDF")
    )
    fig.update_layout(
        xaxis_title="Wealth Bin",
        yaxis_title="Wealth Bin Percent"
    )
    return fig


@st.cache
def create_talented_untalented_df(
        binned_df_list, bin_ranges, mu_talent, sigma_talent
):
    df_talented = pd.DataFrame({"bin": bin_ranges[1:]})
    df_untalented = pd.DataFrame({"bin": bin_ranges[1:]})

    for i in range(len(binned_df_list)):
        talented_df_binned = binned_df_list[i][
            binned_df_list[i]["Talent"] >= mu_talent + 2*sigma_talent
        ]
        wealth_count_col_talented = talented_df_binned.groupby("Bin") \
                                                      .count()["Wealth"]
        df_talented[f"wealth_count_{i}"] = wealth_count_col_talented.to_list()

        untalented_df_binned = binned_df_list[i][
            binned_df_list[i]["Talent"] < mu_talent + 2*sigma_talent
        ]
        wealth_count_col_untalented = untalented_df_binned.groupby("Bin") \
                                                          .count()["Wealth"]
        wealth_count_col_untalented = wealth_count_col_untalented.to_list()
        df_untalented[f"wealth_count_{i}"] = wealth_count_col_untalented

    df_talented["mean_count"] = df_talented.filter(
        regex="^wealth_count"
    ).mean(axis=1)

    df_untalented["mean_count"] = df_untalented.filter(
        regex="^wealth_count"
    ).mean(axis=1)

    df_talented = df_talented[df_talented["mean_count"] > 0]
    df_talented = df_talented[["bin", "mean_count"]]

    df_untalented = df_untalented[df_untalented["mean_count"] > 0]
    df_untalented = df_untalented[["bin", "mean_count"]]
    return df_talented, df_untalented


def talented_vs_untalented_plot(
        results_talented, results_untalented
):
    n_agents_talented = results_talented["mean_count"].sum()
    n_agents_untalented = results_untalented["mean_count"].sum()
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=results_talented["bin"],
            y=results_talented["mean_count"]/n_agents_talented,
            name="Talented"
            )
        )
    fig.add_trace(
        go.Bar(
            x=results_untalented["bin"],
            y=results_untalented["mean_count"]/n_agents_untalented,
            marker_color="firebrick",
            name="Untalented"
            )
        )
    fig.update_layout(
        xaxis_title="Wealth Bin",
        yaxis_title="Wealth Bin Percent"
    )
    return fig


def talent_wealth_corr_plot(results):
    results_long_df = pd.wide_to_long(
        results, "talent_mean", i="bin", j="test", sep="_"
    )
    results_long_df.dropna(inplace=True)
    results_long_df.reset_index(inplace=True)
    
    xy = np.vstack(
        (
            results_long_df["bin"],
            results_long_df["talent_mean"]
        )
    )
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    
    fig = px.scatter(
        x=results_long_df["bin"], y=results_long_df["talent_mean"],
        color=z[idx]
    )
    fig.update_layout(
        xaxis_title="Wealth Bin",
        yaxis_title="Average Talent For Each Rep"
    )
    return fig
