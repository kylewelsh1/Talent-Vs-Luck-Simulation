import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from scipy.stats import norm
from scipy.stats import gaussian_kde


def lognormal(x, mu, sigma) :
    return 1/(np.sqrt(2*np.pi)*sigma*x)*np.exp(-((np.log(x) - mu)**2)/(
        2*sigma**2))


def loglog_plot(results):
    #bin_ranges = list(np.linspace(0, 1000, 70))
    bin_ranges = list(np.linspace(0, np.max(results[:, -1, :].mean(axis=1)), 70))
    final_df = pd.DataFrame({"bin": bin_ranges[1:], "wealth_count": 0})
    binned_df = pd.DataFrame(
        {
            "Wealth": results[:, -1, :].mean(axis=1),
            "Bin": pd.cut(results[:, -1, :].mean(axis=1), bin_ranges)
        }
    )
    wealth_count_col = binned_df.groupby("Bin").count()["Wealth"].to_list()
    final_df["wealth_count"] = wealth_count_col
    final_df = final_df[final_df["wealth_count"] > 0]
    
    Y = np.log(final_df["wealth_count"])
    X = sm.add_constant(np.log(final_df["bin"]))
    
    mod = sm.OLS(Y, X).fit()
    intercept = mod.params[0]
    slope = mod.params[1]
    
    min_x = np.log(final_df["bin"]).min()
    max_x = np.log(final_df["bin"]).max()
    num_points = len(final_df["wealth_count"])

    x_lin = np.linspace(min_x, max_x, num_points)
    y_lin = intercept + slope * x_lin
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=np.log(final_df["wealth_count"]),
            x=np.log(final_df["bin"]),
            mode="markers",
            name="Binned Log-log Wealth"
            )
    )
    fig.add_trace(
        go.Scatter(
            x=x_lin, y=y_lin, mode="lines", name="Fitted Regression Line"
            )
        )
    return fig, slope, intercept


def log_wealth_plot(results):
    log_wealth = np.log(results[:, -1, :].mean(axis=1))
    mu = log_wealth.mean()
    sigma = log_wealth.std()
    
    x = np.linspace(-10, 10, 100)
    y = norm.pdf(x, mu, sigma)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Histogram(
            x=log_wealth, histnorm='probability density', name="Histrogram"
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines", name="Estimated Normal Distribution"
            )
        )
    return fig
    

def plot_power_law_estimate(results, slope, intercept):
    x = np.linspace(.1, np.max(results[:, -1, :].mean(axis=1)), 2000)
    y = np.exp(intercept) * x**slope
    
    x_trimmed = x[y < 10000]
    y_trimmed = y[y < 10000]
    
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=results[:, -1, :].mean(axis=1), name="Histogram")
    )
    fig.add_trace(go.Scatter(x=x_trimmed, y=y_trimmed, name="Estimated Power Law Distribution"))
    return fig


def talented_vs_untalented_plot(
        results, talent, show_curve, mu_talent, sigma_talent
    ):
    
    A_talented = np.where(talent >= mu_talent + 2 * sigma_talent)[0]
    A_untalented = np.where(talent < mu_talent + 2 * sigma_talent)[0]
    
    hist_data= [
        results[A_talented, -1, :].mean(axis=1),
        results[A_untalented, -1, :].mean(axis=1)
    ]
    
    group_labels = [
        'Wealth Distribution Talented',
        "Wealth Distribution Untalented"
    ]
    
    fig = ff.create_distplot(
        hist_data, group_labels, show_rug=False,
        bin_size=10, show_curve=show_curve
    )
    return fig


def wealth_dist_plot(results, show_curve):
    hist_data_all= [
        results[:, -1, :].mean(axis=1),
    ]
    
    group_labels_all = [
        'Wealth Distribution'
    ]
    
    fig = ff.create_distplot(
        hist_data_all, group_labels_all,
        show_rug=False, bin_size=10, show_curve=show_curve
    )
    return fig


def plot_lognormal_estimate(results):
    mu = np.log(results[:, -1, :].mean(axis=1)).mean()
    sigma = np.log(results[:, -1, :].mean(axis=1)).std()
    
    x = np.linspace(.1, np.max(results[:, -1, :].mean(axis=1)), 2000)
    y = lognormal(x, mu, sigma)
    
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=results[:, -1, :].mean(axis=1), histnorm='probability density',
            name="Histogram"
        )
    )
    fig.add_trace(
        go.Scatter(x=x, y=y, name="Estimated Lognormal PDF")
    )
    return fig


def talent_wealth_corr_plot(results, talent):
    xy = np.vstack(
        (
            results[:, -1, :].mean(axis=1),
            talent
        )
    )
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    fig = px.scatter(
        x=results[:, -1, :].mean(axis=1), y=talent, color=z[idx]
    )
    fig.update_layout(
        xaxis_title="Final Wealth",
        yaxis_title="Talent"
    )
    return fig