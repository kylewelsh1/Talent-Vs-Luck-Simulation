import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px


def lognormal(x, mu, sigma) :
    return 1/(np.sqrt(2*np.pi)*sigma*x)*np.exp(-((np.log(x)- 
    mu)**2)/(2*sigma**2))


def loglog_plot(results):
    bin_ranges = list(np.linspace(0, 100, 10))
    final_df = pd.DataFrame({"bin": bin_ranges[1:], "wealth_count": 0})
    
    binned_df = pd.DataFrame(
        {
            "Wealth": results[:, -1, 0],
            "Bin": pd.cut(results[:, -1, 0], bin_ranges)
        }
    )
    final_df["wealth_count"] = binned_df.groupby("Bin").count()["Wealth"].to_list()
    final_df = final_df[final_df["wealth_count"] > 0]
    loglog_fig = px.scatter(
        y=np.log(final_df["wealth_count"]), x=np.log(final_df["bin"])
    )

    return loglog_fig

def talented_vs_untalented_plot(results, talent, show_curve):
    A_talented = np.where(talent >= .8)[0]
    A_untalented = np.where(talent < .8)[0]
    
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
    
    fig_all = ff.create_distplot(
        hist_data_all, group_labels_all,
        show_rug=False, bin_size=10, show_curve=show_curve
    )
    return fig_all


def plot_lognormal_estimate(results):
    x = np.linspace(.1, 2000, 5000)
    mu = np.log(results[:, -1, :].mean(axis=1)).mean()
    sigma = np.log(results[:, -1, :].mean(axis=1)).std()
    
    results_trimmed = results[:, -1, :].mean(axis=1)
    y = lognormal(x, mu, sigma)
    
    log_norm_fig = go.Figure()

    log_norm_fig.add_trace(
        go.Histogram(
            x=results_trimmed, histnorm='probability density',
            name="Histogram"
        )
    )
    log_norm_fig.add_trace(
        go.Scatter(x=x, y=y, name="Estimated Lognormal PDF")
    )
    return log_norm_fig