import Model as model
import Plotting as plotting
import streamlit as st


aggregation_selectbox = st.sidebar.selectbox(
    "Aggregation",
    ("Yes", "No")
)

st.title("Luck Vs Talent Simulation")
p_luck = st.slider(
    label="Probability Of Lucky Event", min_value=0.01, max_value=1.0,
    value=.5
)
p_event = st.slider(
    label="Probability Of Event", min_value=0.01, max_value=1.0,
    value=.04
)

if aggregation_selectbox == "No":
    n_reps = 1
    show_curve=False
else:
    show_curve=True
    n_reps = 10
    
results, talent = model.simulate(
    n_agents=10000, n_ticks=80, n_reps=n_reps, p_lucky=p_luck,
    init_capital=100, mu_talent=.6, sigma_talent=.1, p_event=p_event
)

st.header("Wealth Distribution")
wealth_dist_fig = plotting.wealth_dist_plot(results, show_curve)
st.plotly_chart(wealth_dist_fig)

st.header("Wealth Distribution Talented Vs Untalented")
talented_vs_untalented_fig = plotting.talented_vs_untalented_plot(
    results, talent, show_curve
)
st.plotly_chart(talented_vs_untalented_fig)

st.header("Estimate of Lognormal Wealth Distribution")
lognormal_fig = plotting.plot_lognormal_estimate(results)
st.plotly_chart(lognormal_fig)

st.header("Log-log Plot")
loglog_fig, slope, intercept = plotting.loglog_plot(results)
st.plotly_chart(loglog_fig)

st.header("Power Law Plot")
power_law_fig = plotting.plot_power_law_estimate(results, slope, intercept)
st.plotly_chart(power_law_fig)