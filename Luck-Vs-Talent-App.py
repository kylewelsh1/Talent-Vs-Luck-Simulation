import Model as model
import Plotting as plotting
import streamlit as st


st.title("Luck Vs Talent Simulation")

aggregation_checkbox = st.sidebar.checkbox("Aggregate")
if aggregation_checkbox:
    n_reps = st.sidebar.number_input(
        "Reps", min_value=2, max_value=100, value=10, step=1
    )
    show_curve=True
else:
    n_reps = 1
    show_curve=False
    
years = st.sidebar.number_input(
    "Years", min_value=1, max_value=100, value=40, step=1
)
n_ticks = 2 * years

n_agents = st.sidebar.number_input(
    "Number Of Agents", min_value=1, max_value=20000, value=10000, step=1
)
mu_talent = st.sidebar.slider(
    label="Average Talent", min_value=0.01, max_value=1.0,
    value=.6
)
max_std = min(1 - mu_talent, mu_talent) / 2

sigma_talent = st.sidebar.slider(
    label="Talent Standard Deviation", min_value=0.01, max_value=max_std,
    value=max_std / 2
)
p_luck = st.sidebar.slider(
    label="Probability Of Lucky Event", min_value=0.01, max_value=1.0,
    value=.5
)
p_event = st.sidebar.slider(
    label="Probability Of Event", min_value=0.01, max_value=1.0,
    value=.04
)

results, talent = model.simulate(
    n_agents=n_agents, n_ticks=n_ticks, n_reps=n_reps, p_lucky=p_luck,
    init_capital=100, mu_talent=mu_talent, sigma_talent=sigma_talent,
    p_event=p_event
)

tab1, tab2 = st.tabs(["Walth Distribution", "Talented Vs Untalented"])

with tab1:
    st.header("Wealth Distribution")
    wealth_dist_fig = plotting.wealth_dist_plot(results, show_curve)
    st.plotly_chart(wealth_dist_fig)
    
    st.header("Estimate Wealth Distribution Function")
    dist_choice = st.selectbox("Distribution", ["Log-normal", "Power Law"])
    if dist_choice == "Log-normal":
        st.header("Distribution Of Log Wealth")
        log_wealth_fig = plotting.log_wealth_plot(results)
        st.plotly_chart(log_wealth_fig)
        
        st.header("Estimate of Lognormal Wealth Distribution")
        lognormal_fig = plotting.plot_lognormal_estimate(results)
        st.plotly_chart(lognormal_fig)
    else:
        st.header("Log-log Plot")
        loglog_fig, slope, intercept = plotting.loglog_plot(results)
        st.plotly_chart(loglog_fig)
        
        st.header("Estimate of Power Law Wealth Distribution")
        power_law_fig = plotting.plot_power_law_estimate(
            results, slope, intercept
        )
        st.plotly_chart(power_law_fig)

with tab2:
    st.header("Wealth Distribution Talented Vs Untalented")
    talented_vs_untalented_fig = plotting.talented_vs_untalented_plot(
        results, talent, show_curve, mu_talent, sigma_talent
    )
    st.plotly_chart(talented_vs_untalented_fig)
    
    st.plotly_chart(plotting.talent_wealth_corr_plot(results, talent))

