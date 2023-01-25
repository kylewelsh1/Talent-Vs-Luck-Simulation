import numpy as np
import streamlit as st
from numba import njit


@njit(nopython=True)
def model(n_agents, n_ticks, p_lucky, init_capital, p_event, A_talent):
    A_capital = np.zeros((n_agents, n_ticks + 1))
    A_capital[:, 0] = init_capital
    
    for i in range(1, n_ticks + 1):
        gets_event = np.random.binomial(1, p=p_event, size=n_agents)
        n_events = gets_event.sum()
        
        gets_lucky = np.random.binomial(1, p=p_lucky, size=n_events)

        lucky_idx = np.where(gets_event == 1)[0][gets_lucky == 1]
        unlucky_idx = np.where(gets_event == 1)[0][gets_lucky != 1]
                
        capitalise_luck = (
            np.random.rand(lucky_idx.shape[0]) < A_talent[lucky_idx]
        )
        capitalise_luck_idx = lucky_idx[capitalise_luck]
        not_capitalise_luck_idx = lucky_idx[~capitalise_luck]

        A_capital[gets_event != 1, i] = A_capital[gets_event != 1, i - 1]
        A_capital[unlucky_idx, i] = A_capital[unlucky_idx, i - 1] / 2
        
        A_capital[capitalise_luck_idx, i] = (
            A_capital[capitalise_luck_idx, i - 1] * 2
        )
        A_capital[not_capitalise_luck_idx, i] = (
            A_capital[not_capitalise_luck_idx, i - 1]
        )
    return A_capital


@st.cache
@njit(nopython=True)
def simulate(
    n_agents, n_ticks, n_reps, p_lucky,
    init_capital, mu_talent, sigma_talent, p_event
):
    A_talent = np.clip(
        np.random.normal(mu_talent, sigma_talent, size=(n_agents, n_reps)), 0, 1
    )
    AR_final_capital = np.zeros(
        (n_agents, n_ticks + 1, n_reps), dtype="float64"
    )
    for i in range(n_reps):
        A_capital = model(
            n_agents=n_agents,
            n_ticks=n_ticks,
            p_lucky=p_lucky,
            init_capital=init_capital,
            p_event=p_event,
            A_talent=A_talent[:, i]
        )
        AR_final_capital[:, :, i] = A_capital
    return AR_final_capital, A_talent