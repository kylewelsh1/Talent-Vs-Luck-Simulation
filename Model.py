import math
import time
import plotly.express as px
import numpy as np
import pandas as pd
from numba import njit
from numba import types
from numba.typed import Dict
from scipy import stats
import matplotlib.pyplot as plt
from numba.typed import List


@njit(nopython=True)
def model(n_agents, n_ticks, p_lucky, init_capital, p_event, A_talent):
    A_capital = np.zeros((n_agents, n_ticks + 1))
    A_capital[:, 0] = init_capital
    
    A_event_counter = np.zeros(n_agents)
    A_lucky_event_counter = np.zeros(n_agents)

    for i in range(1, n_ticks + 1):
        gets_event = np.random.binomial(1, p=p_event, size=n_agents)
        n_events = gets_event.sum()
        
        gets_lucky = np.random.binomial(1, p=p_lucky, size=n_events)

        lucky_idx = np.where(gets_event == 1)[0][gets_lucky == 1]
        unlucky_idx = np.where(gets_event == 1)[0][gets_lucky != 1]
        
        A_lucky_event_counter[lucky_idx] += 1
        
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
    return A_capital, A_talent, A_lucky_event_counter


@njit(nopython=True)
def simulate(
    n_agents, n_ticks, n_reps, p_lucky,
    init_capital, mu_talent, sigma_talent, p_event
):
    A_talent = np.clip(
        np.random.normal(mu_talent, sigma_talent, size=n_agents), 0, 1
    )
    AR_final_capital = np.zeros(
        (n_agents, n_ticks + 1, n_reps), dtype="float64"
    )
    for i in range(n_reps):
        A_capital, A_talent, A_lucky_event_counter = model(
            n_agents=n_agents,
            n_ticks=n_ticks,
            p_lucky=p_lucky,
            init_capital=init_capital,
            p_event=p_event,
            A_talent=A_talent
        )

        AR_final_capital[:, :, i] = A_capital
    return AR_final_capital, A_talent, A_lucky_event_counter