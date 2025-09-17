# src/core/utils.py
import numpy as np

def get_payoff(action, num_coop_neighbors, r_factor, cost):
    """Calculates the payoff for an agent in the SPGG."""
    num_players_in_group = 4 + 1
    if action == 'C':
        return (r_factor * (num_coop_neighbors + 1) / num_players_in_group) - cost
    else:
        return (r_factor * num_coop_neighbors) / num_players_in_group

def pi_weight(d, beta):
    """Calculates the accessibility weight (π) using classic exponential decay."""
    return np.exp(-beta * (d - 1))

def omega_weight(delta_u, gamma):
    """Calculates the attribution weight (ω) using classic exponential decay."""
    return np.exp(-gamma * abs(delta_u))