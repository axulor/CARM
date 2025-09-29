# src/core/agent.py
import random
import numpy as np
from .utils import get_payoff, pi_weight, omega_weight

class Agent:
    def __init__(self, agent_id, pos, initial_action, params):
        self.id = agent_id
        self.pos = pos
        self.action = initial_action
        self.neighbors = []
        self.regret_table = np.zeros((5, 2))
        self.params = params # Store parameters
        self.instant_regret = 0 

    def decide_next_action(self):
        # Unpack parameters
        r_factor = self.params['R_FACTOR']
        cost = self.params['COST']
        beta = self.params['BETA']
        gamma = self.params['GAMMA']
        kappa = self.params['KAPPA']

        # Factual and Counterfactual Context ---
        factual_a = self.action
        cf_a = 'D' if factual_a == 'C' else 'C'
        s = sum(1 for n in self.neighbors if n.action == 'C')
        factual_payoff = get_payoff(factual_a, s, r_factor, cost)

        # Calculate Total Instantaneous Attributed Regret
        total_inst_regret = 0.0
        for s_prime in range(5):
            # Decompose
            cf_payoff_at_s = get_payoff(cf_a, s, r_factor, cost)
            cf_payoff_at_s_prime = get_payoff(cf_a, s_prime, r_factor, cost)
            self_contribution = cf_payoff_at_s - factual_payoff
            others_contribution = cf_payoff_at_s_prime - cf_payoff_at_s

            # Attribution Weight (ω)
            omega = omega_weight(others_contribution, gamma)

            # Learning Signal
            learning_signal = self_contribution + omega * others_contribution
            
            # Asymmetric Emotion (κ)
            asymmetric_signal = learning_signal if learning_signal > 0 else kappa * learning_signal
            
            # Accessibility Weight (π)
            distance = abs(s_prime - s) + 1
            pi = pi_weight(distance, beta)
            
            # Accumulate
            total_inst_regret += asymmetric_signal * pi

        self.instant_regret = total_inst_regret

        # Update regret and decide action
        if cf_a == 'C':
            self.regret_table[s, 0] += total_inst_regret
        else:
            self.regret_table[s, 1] += total_inst_regret

        pos_regret_C = max(0, self.regret_table[s, 0])
        pos_regret_D = max(0, self.regret_table[s, 1])
        sum_pos_regrets = pos_regret_C + pos_regret_D

        prob_C = pos_regret_C / sum_pos_regrets if sum_pos_regrets > 0 else 0.5
            

        return 'C' if random.random() < prob_C else 'D'

