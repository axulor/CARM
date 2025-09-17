# src/core/simulation.py
import numpy as np
import random
from .agent import Agent

class Simulation:
    def __init__(self, params):
        self.params = params
        self.L = params['L']
        self.N = self.L * self.L
        
        self.grid = np.empty((self.L, self.L), dtype=object)
        self.agents = []
        self._setup_grid()
        
        self.history_rho_C = []
        self.history_L_CD = []
        self.time_steps = []

    def _setup_grid(self):
        for i in range(self.L):
            for j in range(self.L):
                action = 'C' if random.random() < self.params['INIT_COOP_RATIO'] else 'D'
                agent = Agent(len(self.agents), (i, j), action, self.params)
                self.grid[i, j] = agent
                self.agents.append(agent)
        
        for i in range(self.L):
            for j in range(self.L):
                agent = self.grid[i, j]
                agent.neighbors.append(self.grid[(i - 1) % self.L, j])
                agent.neighbors.append(self.grid[(i + 1) % self.L, j])
                agent.neighbors.append(self.grid[i, (j - 1) % self.L])
                agent.neighbors.append(self.grid[i, (j + 1) % self.L])

    def _collect_metrics(self, step):
        num_cooperators = sum(1 for agent in self.agents if agent.action == 'C')
        boundary_length = 0
        for i in range(self.L):
            for j in range(self.L):
                agent = self.grid[i, j]
                right = self.grid[i, (j + 1) % self.L]
                down = self.grid[(i + 1) % self.L, j]
                if agent.action != right.action: boundary_length += 1
                if agent.action != down.action: boundary_length += 1
        
        self.history_rho_C.append(num_cooperators / self.N)
        self.history_L_CD.append(boundary_length)
        self.time_steps.append(step)

    def run(self):
        print(f"Starting SYNC simulation for {self.params.get('combo_id', 'unknown combo')}...")
        self._collect_metrics(0)
        max_mc_steps = self.params['MAX_MC_STEPS']

        for step in range(1, max_mc_steps + 1):
            actions_prev = {agent.id: agent.action for agent in self.agents}
            
            # All agents decide based on the previous state
            next_actions = [agent.decide_next_action() for agent in self.agents]
            
            # All agents update their actions simultaneously
            for agent, next_action in zip(self.agents, next_actions):
                agent.action = next_action

            if step % self.params['RECORD_INTERVAL_MC'] == 0:
                self._collect_metrics(step)
                print(f"  MC Step {step}/{max_mc_steps} | ρ_C: {self.history_rho_C[-1]:.3f}")
        
        print("SYNC Simulation finished.")
        return self.time_steps, self.history_rho_C, self.history_L_CD