import numpy as np
import matplotlib.pyplot as plt
import random
import math

# ==============================================================================
# 1. PARAMETER CONFIGURATION
# ==============================================================================
# Grid and Agent Parameters
L = 100  # Grid size (L x L)
N = L * L  # Total number of agents
NEIGHBOR_RADIUS = 1  # Von Neumann neighborhood (4 neighbors)

# Game Parameters (Spatial Public Goods Game)
COST = 1.0      # Cost of cooperation (成本 c)
R_FACTOR = 3.5  # Synergy factor for cooperation (增益因子 r)


# CARM Model Parameters - Classic Exponential Form
# BETA: 距离越大，pi权重下降越快。高BETA代表智能体更“现实”，不考虑遥远的反事实。
# GAMMA: “运气”成分(|ΔU|)越大，omega权重下降越快。高GAMMA代表智能体更“严格”，不信任运气。
BETA = 0.5      # Accessibility sensitivity (距离敏感度 β)
GAMMA = 0.5     # Attribution sensitivity (归因敏感度 γ)
KAPPA = 0.1     # Rejoicing Sensitivity (庆幸敏感度 κ), 0 < KAPPA < 1

# Simulation Parameters
MAX_STEPS = 200 * N      # Total simulation time. One step = one agent update.
INIT_COOP_RATIO = 0.5    # Initial ratio of cooperators
RECORD_INTERVAL = N      # Record data every N agent updates (i.e., every "Monte Carlo step")

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def get_payoff(action, num_coop_neighbors):
    """Calculates the payoff for an agent in the SPGG."""
    num_players_in_group = 4 + 1  # 4 neighbors + self
    if action == 'C':
        return (R_FACTOR * (num_coop_neighbors + 1) / num_players_in_group) - COST
    else:  # action == 'D'
        return (R_FACTOR * num_coop_neighbors) / num_players_in_group

def pi_weight(d, beta):
    """Calculates the accessibility weight (π) using classic exponential decay."""
    # d = |s' - s| + 1
    # 我们将 d-1 作为指数，这样当s'=s (d=1)时，权重为exp(0)=1。
    return np.exp(-beta * (d - 1))

def omega_weight(delta_u, gamma):
    """Calculates the attribution weight (ω) using classic exponential decay."""
    return np.exp(-gamma * abs(delta_u))

# ==============================================================================
# 3. AGENT CLASS (with RESTRUCTURED Attribution Mechanism)
# ==============================================================================

class Agent:
    def __init__(self, agent_id, pos, initial_action):
        self.id = agent_id
        self.pos = pos
        self.action = initial_action
        self.neighbors = []
        self.regret_table = np.zeros((5, 2))

    def decide_next_action(self):
        """
        Core CARM logic with the NEW "Regret/Responsibility Separation" mechanism.
        """
        # --- Step A: Identify Factual and Counterfactual Context ---
        factual_a = self.action
        cf_a = 'D' if factual_a == 'C' else 'C'
        s = sum(1 for n in self.neighbors if n.action == 'C')
        factual_payoff = get_payoff(factual_a, s)

        # --- Step B: Calculate Total Instantaneous Attributed Regret ---
        total_inst_regret = 0.0
        num_neighbors = 4

        # Iterate over all possible counterfactual neighbor states s'
        for s_prime in range(num_neighbors + 1):
            
            # --- 1. Decompose the potential regret into Self and Others' Contributions ---
            
            # Payoff if I had taken cf_action, but neighbors remained at s
            cf_payoff_at_s = get_payoff(cf_a, s)
            
            # Payoff if I had taken cf_action, and neighbors changed to s'
            cf_payoff_at_s_prime = get_payoff(cf_a, s_prime)

            # Self-Contribution: The part of payoff change I am 100% responsible for.
            self_contribution = cf_payoff_at_s - factual_payoff

            # Others' Contribution: The part of payoff change due to "luck" or environment.
            others_contribution = cf_payoff_at_s_prime - cf_payoff_at_s

            # --- 2. Calculate Attribution Weight (ω) ---
            # This weight will ONLY discount the "Others' Contribution".
            omega = omega_weight(others_contribution, GAMMA)

            # --- 3. Construct the Attributed Learning Signal ---
            # The signal is my own contribution PLUS the discounted "lucky" part.
            learning_signal = self_contribution + omega * others_contribution
            
            # --- 4. Apply Asymmetric Emotion (κ) to the Learning Signal ---
            # This is where we feel regret or rejoicing, based on the attributed signal.
            if learning_signal > 0:
                asymmetric_signal = learning_signal # Full regret
            else:
                asymmetric_signal = KAPPA * learning_signal # Discounted rejoicing

            # --- 5. Apply Accessibility Weight (π) ---
            # The final signal is weighted by how "believable" this s' is.
            distance = abs(s_prime - s) + 1
            pi = pi_weight(distance, BETA)
            
            # --- 6. Accumulate the final term for this s' ---
            final_term = asymmetric_signal * pi
            total_inst_regret += final_term

        # --- Step C & D (Unchanged): Update regret table and decide next action ---
        if cf_a == 'C':
            self.regret_table[s, 0] += total_inst_regret
        else: # cf_a == 'D'
            self.regret_table[s, 1] += total_inst_regret

        regret_for_C = self.regret_table[s, 0]
        regret_for_D = self.regret_table[s, 1]

        pos_regret_C = max(0, regret_for_C)
        pos_regret_D = max(0, regret_for_D)

        sum_pos_regrets = pos_regret_C + pos_regret_D

        if sum_pos_regrets > 0:
            prob_C = pos_regret_C / sum_pos_regrets
        else:
            prob_C = 0.5
            
        return 'C' if random.random() < prob_C else 'D'

# ==============================================================================
# 4. SIMULATION ENVIRONMENT CLASS (No changes needed here)
# ==============================================================================

class Simulation:
    def __init__(self):
        self.grid = np.empty((L, L), dtype=object)
        self.agents = []
        self._setup_grid()
        
        self.history_rho_C = []
        self.history_L_CD = [] # C-D Boundary Length
        self.time_steps = []

    def _setup_grid(self):
        """Initializes agents and sets up the grid and neighbor relationships."""
        agent_id_counter = 0
        for i in range(L):
            for j in range(L):
                initial_action = 'C' if random.random() < INIT_COOP_RATIO else 'D'
                agent = Agent(agent_id_counter, (i, j), initial_action)
                self.grid[i, j] = agent
                self.agents.append(agent)
                agent_id_counter += 1
        
        for i in range(L):
            for j in range(L):
                agent = self.grid[i, j]
                agent.neighbors.append(self.grid[(i - 1) % L, j])
                agent.neighbors.append(self.grid[(i + 1) % L, j])
                agent.neighbors.append(self.grid[i, (j - 1) % L])
                agent.neighbors.append(self.grid[i, (j + 1) % L])

    def _collect_metrics(self, step):
        """Collects and stores the key metrics rho_C and L_CD."""
        num_cooperators = sum(1 for agent in self.agents if agent.action == 'C')
        
        boundary_length = 0
        for i in range(L):
            for j in range(L):
                agent = self.grid[i, j]
                right_neighbor = self.grid[i, (j + 1) % L]
                down_neighbor = self.grid[(i + 1) % L, j]
                if agent.action != right_neighbor.action:
                    boundary_length += 1
                if agent.action != down_neighbor.action:
                    boundary_length += 1
        
        self.history_rho_C.append(num_cooperators / N)
        self.history_L_CD.append(boundary_length)
        self.time_steps.append(step / N)

    # def run(self):
    #     """Main simulation loop with asynchronous updating."""
    #     print("Starting simulation...")
    #     self._collect_metrics(0)
        
    #     for step in range(1, MAX_STEPS + 1):
    #         agent_to_update = random.choice(self.agents)
    #         next_action = agent_to_update.decide_next_action()
    #         agent_to_update.action = next_action
            
    #         if step % RECORD_INTERVAL == 0:
    #             self._collect_metrics(step)
    #             print(f"Step {step}/{MAX_STEPS} (MC Step: {step/N:.1f}) | "
    #                   f"ρ_C: {self.history_rho_C[-1]:.3f} | "
    #                   f"L_CD: {self.history_L_CD[-1]}")
        
    #     print("Simulation finished.")

    def run(self):
        print("Starting SYNC simulation...")
        self._collect_metrics(0)
        steps = 500

        for step in range(1, steps + 1):
            # 1. 保存所有当前动作
            actions_prev = [agent.action for agent in self.agents]
            
            # 2. 所有智能体同步计算下一步动作
            next_actions = []
            for idx, agent in enumerate(self.agents):
                # 复制neighbor动作为当前actions_prev
                for k, n in enumerate(agent.neighbors):
                    n.action = actions_prev[n.id]
                agent.action = actions_prev[agent.id]  # 自己动作也复原
                # 决策并记录
                next_action = agent.decide_next_action()
                next_actions.append(next_action)
            
            # 3. 同步替换所有动作
            for agent, a_next in zip(self.agents, next_actions):
                agent.action = a_next

            # 4. 记录
            self._collect_metrics(step)
            print(f"Step {step}/{steps} | ρ_C: {self.history_rho_C[-1]:.3f} | L_CD: {self.history_L_CD[-1]}")
        print("SYNC Simulation finished.")


    def plot_results(self):
        """Visualizes the collected metrics."""
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 使用对数时间轴以更好地观察收敛过程
        ax1.set_xscale('log')
        
        color = 'tab:blue'
        ax1.set_xlabel("Time (Monte Carlo Steps, log scale)")
        ax1.set_ylabel("Cooperator Density (ρ_C)", color=color)
        ax1.plot(self.time_steps, self.history_rho_C, color=color, label='ρ_C')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 1)
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel("C-D Boundary Length (L_CD)", color=color)
        ax2.plot(self.time_steps, self.history_L_CD, color=color, alpha=0.7, label='L_CD')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.suptitle(f'CARM Simulation (r={R_FACTOR}, β={BETA}, γ={GAMMA}, κ={KAPPA})')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# ==============================================================================
# 5. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    # 过滤掉 self.time_steps[0] == 0 导致的对数坐标警告
    import warnings
    warnings.filterwarnings("ignore", message="invalid value encountered in log")
    
    sim = Simulation()
    sim.run()
    sim.plot_results()