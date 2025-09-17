# src/run_fig7_simulations.py
import pandas as pd
import numpy as np
import sys
import os
import time
from mpi4py import MPI

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
from src.core.simulation import Simulation
from src.core.agent import Agent

# ===================================================================
# 1. 继承 Agent 和 Simulation 类 (与实验8相同)
# ===================================================================
class AgentForFig7(Agent):
    def decide_next_action(self):
        r_factor = self.params['R_FACTOR']
        cost = self.params['COST']
        beta = self.params['BETA']
        gamma = self.params['GAMMA']
        kappa = self.params['KAPPA']
        factual_a = self.action
        cf_a = 'D' if factual_a == 'C' else 'C'
        s = sum(1 for n in self.neighbors if n.action == 'C')
        get_payoff_func = self.params['get_payoff_func']
        pi_weight_func = self.params['pi_weight_func']
        omega_weight_func = self.params['omega_weight_func']
        factual_payoff = get_payoff_func(factual_a, s, r_factor, cost)
        total_inst_regret = 0.0
        for s_prime in range(5):
            cf_payoff_at_s = get_payoff_func(cf_a, s, r_factor, cost)
            cf_payoff_at_s_prime = get_payoff_func(cf_a, s_prime, r_factor, cost)
            self_contribution = cf_payoff_at_s - factual_payoff
            others_contribution = cf_payoff_at_s_prime - cf_payoff_at_s
            omega = omega_weight_func(others_contribution, gamma)
            learning_signal = self_contribution + omega * others_contribution
            asymmetric_signal = learning_signal if learning_signal > 0 else kappa * learning_signal
            distance = abs(s_prime - s) + 1
            pi = pi_weight_func(distance, beta)
            total_inst_regret += asymmetric_signal * pi
        self.instant_regret = total_inst_regret
        if cf_a == 'C':
            self.regret_table[s, 0] += total_inst_regret
        else:
            self.regret_table[s, 1] += total_inst_regret
        pos_regret_C = max(0, self.regret_table[s, 0])
        pos_regret_D = max(0, self.regret_table[s, 1])
        sum_pos_regrets = pos_regret_C + pos_regret_D
        prob_C = pos_regret_C / sum_pos_regrets if sum_pos_regrets > 0 else 0.5
        return 'C' if np.random.rand() < prob_C else 'D'

class SimulationForFig7(Simulation):
    def _setup_grid(self):
        from src.core.utils import get_payoff, pi_weight, omega_weight
        self.params['get_payoff_func'] = get_payoff
        self.params['pi_weight_func'] = pi_weight
        self.params['omega_weight_func'] = omega_weight
        for i in range(self.L):
            for j in range(self.L):
                action = 'C' if np.random.rand() < self.params['INIT_COOP_RATIO'] else 'D'
                agent = AgentForFig7(len(self.agents), (i, j), action, self.params)
                self.grid[i, j] = agent
                self.agents.append(agent)
        for i in range(self.L):
            for j in range(self.L):
                agent = self.grid[i, j]
                agent.neighbors.append(self.grid[(i - 1) % self.L, j])
                agent.neighbors.append(self.grid[(i + 1) % self.L, j])
                agent.neighbors.append(self.grid[i, (j - 1) % self.L])
                agent.neighbors.append(self.grid[i, (j + 1) % self.L])


def run_resilience_simulation(task_params):
    """
    运行单次仿真，记录合作韧性P(C->C)和转化率P(D->C)的时间序列。
    """
    run_id = task_params['run_id']
    total_steps = task_params['total_steps']
    sim_params = task_params['sim_params']
    combo_id = sim_params['combo_id']
    
    start_time = time.time()
    sim = SimulationForFig7(sim_params)
    
    time_series_data = []
    
    for step in range(1, total_steps + 1):
        actions_prev = {agent.id: agent.action for agent in sim.agents}
        cooperators_t = {agent.id for agent in sim.agents if agent.action == 'C'}
        defectors_t = {agent.id for agent in sim.agents if agent.action == 'D'}
        
        next_actions = [agent.decide_next_action() for agent in sim.agents]
        
        num_cooperators_t = len(cooperators_t)
        num_defectors_t = len(defectors_t)
        
        retained_cooperators = 0
        converted_defectors = 0
        
        for i, agent in enumerate(sim.agents):
            action_t = actions_prev[agent.id]
            action_t1 = next_actions[i]
            
            if action_t == 'C' and action_t1 == 'C':
                retained_cooperators += 1
            elif action_t == 'D' and action_t1 == 'C':
                converted_defectors += 1

        retention_rate = retained_cooperators / num_cooperators_t if num_cooperators_t > 0 else 0
        conversion_rate = converted_defectors / num_defectors_t if num_defectors_t > 0 else 0
        
        time_series_data.append({
            'combo_id': combo_id,
            'run_id': run_id,
            'step': step,
            'p_c_to_c': retention_rate,
            'p_d_to_c': conversion_rate
        })

        for agent, next_action in zip(sim.agents, next_actions):
            agent.action = next_action

    duration = time.time() - start_time
    return pd.DataFrame(time_series_data), duration


def main():
    """
    MPI主程序，用于分发、执行和汇总实验7的仿真。
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- 实验参数 ---
    NUM_RUNS = 10
    TOTAL_STEPS = 10000
    
    # 实验6的三组参数组合
    PARAMS_SET = [
        {
            "combo_id": "oscillation",
            "L": 50, "INIT_COOP_RATIO": 0.5, "COST": 1.0,
            "R_FACTOR": 3.7, "BETA": 0.2, "GAMMA": 0.8, "KAPPA": 0.3
        },
        {
            "combo_id": "high_coop",
            "L": 50, "INIT_COOP_RATIO": 0.5, "COST": 1.0,
            "R_FACTOR": 3.7, "BETA": 0.5, "GAMMA": 0.5, "KAPPA": 1.0
        },
        {
            "combo_id": "low_coop", # 假设这是低合作情况
            "L": 50, "INIT_COOP_RATIO": 0.5, "COST": 1.0,
            "R_FACTOR": 3.7, "BETA": 0.5, "GAMMA": 0.5, "KAPPA": 0.2
        }
    ]
    
    tasks = []
    if rank == 0:
        for params in PARAMS_SET:
            for run_i in range(NUM_RUNS):
                tasks.append({'sim_params': params, 'run_id': run_i, 'total_steps': TOTAL_STEPS})
        print(f"Master (Rank 0): Total {len(tasks)} tasks to distribute among {size} workers.")

    tasks = comm.bcast(tasks, root=0)

    results_this_rank = []
    for i in range(rank, len(tasks), size):
        task = tasks[i]
        combo_id_str = task['sim_params']['combo_id']
        print(f"Rank {rank}: Starting task {i+1}/{len(tasks)} (combo={combo_id_str}, run_id={task['run_id']})")
        df, duration = run_resilience_simulation(task)
        results_this_rank.append(df)
        print(f"Rank {rank}: Finished task {i+1}/{len(tasks)}. Duration: {duration:.2f}s")
    
    all_rank_results = comm.gather(results_this_rank, root=0)

    if rank == 0:
        print("Master (Rank 0): All tasks completed. Merging, aggregating, and saving final data...")
        if not all_rank_results or not any(all_rank_results):
             print("Warning: No data was generated.")
             return

        final_df = pd.concat([item for sublist in all_rank_results for item in sublist], ignore_index=True)
        
        # 对10次运行的结果进行最终聚合，只取均值
        mean_df = final_df.groupby(['combo_id', 'step']).mean().reset_index()
        mean_df = mean_df.drop(columns=['run_id'])

        output_dir = os.path.join(PROJECT_ROOT, 'data', 'fig7')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "fig7_resilience_dynamics_data.csv")
        mean_df.to_csv(output_path, index=False)
        print(f"Master (Rank 0): Final mean resilience dynamics data saved to {output_path}")

if __name__ == "__main__":
    main()