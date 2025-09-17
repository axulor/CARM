# src/run_fig8_simulations.py
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
# 1. 继承 Agent 和 Simulation 类 
# ===================================================================
class AgentForFig8(Agent):
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

class SimulationForFig8(Simulation):
    def _setup_grid(self):
        from src.core.utils import get_payoff, pi_weight, omega_weight
        self.params['get_payoff_func'] = get_payoff
        self.params['pi_weight_func'] = pi_weight
        self.params['omega_weight_func'] = omega_weight
        for i in range(self.L):
            for j in range(self.L):
                action = 'C' if np.random.rand() < self.params['INIT_COOP_RATIO'] else 'D'
                agent = AgentForFig8(len(self.agents), (i, j), action, self.params)
                self.grid[i, j] = agent
                self.agents.append(agent)
        for i in range(self.L):
            for j in range(self.L):
                agent = self.grid[i, j]
                agent.neighbors.append(self.grid[(i - 1) % self.L, j])
                agent.neighbors.append(self.grid[(i + 1) % self.L, j])
                agent.neighbors.append(self.grid[i, (j - 1) % self.L])
                agent.neighbors.append(self.grid[i, (j + 1) % self.L])


def run_final_analysis_simulation(task_params):
    """
    运行单次仿真，并在内部完成所有聚合和理论预测，返回最终的统计结果。
    """
    r_factor = task_params['r']
    run_id = task_params['run_id']
    total_steps = task_params['total_steps']
    sampling_steps = task_params['sampling_steps']
    base_params = task_params['base_params']
    
    sim_params = base_params.copy()
    sim_params['R_FACTOR'] = r_factor
    
    start_time = time.time()
    sim = SimulationForFig8(sim_params)
    
    micro_data_samples = []
    flow_counts = {'C_total': 0, 'C_to_D': 0, 'D_total': 0, 'D_to_C': 0}

    for step in range(1, total_steps + 1):
        actions_prev = {agent.id: agent.action for agent in sim.agents}
        next_actions = [agent.decide_next_action() for agent in sim.agents]
        
        if step > total_steps - sampling_steps:
            for i, agent in enumerate(sim.agents):
                s = sum(1 for n in agent.neighbors if n.action == 'C')
                action_t = actions_prev[agent.id]
                action_t1 = next_actions[i]
                
                micro_data_samples.append({
                    'agent_action': action_t, 'num_coop_neighbors': s,
                    'regret_C': agent.regret_table[s, 0], 'regret_D': agent.regret_table[s, 1]
                })
                
                if action_t == 'C':
                    flow_counts['C_total'] += 1
                    if action_t1 == 'D':
                        flow_counts['C_to_D'] += 1
                else:
                    flow_counts['D_total'] += 1
                    if action_t1 == 'C':
                        flow_counts['D_to_C'] += 1

        for agent, next_action in zip(sim.agents, next_actions):
            agent.action = next_action
    
    if not micro_data_samples:
        return pd.DataFrame(), time.time() - start_time

    raw_df = pd.DataFrame(micro_data_samples)
    
    # 1. 计算实际观测值
    rho_c_observed = (raw_df['agent_action'] == 'C').mean()
    p_c_to_d_observed = flow_counts['C_to_D'] / flow_counts['C_total'] if flow_counts['C_total'] > 0 else 0
    p_d_to_c_observed = flow_counts['D_to_C'] / flow_counts['D_total'] if flow_counts['D_total'] > 0 else 0

    # 2. 计算理论预测值所需的所有中间量
    p_s_given_C = raw_df[raw_df['agent_action'] == 'C']['num_coop_neighbors'].value_counts(normalize=True)
    p_s_given_D = raw_df[raw_df['agent_action'] == 'D']['num_coop_neighbors'].value_counts(normalize=True)
    raw_df['pos_regret_C'] = raw_df['regret_C'].clip(lower=0)
    raw_df['pos_regret_D'] = raw_df['regret_D'].clip(lower=0)
    sum_pos_regrets = raw_df['pos_regret_C'] + raw_df['pos_regret_D']
    prob_C = pd.Series(0.5, index=raw_df.index)
    mask = sum_pos_regrets > 0
    prob_C[mask] = raw_df['pos_regret_C'][mask] / sum_pos_regrets[mask]
    raw_df['prob_C'] = prob_C
    raw_df['prob_D'] = 1 - prob_C
    prob_C_to_D_s = raw_df[raw_df['agent_action'] == 'C'].groupby('num_coop_neighbors')['prob_D'].mean()
    prob_D_to_C_s = raw_df[raw_df['agent_action'] == 'D'].groupby('num_coop_neighbors')['prob_C'].mean()
    p_c_to_d_predicted = (p_s_given_C * prob_C_to_D_s).sum()
    p_d_to_c_predicted = (p_s_given_D * prob_D_to_C_s).sum()
    
    if (p_d_to_c_predicted + p_c_to_d_predicted) == 0:
        rho_c_predicted = rho_c_observed
    else:
        rho_c_predicted = p_d_to_c_predicted / (p_d_to_c_predicted + p_c_to_d_predicted)

    # 3. 将所有用于绘图的数据整理成一个DataFrame
    s_range = pd.DataFrame({'num_coop_neighbors': range(5)})
    plot_df = pd.merge(s_range, p_s_given_C.reset_index().rename(columns={'proportion': 'p_s_given_C'}), on='num_coop_neighbors', how='left')
    plot_df = pd.merge(plot_df, p_s_given_D.reset_index().rename(columns={'proportion': 'p_s_given_D'}), on='num_coop_neighbors', how='left')
    plot_df = pd.merge(plot_df, prob_C_to_D_s.reset_index().rename(columns={'prob_D': 'prob_C_to_D_s'}), on='num_coop_neighbors', how='left')
    plot_df = pd.merge(plot_df, prob_D_to_C_s.reset_index().rename(columns={'prob_C': 'prob_D_to_C_s'}), on='num_coop_neighbors', how='left')
    plot_df.fillna(0, inplace=True)
    
    plot_df['r'] = r_factor
    plot_df['run_id'] = run_id
    
    plot_df['rho_c_observed'] = rho_c_observed
    plot_df['p_c_to_d_observed'] = p_c_to_d_observed
    plot_df['p_d_to_c_observed'] = p_d_to_c_observed
    plot_df['p_c_to_d_predicted'] = p_c_to_d_predicted
    plot_df['p_d_to_c_predicted'] = p_d_to_c_predicted
    plot_df['rho_c_predicted'] = rho_c_predicted

    duration = time.time() - start_time
    return plot_df, duration

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    R_VALUES = [2.2, 3.0, 3.8]
    NUM_RUNS = 10
    TOTAL_STEPS = 10000
    SAMPLING_STEPS = 1000
    BASE_PARAMS = {"L": 50, "INIT_COOP_RATIO": 0.5, "COST": 1.0, "BETA": 0.5, "GAMMA": 0.5, "KAPPA": 0.0}
    
    tasks = []
    if rank == 0:
        for r_val in R_VALUES:
            for run_i in range(NUM_RUNS):
                tasks.append({'r': r_val, 'run_id': run_i, 'total_steps': TOTAL_STEPS, 
                              'sampling_steps': SAMPLING_STEPS, 'base_params': BASE_PARAMS})
        print(f"Master (Rank 0): Total {len(tasks)} tasks to distribute among {size} workers.")

    tasks = comm.bcast(tasks, root=0)

    results_this_rank = []
    for i in range(rank, len(tasks), size):
        task = tasks[i]
        print(f"Rank {rank}: Starting task {i+1}/{len(tasks)} (r={task['r']}, run_id={task['run_id']})")
        df, duration = run_final_analysis_simulation(task)
        results_this_rank.append(df)
        print(f"Rank {rank}: Finished task {i+1}/{len(tasks)}. Duration: {duration:.2f}s")
    
    all_rank_results = comm.gather(results_this_rank, root=0)

    if rank == 0:
        print("Master (Rank 0): All tasks completed. Saving final data...")
        if not all_rank_results or not any(all_rank_results):
             print("Warning: No data was generated.")
             return

        final_df = pd.concat([item for sublist in all_rank_results for item in sublist], ignore_index=True)
        
        output_dir = os.path.join(PROJECT_ROOT, 'data', 'fig8')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "fig8_final_verification_data.csv")
        final_df.to_csv(output_path, index=False)
        print(f"Master (Rank 0): Final verification data saved to {output_path}")

if __name__ == "__main__":
    main()
