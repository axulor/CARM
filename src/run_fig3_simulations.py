import json
import os
import pandas as pd
import numpy as np
import sys
import time
from itertools import product
from collections import deque
from mpi4py import MPI

# 确保可以从src/core导入模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
from src.core.simulation import Simulation

def run_single_simulation(task_info):
    """
    执行单次仿真，无早停，返回最后 N 步的分布。
    这是MPI并行池的最小工作单元。
    """
    params, run_id, exp_id, varying_param_name = task_info
    
    sim = Simulation(params)
    
    max_steps = params['max_total_steps']
    # 注意：在原始脚本中，early_stop_window被用作采样长度
    sampling_window = params['early_stop_window'] 
    
    rho_c_history = deque(maxlen=sampling_window)
    
    # 严格运行所有步数
    for step in range(1, max_steps + 1):
        next_actions = [agent.decide_next_action() for agent in sim.agents]
        for agent, next_action in zip(sim.agents, next_actions):
            agent.action = next_action
        
        current_rho_c = sum(1 for agent in sim.agents if agent.action == 'C') / sim.N
        rho_c_history.append(current_rho_c)
        
    # 返回结果和所有必要的标识符，以便后续聚合
    return {
        'exp_id': exp_id,
        'r': params['R_FACTOR'],
        'varying_param_name': varying_param_name,
        'varying_param_value': params[varying_param_name],
        'run_id': run_id,
        'distribution': list(rho_c_history)
    }

def run_all_experiments_mpi():
    """
    读取所有Fig3配置文件，生成总任务列表，用MPI并行运行，
    最后由0号进程聚合结果并保存。
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- 只有0号进程负责生成总任务列表 ---
    all_tasks = None
    if rank == 0:
        print(f"===== Rank 0: Generating task list for Figure 3 on {size} total workers... =====")
        configs_to_run = [
            'config/fig3a_vary_beta_g0.5_k0.5.json',
            'config/fig3b_vary_gamma_b0.5_k0.5.json',
            'config/fig3c_vary_kappa_b0.5_g0.5.json',
        ]
        config_paths = [os.path.join(PROJECT_ROOT, conf) for conf in configs_to_run]
        
        task_list = []
        for config_path in config_paths:
            with open(config_path, 'r') as f:
                config = json.load(f)

            exp_id = config['experiment_id']
            num_runs = config['num_runs']
            r_range = config['R_FACTOR_RANGE']
            r_values = np.arange(r_range[0], r_range[1] + 1e-9, r_range[2])
            
            varying_param_name, varying_values = None, None
            if 'BETA_VALUES' in config:
                varying_param_name, varying_values = 'BETA', config['BETA_VALUES']
            elif 'GAMMA_VALUES' in config:
                varying_param_name, varying_values = 'GAMMA', config['GAMMA_VALUES']
            elif 'KAPPA_VALUES' in config:
                varying_param_name, varying_values = 'KAPPA', config['KAPPA_VALUES']

            for r_val, p_val in product(r_values, varying_values):
                for i in range(num_runs):
                    current_params = config.copy()
                    current_params['R_FACTOR'] = r_val
                    current_params[varying_param_name] = p_val
                    task_info = (current_params, i, exp_id, varying_param_name)
                    task_list.append(task_info)
        
        all_tasks = task_list
        print(f"Total individual simulations to run: {len(all_tasks)}")

    # --- 广播任务列表给所有进程 ---
    all_tasks = comm.bcast(all_tasks, root=0)

    # --- 每个进程计算自己的任务子集 ---
    my_tasks = [task for i, task in enumerate(all_tasks) if i % size == rank]
    
    local_results = []
    for i, task in enumerate(my_tasks):
        local_results.append(run_single_simulation(task))
        if (i + 1) % 5 == 0: # 打印本地进度
            print(f"  Rank {rank}: completed {i+1}/{len(my_tasks)} local tasks.", flush=True)

    # --- 收集所有结果到0号进程 ---
    gathered_results = comm.gather(local_results, root=0)

    if rank == 0:
        print("\n===== Rank 0: All simulations finished. Aggregating results... =====")
        flat_results = [item for sublist in gathered_results for item in sublist]
        df = pd.DataFrame(flat_results)
        
        # --- 聚合和保存 ---
        for exp_id, group_df in df.groupby('exp_id'):
            print(f"  Processing experiment: {exp_id}")
            varying_param_name = group_df['varying_param_name'].iloc[0]
            
            # 对每个参数组合的所有runs的分布进行平均
            averaged_distributions = group_df.groupby(['r', 'varying_param_value'])['distribution'].apply(
                lambda x: np.mean(np.vstack(x.tolist()), axis=0)
            )
            
            # 转换为长格式DataFrame
            final_rows = []
            for index, avg_dist in averaged_distributions.items():
                r_val, p_val = index
                for i, avg_rho_c in enumerate(avg_dist):
                    final_rows.append({
                        'r': r_val,
                        varying_param_name: p_val,
                        'sample_step': i,
                        'averaged_rho_c': avg_rho_c
                    })
            
            final_df = pd.DataFrame(final_rows)
            
            # 保存文件
            output_dir = os.path.join(PROJECT_ROOT, 'data', 'fig3')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{exp_id}.csv")
            final_df.to_csv(output_path, index=False)
            print(f"    -> Data saved to {output_path}")

if __name__ == "__main__":
    start_time = time.time()
    run_all_experiments_mpi()
    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.Get_rank() == 0:
        end_time = time.time()
        print(f"\n===== All experiments for Figure 3 complete. Total time: {end_time - start_time:.2f} seconds. =====")