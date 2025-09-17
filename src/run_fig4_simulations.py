import json
import os
import pandas as pd
import numpy as np
import sys
from itertools import product
from mpi4py import MPI
import time

# 确保可以从src/core导入模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
from src.core.simulation import Simulation

def run_single_replica(task_info):
    """
    执行单次独立重复仿真，并返回完整的采样序列。
    这是MPI并行的最小工作单元。
    """
    params, run_id, exp_id, x_param_name, y_param_name = task_info
    
    sim = Simulation(params)
    
    # 预热阶段
    for _ in range(params['TRANSIENT_MC_STEPS']):
        next_actions = [agent.decide_next_action() for agent in sim.agents]
        for agent, next_action in zip(sim.agents, next_actions):
            agent.action = next_action

    # 采样阶段
    samples = np.zeros(params['SAMPLING_MC_STEPS'])
    for step in range(params['SAMPLING_MC_STEPS']):
        next_actions = [agent.decide_next_action() for agent in sim.agents]
        for agent, next_action in zip(sim.agents, next_actions):
            agent.action = next_action
        
        num_cooperators = sum(1 for agent in sim.agents if agent.action == 'C')
        samples[step] = num_cooperators / sim.N

    # 返回原始采样序列和所有必要的标识符
    return {
        'exp_id': exp_id,
        x_param_name: params[x_param_name],
        y_param_name: params[y_param_name],
        'run_id': run_id,
        'sample_series': samples
    }

def get_axis_values(axis_info):
    """根据配置生成参数轴的值"""
    if axis_info['scale'] == 'log':
        return np.logspace(np.log10(axis_info['min']), np.log10(axis_info['max']), axis_info['num'])
    else: # linear
        return np.linspace(axis_info['min'], axis_info['max'], axis_info['num'])

def run_all_experiments_mpi():
    """
    读取所有Fig4配置文件，生成总任务列表，用MPI并行运行，
    最后由0号进程聚合所有采样序列，计算最终统计量并保存。
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- 1. Rank 0: 生成总任务列表 ---
    all_tasks = None
    if rank == 0:
        print(f"===== Rank 0: Generating massive task list for Figure 4 on {size} total workers... =====")
        configs_to_run = [
            'config/fig4a_beta_vs_gamma.json',
            'config/fig4b_gamma_vs_kappa.json',
            'config/fig4c_kappa_vs_beta.json',
        ]
        config_paths = [os.path.join(PROJECT_ROOT, conf) for conf in configs_to_run]
        
        task_list = []
        for config_path in config_paths:
            with open(config_path, 'r') as f: config = json.load(f)
            exp_id = config['experiment_id']
            n_replicas = config['N_REPLICAS']
            
            axis_keys = [k for k in config if k.endswith('_AXIS')]
            x_axis_key, y_axis_key = axis_keys[0], axis_keys[1]
            x_param_name, y_param_name = x_axis_key.replace('_AXIS', ''), y_axis_key.replace('_AXIS', '')
            x_values, y_values = get_axis_values(config[x_axis_key]), get_axis_values(config[y_axis_key])

            for x_val, y_val in product(x_values, y_values):
                for i in range(n_replicas):
                    params = config.copy()
                    params[x_param_name], params[y_param_name] = x_val, y_val
                    task_list.append((params, i, exp_id, x_param_name, y_param_name))
        
        all_tasks = task_list
        print(f"Total individual replica simulations to run: {len(all_tasks)}")

    # --- 2. 广播任务列表给所有进程 ---
    all_tasks = comm.bcast(all_tasks, root=0)

    # --- 3. 每个进程计算自己的任务子集 ---
    my_tasks = [task for i, task in enumerate(all_tasks) if i % size == rank]
    
    local_results = []
    for i, task in enumerate(my_tasks):
        # *** 核心修复：确保调用正确的、已定义的函数名 ***
        result = run_single_replica(task)
        local_results.append(result)
        if rank == 0 and (i + 1) % 5 == 0:
             print(f"  Rank 0 heartbeat: completed {i+1}/{len(my_tasks)} local tasks.", flush=True)

    # --- 4. 收集所有结果到0号进程 ---
    gathered_results = comm.gather(local_results, root=0)

    # --- 5. Rank 0: 聚合和保存 ---
    if rank == 0:
        print("\n===== Rank 0: All simulations finished. Aggregating final statistics... =====")
        
        if not gathered_results or not any(s for s in gathered_results if s):
            print("!!! WARNING: No results were gathered from worker processes. Exiting.")
            return
            
        flat_results = [item for sublist in gathered_results for item in sublist]
        
        if not flat_results:
            print("!!! WARNING: Result list is empty after flattening. Exiting.")
            return

        df = pd.DataFrame(flat_results)
        
        # 定义聚合函数：将一个组内所有采样序列合并成一个大数组
        def combine_all_samples(series):
            if series.dropna().empty:
                return np.array([]) # 返回空数组而不是None
            return np.concatenate(series.dropna().tolist())

        # 按实验ID和参数点分组
        param_names_all = [col for col in df.columns if col not in ['exp_id', 'run_id', 'sample_series']]
        
        for exp_id, exp_df in df.groupby('exp_id'):
            print(f"  Processing and saving experiment: {exp_id}")

            exp_df_cleaned = exp_df.dropna(axis=1, how='all')
            param_names_cleaned = [col for col in exp_df_cleaned.columns if col not in ['exp_id', 'run_id', 'sample_series']]
            
            if len(param_names_cleaned) != 2:
                print(f"  !!! ERROR: Could not determine correct 2 parameter columns for {exp_id}. Found: {param_names_cleaned}. Skipping.")
                continue

            # 合并所有采样点
            combined_series_df = exp_df_cleaned.groupby(param_names_cleaned)['sample_series'].apply(combine_all_samples)
            
            # 计算最终统计量
            final_stats = pd.DataFrame({
                'mean_rho_c': combined_series_df.apply(np.mean),
                'var_rho_c': combined_series_df.apply(np.var),
                'std_rho_c': combined_series_df.apply(np.std)
            }).reset_index()

            final_stats['cv_rho_c'] = final_stats['std_rho_c'] / final_stats['mean_rho_c'].replace(0, 1e-9)
            final_stats.fillna(0, inplace=True)
            
            columns_to_save = param_names_cleaned + ['mean_rho_c', 'var_rho_c', 'std_rho_c', 'cv_rho_c']
            df_to_save = final_stats[columns_to_save]
            
            output_dir = os.path.join(PROJECT_ROOT, 'data', 'fig4_final_stats_final')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{exp_id}_data.csv")
            df_to_save.to_csv(output_path, index=False)
            print(f"    -> Data saved to {output_path}")

if __name__ == "__main__":
    start_time = time.time()
    run_all_experiments_mpi()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm.Barrier()
    if rank == 0:
        end_time = time.time()
        print(f"\n===== All experiments for Figure 4 complete. Total time: {end_time - start_time:.2f} seconds. =====")