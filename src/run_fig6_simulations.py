# src/run_fig6_simulations.py
import json
import os
import pandas as pd
import numpy as np
import sys
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.simulation import Simulation

def run_simulation(params, seed=None, capture_times=None):
    """
    通用仿真函数。
    - 强制 MAX_MC_STEPS = 2000
    - 仅按给定 capture_times 抓取快照
    - timeseries 仍然按原逻辑收集（用于一致性）
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 强制步长为 2000
    params = dict(params)  # 避免原地修改传入 dict
    params['MAX_MC_STEPS'] = 2000
    if 'RECORD_INTERVAL_MC' not in params or params['RECORD_INTERVAL_MC'] <= 0:
        params['RECORD_INTERVAL_MC'] = 1

    sim = Simulation(params)

    # --- 快照数据容器 ---
    snapshot_data = {}

    # t=0 的初始状态快照（策略 + 遗憾置零）
    if capture_times is not None and 0 in capture_times:
        strategy_grid = np.array([[1 if agent.action == 'C' else 0 for agent in row] for row in sim.grid])
        regret_grid = np.zeros_like(strategy_grid, dtype=float)
        snapshot_data['strategy_0'] = strategy_grid
        snapshot_data['regret_0'] = regret_grid

    # --- 主循环 ---
    for step in range(1, params['MAX_MC_STEPS'] + 1):
        next_actions = [agent.decide_next_action() for agent in sim.agents]
        for agent, na in zip(sim.agents, next_actions):
            agent.action = na

        # 指定时刻保存快照
        if capture_times is not None and step in capture_times:
            strategy_grid = np.array([[1 if agent.action == 'C' else 0 for agent in row] for row in sim.grid])
            regret_grid = np.array([[agent.instant_regret for agent in row] for row in sim.grid])
            snapshot_data[f'strategy_{step}'] = strategy_grid
            snapshot_data[f'regret_{step}'] = regret_grid

        # 记录宏观数据
        if step % params['RECORD_INTERVAL_MC'] == 0:
            sim._collect_metrics(step)

    # 记录 t=0 的宏观数据（与原脚本一致）
    sim._collect_metrics(0)

    # timeseries 与原格式一致（rho_CD 仍按归一化处理）
    timeseries = pd.DataFrame({
        'mc_step': sim.time_steps,
        'rho_C': sim.history_rho_C,
        'rho_CD': np.array(sim.history_L_CD) / (2 * params['L']**2)
    }).sort_values('mc_step').reset_index(drop=True)

    return timeseries, snapshot_data

def process_config(config_path, seed=42):
    """
    读取配置并运行一次仿真，固定 2000 步，
    快照固定为 [0, 10, 100, 1998, 1999, 2000] 六个时刻。
    保存 npz，键名与原脚本保持一致。
    """
    print(f"\n--- Processing config: {os.path.basename(config_path)} ---")
    with open(config_path, 'r') as f:
        params = json.load(f)

    # 固定的 6 个快照时刻
    capture_times = [0, 10, 100, 1998, 1999, 2000]

    # 单次运行直接捕捉
    _, captured_data = run_simulation(params, seed=seed, capture_times=capture_times)

    # 保存 npz（文件名与原脚本一致）
    output_dir = os.path.join('data', 'fig6')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{params['combo_id']}_snapshots.npz")

    # 保存并带上 snapshot_times（int 数组）
    np.savez(output_path, **captured_data, snapshot_times=np.array(capture_times, dtype=int))
    print(f"  -> Snapshot data saved to {output_path}")

if __name__ == "__main__":
    config_files = [
        'config/fig5_combo_a_stable.json',
        'config/fig5_combo_b_limit_cycle.json',
        'config/fig5_combo_c_complex.json'
    ]

    master_seed = 12345  # 固定随机种子以保证可复现
    for cfg in config_files:
        process_config(cfg, seed=master_seed)

    print("\nAll snapshot generation for Figure 6 is complete.")
