# src/run_fig5_simulations.py
import json
import os
import pandas as pd
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.simulation import Simulation

def run_simulation_from_config(config_path):
    """读取单个配置文件，运行仿真，并保存完整的时序数据。"""
    with open(config_path, 'r') as f:
        params = json.load(f)

    output_dir = os.path.join('data', 'fig5')
    os.makedirs(output_dir, exist_ok=True)
    
    sim = Simulation(params)
    time_steps, rho_C_history, L_CD_history = sim.run()
    
    df = pd.DataFrame({
        'mc_step': time_steps,
        'rho_C': rho_C_history,
        'rho_CD': np.array(L_CD_history) / (2 * params['L']**2) # 归一化边界密度
    })
    
    combo_id = params['combo_id']
    output_path = os.path.join(output_dir, f"{combo_id}_timeseries.csv")
    df.to_csv(output_path, index=False)
    print(f"Data for {combo_id} saved to {output_path}")

if __name__ == "__main__":
    config_files = [
        'config/fig5_combo_a_stable.json',
        'config/fig5_combo_b_limit_cycle.json',
        'config/fig5_combo_c_complex.json'
    ]
    
    for config_file in config_files:
        run_simulation_from_config(config_file)
        
    print("\nAll simulations for Figure 5 are complete.")