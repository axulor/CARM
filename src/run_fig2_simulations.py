# src/run_fig2_simulations_parallel.py
import json
import os
import pandas as pd
import sys
import time
import concurrent.futures

# 确保可以从src/core导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.simulation import Simulation

def run_simulation_from_config(config_path):
    """
    一个独立的“工作单元”函数。
    它读取单个配置文件，运行完整的仿真，并保存结果。
    这是将被并行执行的任务。
    """
    try:
        # 1. 读取配置
        with open(config_path, 'r') as f:
            params = json.load(f)
        
        combo_id = params['combo_id']
        print(f"[Process {os.getpid()}] Starting simulation for: {combo_id}")

        # 2. 创建输出目录
        output_dir = os.path.join('data', 'fig2')
        os.makedirs(output_dir, exist_ok=True)
        
        # 3. 运行仿真
        sim = Simulation(params)
        time_steps, rho_C_history, L_CD_history = sim.run()
        
        # 4. 整理并保存数据
        df = pd.DataFrame({
            'mc_step': time_steps,
            'rho_C': rho_C_history,
            'L_CD': L_CD_history
        })
        
        output_path = os.path.join(output_dir, f"{combo_id}_timeseries.csv")
        df.to_csv(output_path, index=False)
        
        result_message = f"[Process {os.getpid()}] Finished simulation for {combo_id}. Data saved to {output_path}"
        print(result_message)
        return result_message

    except Exception as e:
        error_message = f"Error running simulation for {config_path}: {e}"
        print(error_message)
        return error_message


if __name__ == "__main__":
    
    start_time = time.time()
    
    # 定义要运行的配置文件列表
    config_files = [
        # 'config/fig2_combo1_low_coop.json',
        'config/fig2_combo2_high_coop.json',
        # 'config/fig2_combo3_oscillation.json'
    ]
    
    # 使用 ProcessPoolExecutor 来并行运行任务
    max_cores = os.cpu_count()
    print(f"Starting parallel simulations on up to {max_cores} cores...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:

        results = executor.map(run_simulation_from_config, config_files)

        for result in results:
            if "Error" in result:
                print(f"A simulation task failed: {result}")

    end_time = time.time()
    
    print("\n" + "="*40)
    print("All simulations for Figure 2 are complete.")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

    print("="*40)

