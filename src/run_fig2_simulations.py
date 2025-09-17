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

        # 2. 创建输出目录 (线程安全)
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
    # 这一行至关重要，它保护了主程序入口点。
    # 在Windows和macOS上，multiprocessing会重新导入主脚本来启动子进程，
    # 如果没有这个保护，会无限递归地创建新进程导致崩溃。
    
    start_time = time.time()
    
    # 定义要运行的配置文件列表
    config_files = [
        # 'config/fig2_combo1_low_coop.json',
        'config/fig2_combo2_high_coop.json',
        # 'config/fig2_combo3_oscillation.json'
    ]
    
    # 使用 ProcessPoolExecutor 来并行运行任务
    # max_workers=None 会自动使用机器上所有可用的CPU核心
    # 在HPC上，您可能需要根据分配给您的核心数来设置它，例如 max_workers=8
    max_cores = os.cpu_count()
    print(f"Starting parallel simulations on up to {max_cores} cores...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        # executor.map 会将 config_files 列表中的每个元素
        # 作为参数传递给 run_simulation_from_config 函数，并并行执行它们。
        # 它会阻塞，直到所有任务完成。
        results = executor.map(run_simulation_from_config, config_files)

        # 检查每个任务的结果（可选，但有助于调试）
        for result in results:
            if "Error" in result:
                print(f"A simulation task failed: {result}")

    end_time = time.time()
    
    print("\n" + "="*40)
    print("All simulations for Figure 2 are complete.")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
    print("="*40)