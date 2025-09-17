# analysis/plot_figure_5.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.fft import fft, fftfreq

def plot_dynamics_panel(ax, data_path, title, transient_steps=1000):
    """绘制单个动力学行为子图，包括相空间轨迹和功率谱插图。"""
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        ax.text(0.5, 0.5, 'Data not found', ha='center', va='center')
        return

    # 舍弃暂态过程，只分析稳态后的数据
    df_steady = df[df['mc_step'] >= transient_steps].copy()
    if len(df_steady) < 2:
        ax.text(0.5, 0.5, 'Not enough steady-state data', ha='center', va='center')
        return

    # --- 主图：相空间轨迹 ---
    x = df_steady['rho_C']
    y = df_steady['rho_CD']
    
    # 用颜色渐变表示时间流逝
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm
    
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(x)))
    lc = LineCollection(segments, colors=colors, linewidth=0.7)
    
    ax.add_collection(lc)
    ax.set_xlim(x.min() - 0.05, x.max() + 0.05)
    ax.set_ylim(y.min() - 0.05, y.max() + 0.05)
    ax.set_xlabel('Cooperator Density (ρ_C)', fontsize=10)
    ax.set_ylabel('Boundary Density (ρ_CD)', fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

    # --- 插图：功率谱密度 ---
    ax_inset = ax.inset_axes([0.65, 0.65, 0.3, 0.3])
    
    signal = df_steady['rho_C'].values
    N = len(signal)
    T = 1.0  # 采样间隔为1个MC步
    
    yf = fft(signal - np.mean(signal)) # 减去均值
    xf = fftfreq(N, T)[:N//2]
    power = 2.0/N * np.abs(yf[0:N//2])
    
    ax_inset.plot(xf, power, color='crimson')
    ax_inset.set_yscale('log')
    ax_inset.set_title('Power Spectrum', fontsize=8)
    ax_inset.set_xlabel('Frequency', fontsize=7)
    ax_inset.set_ylabel('Power', fontsize=7)
    ax_inset.tick_params(axis='both', which='major', labelsize=6)
    ax_inset.grid(True, linestyle='--', alpha=0.5)

def create_figure_5():
    """创建并保存 Figure 5。"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    
    fig.suptitle('Figure 5: Phase Space Portraits of Different Dynamical Regimes', fontsize=18, y=1.02)

    plot_info = [
        {'ax': axes[0], 'path': 'data/fig5/fig5_combo_a_stable_timeseries.csv', 'title': '(a) Stable Equilibrium'},
        {'ax': axes[1], 'path': 'data/fig5/fig5_combo_b_limit_cycle_timeseries.csv', 'title': '(b) Limit Cycle Oscillation'},
        {'ax': axes[2], 'path': 'data/fig5/fig5_combo_c_complex_timeseries.csv', 'title': '(c) Complex / Chaotic-like Behavior'}
    ]

    for info in plot_info:
        # 确定暂态步数，对于复杂行为案例，让它更长
        transient = 2500 if 'complex' in info['path'] else 1000
        plot_dynamics_panel(info['ax'], info['path'], info['title'], transient_steps=transient)
        
    plt.tight_layout()
    
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, 'Figure_5.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure 5 saved to {fig_path}")
    plt.show()

if __name__ == "__main__":
    create_figure_5()