# analysis/plot_figure_2_final.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

def adaptive_downsample(x, y):
    """
    智能的对数感知降采样函数。
    """
    if len(x) < 500:
        return x, y

    log_x = np.log10(x)
    num_bins = int(len(x) / 10)
    bins = np.logspace(log_x.min(), log_x.max(), num=num_bins)

    df = pd.DataFrame({'x': x, 'y': y})
    df['bin'] = pd.cut(df['x'], bins=bins, labels=False, include_lowest=True)
    downsampled_df = df.groupby('bin').mean()

    return downsampled_df['x'].to_numpy(), downsampled_df['y'].to_numpy()

def _nice_ceiling(x, step=1000):
    """把上界 x 向上取整到 step 的整数倍（默认 1000）。"""
    if x <= 0:
        return step
    return int(np.ceil(x / step) * step)

def plot_single_figure(
    data_path,
    output_path,
    is_oscillation=False,
    L_CD_ylim=None,      # 统一设置的 L_CD y 轴范围 (min, max) —— 这里传入“带边距的范围”
    L_CD_yticks=None,    # 统一设置的 L_CD y 轴刻度（数组）
    legend_loc=None,     # 可选：图例位置
    legend_bbox=None     # 可选：图例 bbox_to_anchor
):
    """
    绘制并保存单个、精美的子图。
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return

    # --- 1. 数据准备 ---
    df_t0 = df.loc[df['mc_step'] == 0].iloc[0]
    df_t_positive = df[df['mc_step'] > 0].copy()

    x_pos  = df_t_positive['mc_step'].to_numpy()
    y_rho  = df_t_positive['rho_C'].to_numpy()
    y_lcd  = df_t_positive['L_CD'].to_numpy()

    # --- 2. 先降采样，后连接 ---
    if not is_oscillation:
        dx, dy_rho = adaptive_downsample(x_pos, y_rho)
        _,  dy_lcd = adaptive_downsample(x_pos, y_lcd)
        alpha = 1.0
    else:
        dx, dy_rho, dy_lcd = x_pos, y_rho, y_lcd
        alpha = 0.7

    plot_x     = np.insert(dx, 0, 1)
    plot_y_rho = np.insert(dy_rho, 0, df_t0['rho_C'])
    plot_y_lcd = np.insert(dy_lcd, 0, df_t0['L_CD'])

    # --- 3. 绘图设置 ---
    fig, ax = plt.subplots(figsize=(10, 8))
    color1, color2 = 'tab:blue', 'tab:red'

    line1, = ax.plot(plot_x, plot_y_rho, color=color1,
                     label=r'Cooperator ratio $\rho_C$', alpha=alpha, zorder=10)

    ax2 = ax.twinx()
    line2, = ax2.plot(plot_x, plot_y_lcd, color=color2,
                      label=r'C-D Boundary Length $L_{CD}$', alpha=alpha, zorder=10)

    # --- 4. 坐标轴与格式化 ---
    ax.set_xscale('log')
    # ax.set_xlabel('Monte Carlo Steps (Log scale)', fontsize=22)

    ax.set_ylabel(r'Cooperator ratio $\rho_C$', color=color1, fontsize=22)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)

    ax2.set_ylabel(r'C-D Boundary Length $L_{CD}$', color=color2, fontsize=22)
    ax2.tick_params(axis='y', labelsize=20)

    # —— 统一的 L_CD 刻度/范围（对三张图都一致）——
    if L_CD_ylim is not None:
        ax2.set_ylim(L_CD_ylim)      # 这里的范围已包含上下 4% 边距
    else:
        ax2.set_ylim(bottom=0)

    if L_CD_yticks is not None:
        ax2.set_yticks(L_CD_yticks)

    # --- 5. 图例和网格 ---
    lines  = [line1, line2]
    labels = [l.get_label() for l in lines]

    if legend_loc is not None:
        ax.legend(lines, labels, loc=legend_loc,
                  bbox_to_anchor=legend_bbox if legend_bbox is not None else None,
                  borderaxespad=0.0, fontsize=20)
    else:
        if is_oscillation:
            # 振荡图把图例从右上角向左移动一些
            ax.legend(lines, labels, loc='upper right',
                      bbox_to_anchor=(0.68, 0.98), borderaxespad=0.0, fontsize=20)
        else:
            ax.legend(lines, labels, loc='best', fontsize=20)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- 6. 保存图像 ---
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_all_final_figures():
    """
    统一三张图的 L_CD y 轴刻度样式：
    1) 读取三份数据，拿到三个 L_CD 的全局最大值；
    2) 把上界 nice 到 1000 的整数倍；
    3) 统一刻度：yticks = np.linspace(0, y_max_nice, 5)；
    4) 为避免刻度贴边框，三图 y 限统一加“上下 4%”边距；
    5) 2b 的图例放右下角但内缩（不贴边），2c 的图例左移，其余不变。
    """
    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        '2a': 'data/fig2/combo1_low_coop_timeseries.csv',
        '2b': 'data/fig2/combo2_high_coop_timeseries.csv',
        '2c': 'data/fig2/combo3_oscillation_timeseries.csv',
    }

    # —— 计算全局 L_CD 最大值 —— #
    lcd_max_list = []
    for p in paths.values():
        df_tmp = pd.read_csv(p)
        lcd_max_list.append(df_tmp['L_CD'].max())
    global_max = np.nanmax(lcd_max_list)
    y_max_nice = _nice_ceiling(global_max, step=1000)  # 可按需改步长

    # 统一刻度（不含边距）
    common_yticks = np.linspace(0, y_max_nice, 5)

    # 统一 y 轴范围（加入上 4% 的边距，避免刻度贴边）
    pad = 0.04 * y_max_nice
    common_ylim_padded = (0, y_max_nice + pad)

    # —— 绘制三张图（统一的 ylim/yticks）—— #
    # 2a
    plot_single_figure(
        data_path=paths['2a'],
        output_path=os.path.join(output_dir, 'Figure_2a_low_coop_final.png'),
        is_oscillation=False,
        L_CD_ylim=common_ylim_padded,
        L_CD_yticks=common_yticks,
        legend_loc=None,
        legend_bbox=None
    )

    # 2b（图例右下角但内缩，避免贴边）
    plot_single_figure(
        data_path=paths['2b'],
        output_path=os.path.join(output_dir, 'Figure_2b_high_coop_final.png'),
        is_oscillation=False,
        L_CD_ylim=common_ylim_padded,
        L_CD_yticks=common_yticks,
        legend_loc='lower right',
        legend_bbox=(0.94, 0.08)   # 向内缩一点；可微调如 (0.94, 0.08)
    )

    # 2c（图例左移，保持之前风格）
    plot_single_figure(
        data_path=paths['2c'],
        output_path=os.path.join(output_dir, 'Figure_2c_oscillation_final.png'),
        is_oscillation=True,
        L_CD_ylim=common_ylim_padded,
        L_CD_yticks=common_yticks,
        legend_loc=None,
        legend_bbox=None
    )

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning,
                            message="Attempt to set non-positive xlim on a log-scaled axis")
    generate_all_final_figures()
    print("\nAll final figures generated.")
