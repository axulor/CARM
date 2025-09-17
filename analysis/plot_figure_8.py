# analysis/plot_figure_8.py  — publication-ready (wider panels, tighter gaps, left-only ticks, tinted textboxes, larger markers)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['text.usetex'] = False

# ======= 配色（柔和柱色 + 冷峻深色折线）=======
BAR_C = '#0CA678'   # cooperators 环境频率 p(s|C)
BAR_D = '#F59F00'   # defectors  环境频率 p(s|D)
LINE_C = '#4263EB'  # 折线：P(C→D|s)
LINE_D = '#C92A2A'  # 折线：P(D→C|s)
MARKER_C = 'o'
MARKER_D = 'X'

# ======= 字号 =======
AXLABEL_FT = 24
TICK_FT    = 22
TITLE_FT   = 20
LEGEND_FT  = 26

# 文本框底色
TEXTBOX_FC = '#F7E7CC'   # 要求的底色
TEXTBOX_EC = '#D8C5A3'   # 边框微暖

def _textbox_xy(panel_idx: int):
    """按面板索引返回文本框在轴坐标下的位置 (x, y)。"""
    if panel_idx == 0:        # 子图1：略微下移
        return (0.97, 0.90)
    elif panel_idx == 1:      # 子图2：下移更多
        return (0.97, 0.72)
    else:                     # 子图3：向左移防遮挡
        return (0.44, 0.94)

def _draw_one_panel(ax, plot_df, summary, r_val, panel_idx, add_left_ylabel=False):
    """
    单个面板：
      - 左轴：环境频率柱状（p(s|C), p(s|D)）
      - 右轴：转换概率折线（P(C→D|s), P(D→C|s)）；但右轴刻度标签全部隐藏（统一只用左刻度）
    """
    s = plot_df['num_coop_neighbors'].to_numpy()

    # --- 背景柱状图（左轴） ---
    bw = 0.4
    ax.bar(s - bw/2, plot_df['p_s_given_C'], width=bw, color=BAR_C, alpha=0.75,
           label='p(s|C)', zorder=1)
    ax.bar(s + bw/2, plot_df['p_s_given_D'], width=bw, color=BAR_D, alpha=0.75,
           label='p(s|D)', zorder=1)

    # --- 前景折线（右轴） ---
    ax2 = ax.twinx()
    # 标记稍大一点
    ax2.plot(s, plot_df['prob_C_to_D_s'], color=LINE_C, marker=MARKER_C, ms=10,
             linestyle='--', linewidth=2.2, label=r'$\langle P(C \to D\,|\,s)\rangle_C$', zorder=5)
    ax2.plot(s, plot_df['prob_D_to_C_s'], color=LINE_D, marker=MARKER_D, ms=10,
             linestyle='--', linewidth=2.2, label=r'$\langle P(D \to C\,|\,s)\rangle_D$', zorder=5)

    # 右轴刻度标签全部取消（统一使用左刻度）
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax2.tick_params(axis='y', labelsize=TICK_FT)
    plt.setp(ax2.get_yticklabels(), visible=False)

    # --- 文本框（按面板微调位置） ---
    p_c_d = float(summary['p_c_to_d_observed'])
    p_d_c = float(summary['p_d_to_c_observed'])
    rho_c = float(summary['rho_c_observed'])
    text = (
        rf"$\bf{{r={r_val}}}$" + "\n\n"
        rf"$P(C\!\to\!D):\ {p_c_d:.3f}$" + "\n"
        rf"$P(D\!\to\!C):\ {p_d_c:.3f}$" + "\n\n"
        rf"$\rho_C = \frac{{{p_d_c:.3f}}}{{{p_d_c:.3f}+{p_c_d:.3f}}} = {rho_c:.3f}$"
    )
    x_text, y_text = _textbox_xy(panel_idx)
    ax.text(x_text, y_text, text, transform=ax.transAxes, fontsize=TITLE_FT,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.55', fc=TEXTBOX_FC, ec=TEXTBOX_EC, alpha=0.95))

    # --- 轴/刻度/范围（左轴） ---
    ax.set_xlabel('Number of Cooperative Neighbors (s)', fontsize=AXLABEL_FT, labelpad=16)
    ax.set_xticks(range(5))
    ax.tick_params(axis='x', labelsize=TICK_FT)

    # 让 0 刻度不贴边：下界微微小于 0
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.tick_params(axis='y', labelsize=TICK_FT)

    # 只给第一个子图加左轴标题，其余保留左刻度但不重复标题
    if add_left_ylabel:
        ax.set_ylabel('Probability or Frequency', fontsize=AXLABEL_FT, labelpad=16)

def create_figure_8():
    data_path = os.path.join('data', 'fig8', 'fig8_final_verification_data.csv')
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        print(f"[Error] Missing data file: {e.filename}")
        return

    # 聚合：每个 r、每个 s 的均值；以及每个 r 的总体均值
    plot_df = (df.groupby(['r', 'num_coop_neighbors']).mean(numeric_only=True)
                 .reset_index())
    df_means = df.groupby('r').mean(numeric_only=True).reset_index()

    r_list = [2.2, 3.0, 3.8]

    # 画布更宽，子图间距更紧
    fig, axes = plt.subplots(1, 3, figsize=(36, 9), sharey=False)
    plt.subplots_adjust(wspace=0.10, top=0.86)

    for j, (ax, r_val) in enumerate(zip(axes, r_list)):
        _draw_one_panel(
            ax=ax,
            plot_df=plot_df[plot_df['r'] == r_val],
            summary=df_means[df_means['r'] == r_val].iloc[0],
            r_val=r_val,
            panel_idx=j,
            add_left_ylabel=(j == 0)  # 只有第一个子图写“Probability or Frequency”
        )

    # -------- 顶部图例（顺序：先两柱，再两折线） --------
    legend_handles = [
        Patch(facecolor=BAR_C, alpha=0.75, label='p(s|C)'),
        Patch(facecolor=BAR_D, alpha=0.75, label='p(s|D)'),
        Line2D([0], [0], color=LINE_C, marker=MARKER_C, linestyle='--', linewidth=2.4, markersize=11,
               label=r'$\langle P(C \to D\,|\,s)\rangle_C$'),
        Line2D([0], [0], color=LINE_D, marker=MARKER_D, linestyle='--', linewidth=2.4, markersize=11,
               label=r'$\langle P(D \to C\,|\,s)\rangle_D$'),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles],
               loc='upper center', ncol=4, fontsize=LEGEND_FT, frameon=False)

    # 保存
    out_dir = 'figures'
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, 'Figure_8_Publication_Ready.png')
    out_pdf = os.path.join(out_dir, 'Figure_8_Publication_Ready.pdf')
    plt.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved:\n  {out_png}\n  {out_pdf}")

if __name__ == "__main__":
    create_figure_8()
