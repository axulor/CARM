import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from matplotlib.colors import to_rgb, to_hex  # 用于生成冷峻梯度配色

PALETTE_ARCTIC = {
    "ink": "#0F172A",
    "slate": "#334155",
    "mist": "#E5E7EB",
    "steel": "#4263EB",
    "teal":  "#0CA678",
    "indigo":"#5F3DC4",
    "crimson":"#C92A2A",
    "amber": "#F59F00",
}

def _mix_hex(hex1, hex2, w):
    """线性混色：w=0→hex1, w=1→hex2。"""
    c1 = np.array(to_rgb(hex1))
    c2 = np.array(to_rgb(hex2))
    c = (1 - w) * c1 + w * c2
    return to_hex(c, keep_alpha=False)

def _build_sequential_palette(base_hex, n=4):
    """
    生成 4 级序列色：
    1、2：与白色混合（浅色）
    3：原色
    4：与黑色混合（加深）
    """
    tint_w1 = 0.40
    tint_w2 = 0.20
    shade_w = 0.15
    seq = [
        _mix_hex(base_hex, "#FFFFFF", tint_w1),  # 浅
        _mix_hex(base_hex, "#FFFFFF", tint_w2),  # 次浅
        base_hex,                                # 原色
        _mix_hex(base_hex, "#000000", shade_w),  # 加深
    ]
    return seq[:n]

def plot_figure_facet(
    data_path,
    output_path,
    x_var,
    hue_var,
    legend_title,
    palette,
    hline_at_x=None,                 # 若指定，则在指定 x 值处画均值水平线
    hline_only_for_hue_value=None,   # 仅在某个 hue 值（如 κ=0）对应的子图上画
    hline_color="#3B82F6",           # 更亮的蓝色
    hline_width=2.2                  # 稍粗
):
    """
    为单个CSV文件绘制一个4x1的堆叠子图（Facet Grid），每个子图展示一个参数值。
    若 hline_at_x 指定，则对满足 hline_only_for_hue_value 的那个分面，在该 x 的 averaged_rho_c
    均值处绘制一条水平虚线。
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"!!! Data file not found: {data_path}. Skipping plot.")
        return

    df[x_var] = df[x_var].round(2)

    unique_hues = sorted(df[hue_var].unique())
    n_hues = len(unique_hues)

    fig, axes = plt.subplots(n_hues, 1, figsize=(25, 5 * n_hues), sharex=True)
    if n_hues == 1:
        axes = [axes]

    for i, hue_value in enumerate(unique_hues):
        ax = axes[i]
        subset_df = df[df[hue_var] == hue_value].copy()

        # 添加微小的随机噪声 ("jitter")
        jitter_strength = subset_df['averaged_rho_c'].std() * 0.05
        if pd.isna(jitter_strength) or jitter_strength == 0:
            jitter_strength = 0.001
        subset_df['plot_rho_c'] = subset_df['averaged_rho_c'] + np.random.normal(
            0, jitter_strength, size=len(subset_df)
        )
        subset_df['plot_rho_c'] = subset_df['plot_rho_c'].clip(0, 1)

        # 小提琴图
        sns.violinplot(
            data=subset_df,
            x=x_var,
            y='plot_rho_c',
            color=palette[i],
            scale='width',
            inner=None,
            cut=0,
            linewidth=1.5,
            ax=ax
        )

        # 分面角标
        ax.text(
            0.02, 0.95, f'{legend_title.split(" ")[0]} = {hue_value}',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=30,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8)
        )

        # 轴样式
        ax.set_ylabel(r'$\rho_C$', fontsize=30, fontweight='bold', labelpad=24)
        ax.tick_params(axis='y', labelsize=22)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, which='both', linestyle=':', linewidth=1.0, alpha=0.7)
        ax.set_xlabel('')

        # —— 仅在指定的 hue 值（如 κ=0）上画水平虚线 —— #
        if (hline_at_x is not None) and \
           (hline_only_for_hue_value is not None) and \
           np.isclose(float(hue_value), float(hline_only_for_hue_value)):
            mask = subset_df[x_var].round(2).eq(round(hline_at_x, 2))
            if mask.any():
                mean_at_x = subset_df.loc[mask, 'averaged_rho_c'].mean()
                ax.axhline(
                    mean_at_x,
                    color=hline_color,
                    linestyle='--',
                    linewidth=hline_width,
                    alpha=0.95
                )

    # 统一 X 轴
    axes[-1].set_xlabel(r'$r$', fontsize=34, labelpad=15)
    axes[-1].tick_params(axis='x', rotation=45, labelsize=22, pad=10)

    plt.subplots_adjust(hspace=0.1)

    # 导出 PNG & PDF
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {output_path} and {pdf_path}")

def generate_all_facet_figures():
    plt.style.use('seaborn-v0_8-whitegrid')

    # 冷峻配色基色
    steel_base  = "#4263EB"  # β
    teal_base   = "#0CA678"  # γ
    indigo_base = "#C92A2A"  # κ

    palette_beta  = _build_sequential_palette(steel_base,  n=4)
    palette_gamma = _build_sequential_palette(teal_base,   n=4)
    palette_kappa = _build_sequential_palette(indigo_base, n=4)

    output_dir = os.path.join(PROJECT_ROOT, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    data_root = os.path.join(PROJECT_ROOT, 'data', 'fig3')

    # Figure 3a：不改动
    plot_figure_facet(
        data_path=os.path.join(data_root, 'fig3a_vary_beta_g0.5_k0.5.csv'),
        output_path=os.path.join(output_dir, 'Figure_3a_Facet.png'),
        x_var='r', hue_var='BETA', legend_title='β', palette=palette_beta,
        hline_at_x=None
    )

    # Figure 3b：不改动
    plot_figure_facet(
        data_path=os.path.join(data_root, 'fig3b_vary_gamma_b0.5_k0.5.csv'),
        output_path=os.path.join(output_dir, 'Figure_3b_Facet.png'),
        x_var='r', hue_var='GAMMA', legend_title='γ', palette=palette_gamma,
        hline_at_x=None
    )

    # Figure 3c：仅在 κ=0 的子图，于 r=2.2 的均值处添加更亮、更粗的蓝色虚线
    plot_figure_facet(
        data_path=os.path.join(data_root, 'fig3c_vary_kappa_b0.5_g0.5.csv'),
        output_path=os.path.join(output_dir, 'Figure_3c_Facet.png'),
        x_var='r', hue_var='KAPPA', legend_title='κ', palette=palette_kappa,
        hline_at_x=2.2,
        hline_only_for_hue_value=0.0,     # 只在 κ=0 的分面子图上画
        hline_color="#3B82F6",            # 稍亮的蓝色
        hline_width=2.2                   # 稍粗
    )

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    sys.path.insert(0, PROJECT_ROOT)
    generate_all_facet_figures()
    print("\nAll facet plot figures for Figure 3 generated.")
