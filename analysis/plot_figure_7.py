# analysis/plot_figure_7.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----- 基础风格 -----
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['text.usetex'] = False

# ----- 冷峻配色（与前文一致）-----
COLOR_STEEL   = '#649EC7'   # 蓝
COLOR_TEAL    = '#21A57B'   # 绿（青绿）
COLOR_CRIMSON = '#B14C59'   # 红

# ---------- 图例标签 ----------
def _label_from_params(df_one):
    """优先从列里读 beta/gamma/kappa，回退用 combo_id 对应的文本。"""
    def _pick(cols):
        for c in cols:
            if c in df_one.columns:
                return df_one[c].iloc[0]
        return None

    beta  = _pick(['beta', 'BETA'])
    gamma = _pick(['gamma', 'GAMMA'])
    kappa = _pick(['kappa', 'KAPPA'])
    if beta is not None and gamma is not None and kappa is not None:
        return rf'$\beta={beta:.1f},\ \gamma={gamma:.1f},\ \kappa={kappa:.1f}$'

    cid = str(df_one['combo_id'].iloc[0])
    # 你当前的数据映射（保持一致）
    mapping = {
        'oscillation':  r'$\beta=0.2,\ \gamma=0.8,\ \kappa=0.3$',
        'high_coop':   r'$\beta=0.5,\ \gamma=0.5,\ \kappa=1.0$',
        'low_coop':    r'$\beta=0.5,\ \gamma=0.5,\ \kappa=0.2$',
    }
    return mapping.get(cid, cid)

def _is_oscillation_combo(cid: str) -> bool:
    return 'oscill' in str(cid).lower()

def plot_figure_7_final(data_csv, out_png, out_pdf):
    try:
        df = pd.read_csv(data_csv)
    except FileNotFoundError:
        print(f"[Error] data not found: {data_csv}")
        return

    # 子图：上下排列，共享 x（log）
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'hspace': 0.18}
    )

    # 统一字号
    AX_LABEL_FT = 20
    TICK_FT     = 17
    LEGEND_FT   = 17

    # 组合顺序：把振荡系列放在最先绘制（底层逻辑改为同一 zorder，通过 alpha 叠色）
    combo_ids = list(df['combo_id'].unique())
    combo_ids.sort(key=lambda c: 0 if _is_oscillation_combo(c) else 1)

    # 给每条曲线分配颜色（保持冷峻三色循环）
    colors = {}
    palette = [COLOR_STEEL, COLOR_TEAL, COLOR_CRIMSON]
    for i, cid in enumerate(combo_ids):
        colors[cid] = palette[i % len(palette)]

    # 用于强制图例顺序：绿 -> 蓝 -> 红
    legend_target_colors = [COLOR_TEAL, COLOR_CRIMSON, COLOR_STEEL ]
    legend_handle_by_color = {}

    # 通用绘图参数（保证叠色而非遮挡）
    line_kw = dict(lw=1.6, alpha=1.0, zorder=2,
                   solid_joinstyle='round', solid_capstyle='round')

    # ============ (a) ⟨P(C→C)⟩ ============
    for cid in combo_ids:
        sub = df[df['combo_id'] == cid].sort_values('step')
        label = _label_from_params(sub)

        x = sub['step'].to_numpy()
        y = sub['p_c_to_c'].to_numpy()

        color = colors[cid]
        h, = ax_top.plot(x, y, color=color, label=label, **line_kw)

        # 记录句柄，方便后面按颜色顺序重排图例
        legend_handle_by_color[color] = h

    ax_top.set_ylabel(r'$ P(C\!\to\!C)$', fontsize=AX_LABEL_FT, labelpad=10)
    ax_top.set_ylim(-0.02, 0.32)
    ax_top.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_top.tick_params(axis='both', labelsize=TICK_FT)

    # ============ (b) ⟨P(D→C)⟩ ============
    for cid in combo_ids:
        sub = df[df['combo_id'] == cid].sort_values('step')

        x = sub['step'].to_numpy()
        y = sub['p_d_to_c'].to_numpy()

        color = colors[cid]
        ax_bot.plot(x, y, color=color, label=_label_from_params(sub), **line_kw)

    ax_bot.set_xscale('log')
    ax_bot.set_xlabel('Monte Carlo Steps (Log scale)', fontsize=AX_LABEL_FT, labelpad=17)
    ax_bot.set_ylabel(r'$ P(D\!\to\!C)$', fontsize=AX_LABEL_FT, labelpad=10)
    ax_bot.set_ylim(-0.02, 1.02)
    ax_bot.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_bot.tick_params(axis='both', labelsize=TICK_FT)

    # ---------- 仅在第一行放图例，顺序：绿 -> 蓝 -> 红，且稍微抬高 ----------
    handles_ordered = []
    labels_ordered  = []
    for c in legend_target_colors:
        h = legend_handle_by_color.get(c, None)
        if h is not None:
            handles_ordered.append(h)
            labels_ordered.append(h.get_label())

    ax_top.legend(
        handles_ordered, labels_ordered,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.22),   # 再拉高一点距离
        ncol=len(handles_ordered),
        fontsize=LEGEND_FT,
        frameon=True,
        borderaxespad=0.4,
        columnspacing=1.4,
        handlelength=2.6
    )

    # 底部子图不放图例
    # 给抬高的图例留空间
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # 保存 PNG + PDF
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches='tight')
    fig.savefig(out_pdf,  format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved:\n  {out_png}\n  {out_pdf}")

if __name__ == "__main__":
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_csv = os.path.join(project_root, 'data', 'fig7', 'fig7_resilience_dynamics_data.csv')
    out_png  = os.path.join(project_root, 'figures', 'Figure_7_Resilience_Analysis.png')
    out_pdf  = os.path.join(project_root, 'figures', 'Figure_7_Resilience_Analysis.pdf')
    plot_figure_7_final(data_csv, out_png, out_pdf)
