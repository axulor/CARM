# analysis/plot_figure_6.py
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

# 柔和红/蓝：0=背叛(红), 1=合作(蓝)
STRAT_CMAP = ListedColormap(['#E45756', '#4C78A8'])
# 遗憾热力图：避免与策略配色冲突（红蓝），改用发散色图
REGRET_CMAP = 'PiYG_r'

# 统一色条刻度（不显示最小/最大，只显示固定中间刻度）
FIXED_BAR_TICKS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

def _scan_global_regret_maxabs(npz_paths):
    """扫描多个 npz 快照文件，得到所有遗憾值的全局 |max|。"""
    max_abs = 0.0
    for p in npz_paths:
        try:
            data = np.load(p, allow_pickle=True)
        except FileNotFoundError:
            print(f"[Warn] Snapshot not found: {p}")
            continue
        if 'snapshot_times' not in data:
            continue
        for t in data['snapshot_times']:
            key = f'regret_{t}'
            if key in data:
                arr = data[key]
                if arr.size:
                    cand = np.nanmax(np.abs(arr))
                    if np.isfinite(cand):
                        max_abs = max(max_abs, float(cand))
    return max_abs

def create_single_figure(data_path, output_path, global_abs_for_norm):
    """
    为单个动力学案例创建一张包含两行快照的图：
      - 第一排：策略快照（0=背叛, 1=合作）
      - 第二排：即时非对称归因遗憾热力图
      - 第二排下方：共享横向色条
    不添加标题；两排面板尺寸完全一致。
    """
    try:
        data = np.load(data_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: Snapshot file not found at {data_path}")
        return

    snapshot_times = data['snapshot_times']
    num_snapshots = len(snapshot_times)
    if num_snapshots == 0:
        print(f"No snapshots found in {data_path}")
        return

    # 色条范围：至少覆盖 ±2.0，确保固定刻度都在范围内
    V = max(2.0, float(global_abs_for_norm))
    vmin, vmax = -V, V

    # 布局：第三行仅放色条，保证前两排等高等宽
    fig = plt.figure(figsize=(3 * num_snapshots, 7.0))
    gs = gridspec.GridSpec(
        nrows=3, ncols=num_snapshots,
        height_ratios=[1.0, 1.0, 0.10],  # 第三行给横向色条
        hspace=0.06, wspace=0.10
    )

    # 让“t = xx”位于两排之间的中部
    XLABEL_FONTSIZE = 16
    XLABEL_PAD = 10  # 下移一些，视觉上居于两排中间

    last_im = None  # 记录最后一个遗憾图以创建色条

    for i, t in enumerate(snapshot_times):
        # 第一排：策略
        ax_strat = fig.add_subplot(gs[0, i])
        strat_grid = data[f'strategy_{t}']
        ax_strat.imshow(strat_grid, cmap=STRAT_CMAP, interpolation='nearest', aspect='equal')
        ax_strat.set_xticks([]); ax_strat.set_yticks([])
        if i == 0:
            ax_strat.set_ylabel('Strategy', fontsize=16)
        ax_strat.set_xlabel(f't = {t}', fontsize=XLABEL_FONTSIZE, labelpad=XLABEL_PAD)

        # 第二排：遗憾热力图
        ax_regret = fig.add_subplot(gs[1, i])
        regret_grid = data[f'regret_{t}']
        last_im = ax_regret.imshow(
            regret_grid, cmap=REGRET_CMAP,
            vmin=vmin, vmax=vmax,
            interpolation='nearest', aspect='equal'
        )
        ax_regret.set_xticks([]); ax_regret.set_yticks([])
        if i == 0:
            ax_regret.set_ylabel('Regret', fontsize=16)

    # 第三行：横向共享色条，长度“短两个子图宽度”，并居中（跨列 gs[2, 1:-1]）
    cax = fig.add_subplot(gs[2, 1:-1])
    cbar = fig.colorbar(last_im, cax=cax, orientation='horizontal')
    # 固定刻度
    cbar.set_ticks(FIXED_BAR_TICKS)
    cbar.ax.set_xticklabels([f"{v:.1f}" for v in FIXED_BAR_TICKS], fontsize=14)
    # 稍微拉开色条标签与色条的距离
    cbar.set_label(r'Instantaneous Regret Value $R$ ', fontsize=16, labelpad=12)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    # —— 同名 PNG + PDF —— #
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    pdf_path = os.path.splitext(output_path)[0] + '.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {output_path} and {pdf_path}")

def generate_all_figures():
    """生成所有 Figure 6 的子图（分别保存，便于后续在 Visio 拼接），并全局统一色条范围与刻度。"""
    plt.style.use('seaborn-v0_8-whitegrid')

    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)

    # 三个数据源
    npz_paths = [
        'data/fig6/fig5_combo_a_stable_snapshots.npz',
        'data/fig6/fig5_combo_b_limit_cycle_snapshots.npz',
        'data/fig6/fig5_combo_c_complex_snapshots.npz'
    ]

    # 扫描全局最大绝对遗憾，统一三张图的色条范围；至少覆盖 ±2.0 以容纳固定刻度
    global_abs = _scan_global_regret_maxabs(npz_paths)

    jobs = [
        (npz_paths[0], os.path.join(output_dir, 'Figure_6a.png')),
        (npz_paths[1], os.path.join(output_dir, 'Figure_6b.png')),
        (npz_paths[2], os.path.join(output_dir, 'Figure_6c.png')),
    ]
    for data_path, out_path in jobs:
        create_single_figure(
            data_path=data_path,
            output_path=out_path,
            global_abs_for_norm=global_abs
        )

if __name__ == "__main__":
    generate_all_figures()
