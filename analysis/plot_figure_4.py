# analysis/plot_figure_4_export_six_uniform.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 基础风格 =========
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'

# ========= 统一字号 =========
AX_LABEL_FONTSIZE = 20   # 坐标轴标题
TICK_LABELSIZE    = 15   # 坐标轴刻度
CBAR_LABELSIZE    = 15   # 色标标题
CBAR_TICKSIZE     = 13   # 色标刻度
LABELPAD_X        = 8
LABELPAD_Y        = 2

# ========= 画布与固定布局（关键） =========
# 画布大小（英寸）——六张图都会完全一致
FIGSIZE = (6.0, 5.2)

# 主图与色标使用绝对坐标放置（[left, bottom, width, height]，相对整个画布）
# 这样不同数据/坐标不会改变绘图区和色标的物理大小
AX_RECT    = [0.14, 0.14, 0.70, 0.76]   # 主绘图区
CBAR_RECT  = [0.86, 0.14, 0.03, 0.76]   # 右侧色标

# ========= 配色 =========
CMAP_MEAN = plt.get_cmap('viridis')  # 平均合作率
CMAP_STD  = plt.get_cmap('YlOrRd')   # 标准差：低值浅黄，高值偏红

def _format_label(name: str) -> str:
    return {
        'BETA':  r'$\beta$',
        'GAMMA': r'$\gamma$',
        'KAPPA': r'$\kappa$',
    }.get(name.upper(), name)

def _pivot(csv_path: str, x_col: str, y_col: str, v_col: str):
    """读取并透视为网格数据。"""
    df = pd.read_csv(csv_path)
    piv = df.pivot(index=y_col, columns=x_col, values=v_col)
    return piv.columns.values, piv.index.values, piv.values

def _draw_single_heatmap(csv_file: str,
                         x_name: str, y_name: str,
                         value_col: str, cmap,
                         vmin=None, vmax=None,
                         x_log=False, y_log=False,
                         cbar_label: str = '',
                         out_basename: str = 'Figure_4_panel'):
    """生成一张尺寸与布局完全一致的热力图（含色标）。"""
    # --- 数据 ---
    x_vals, y_vals, Z = _pivot(csv_file, x_name, y_name, value_col)
    # 清理无效值
    Z = np.where(np.isfinite(Z), Z, np.nan)
    # 若需要 per-panel 自适应 vmax（例如标准差）
    if vmin is None:
        vmin = np.nanmin(Z) if np.isfinite(Z).any() else 0.0
    if vmax is None:
        vmax = np.nanmax(Z) if np.isfinite(Z).any() else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-6

    # --- 画布与固定位置的轴 ---
    fig = plt.figure(figsize=FIGSIZE)
    ax  = fig.add_axes(AX_RECT)        # 主图
    cax = fig.add_axes(CBAR_RECT)      # 色标

    # --- 绘制 ---
    im = ax.pcolormesh(x_vals, y_vals, Z, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    # 轴样式
    ax.set_xlabel(_format_label(x_name), fontsize=AX_LABEL_FONTSIZE, labelpad=LABELPAD_X)
    ax.set_ylabel(_format_label(y_name), fontsize=AX_LABEL_FONTSIZE, labelpad=LABELPAD_Y)
    ax.tick_params(axis='both', labelsize=TICK_LABELSIZE)
    if x_log:
        ax.set_xscale('log')
    if y_log:
        ax.set_yscale('log')

    # 色标（固定位置，宽度一致）
    cbar = fig.colorbar(im, cax=cax)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=CBAR_LABELSIZE)
    cbar.ax.tick_params(labelsize=CBAR_TICKSIZE)

    # --- 输出 ---
    os.makedirs("figures", exist_ok=True)
    png_path = os.path.join("figures", f"{out_basename}.png")
    pdf_path = os.path.join("figures", f"{out_basename}.pdf")
    fig.savefig(png_path, dpi=600)        # 不用 tight，确保外框一致
    fig.savefig(pdf_path, format='pdf')
    plt.close(fig)
    print(f"Saved: {png_path} and {pdf_path}")

def export_six_uniform_heatmaps():
    """
    逐一导出 6 张尺寸与布局完全一致的图：
    左列（mean，viridis），右列（std，YlOrRd）
    (a) β–γ, (b) γ–κ, (c) κ–β
    """
    cfgs = [
        # csv_path,   x,       y,       x_log, y_log, panel_tag
        ("data/fig4/fig4a_beta_vs_gamma_data.csv",  "BETA",  "GAMMA", True,  True,  "4a"),
        ("data/fig4/fig4b_gamma_vs_kappa_data.csv",  "KAPPA", "GAMMA", False, True,   "4b"),
        ("data/fig4/fig4c_kappa_vs_beta_data.csv",  "KAPPA", "BETA",  False, True,  "4c"),
    ]

    for csv_file, x_name, y_name, xlog, ylog, tag in cfgs:
        # —— 平均合作率 ——（固定 [0,1] 色标范围）
        _draw_single_heatmap(
            csv_file,
            x_name, y_name,
            value_col="mean_rho_c",
            cmap=CMAP_MEAN,
            vmin=0.0, vmax=1.0,
            x_log=xlog, y_log=ylog,
            cbar_label=r"$\langle \rho_C \rangle$",
            out_basename=f"Figure_{tag}_mean"
        )

        # —— 标准差 ——（每张图按自身数据自适应 vmax，vmin=0）
        # 先取一次数据以获取该面板的最大值
        _, _, stdZ = _pivot(csv_file, x_name, y_name, "std_rho_c")
        stdZ = np.where(np.isfinite(stdZ) & (stdZ >= 0), stdZ, 0.0)
        panel_vmax = float(np.nanmax(stdZ)) if np.isfinite(stdZ).any() else 1.0
        if panel_vmax <= 0:
            panel_vmax = 1.0

        _draw_single_heatmap(
            csv_file,
            x_name, y_name,
            value_col="std_rho_c",
            cmap=CMAP_STD,
            vmin=0.0, vmax=panel_vmax,
            x_log=xlog, y_log=ylog,
            cbar_label=r"$\mathrm{Std}(\rho_C)$",
            out_basename=f"Figure_{tag}_std"
        )

if __name__ == "__main__":
    export_six_uniform_heatmaps()
