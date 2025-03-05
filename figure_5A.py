import numpy as np
import matplotlib.pyplot as plt

#==================== 1) 读取三维 SHAP 数据 ====================
# 假设:
#  - dna_shap.shape  = (n_genes_dna, 4, 2000)   -> [A=0, G=1, C=2, T=3]
#  - full_shap.shape = (n_genes_full, 5, 2000) -> [A=0, G=1, C=2, T=3, ATAC=4]
dna_shap  = np.load("dna_model_all_shap.npy")
full_shap = np.load("full_model_all_shap.npy")

#==================== 2) 通道信息及可视化参数 ====================
# 如果你的通道顺序不同，请在这里修改对应索引
channel_idx_dna = {
    "A": 0,
    "G": 1,
    "C": 2,
    "T": 3
}

channel_idx_full = {
    "A": 0,
    "G": 1,
    "C": 2,
    "T": 3,
    "ATAC": 4
}

# 与示例图类似的颜色
channel_colors = {
    "A":    "green",
    "G":    "gold",
    "C":    "blue",
    "T":    "lightcoral",  # 或 "red"
    "ATAC": "black"
}

# 位置坐标: -1000..999（共2000个点）
n_positions = dna_shap.shape[-1]  # 2000
positions = np.arange(-n_positions//2, n_positions//2)

#==================== 3) 辅助函数: 画“均值 ± 标准差” + y=0,0.0001 虚线 ====================
def plot_shap_mean_std(ax, shap_data, channel_idx, color):
    """
    在子图 ax 上绘制:
      - 所有基因在该通道、每个位置上的平均值曲线
      - ±1 标准差阴影
      - 在 y=0 和 y=0.0001 处加灰色虚线 (手动添加水平线)
    """
    vals = shap_data[:, channel_idx, :]  # (n_genes, n_positions)
    mean_vals = vals.mean(axis=0)
    std_vals  = vals.std(axis=0)

    # 均值曲线 + 阴影
    ax.plot(positions, mean_vals, color=color)
    ax.fill_between(positions, mean_vals - std_vals, mean_vals + std_vals,
                    color=color, alpha=0.2)

    # 在 y=0 和 y=0.0001 处各画一条灰色虚线 (水平)
    for yline in [0, 0.0001]:
        ax.axhline(yline, color='gray', linestyle='--', linewidth=0.8)

#==================== 4) 创建 5行 × 2列 子图布局 ====================
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8, 10), sharex=True)
plt.subplots_adjust(top=0.92, wspace=0.3, hspace=0.4)

#==================== 5) 第 1 行: 左空, 右画 ATAC ====================
axes[0, 0].set_visible(False)  # 左侧留白
ax_atac = axes[0, 1]
plot_shap_mean_std(ax_atac, full_shap, channel_idx_full["ATAC"], channel_colors["ATAC"])
ax_atac.set_title("ATAC", color=channel_colors["ATAC"])

#==================== 6) 第 2~5 行: A/G/C/T 分别在左( DNA-only ), 右( DNA+ATAC ) ====================
# 这里我们用行序来固定: 第 2 行=A, 第 3 行=G, 第 4 行=C, 第 5 行=T
channel_order = ["A", "G", "C", "T"]

for row, ch in enumerate(channel_order, start=1):
    ax_left  = axes[row, 0]  # DNA-only
    ax_right = axes[row, 1]  # DNA+ATAC

    # 左列: DNA-only
    plot_shap_mean_std(ax_left, dna_shap, channel_idx_dna[ch], channel_colors[ch])
    ax_left.set_title(ch, color=channel_colors[ch])

    # 右列: DNA+ATAC
    plot_shap_mean_std(ax_right, full_shap, channel_idx_full[ch], channel_colors[ch])
    ax_right.set_title(ch, color=channel_colors[ch])

#==================== 7) 去除子图边框, 去除多余坐标轴标签, 设置坐标 & 网格 ====================
for i in range(5):
    for j in range(2):
        ax = axes[i, j]
        if not ax.get_visible():
            continue  # 跳过第1行左侧（隐藏）

        # (a) 去掉所有边框
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)

        # (b) 去掉 Y 轴标签
        ax.set_ylabel("")

        # (c) 只有最底行显示 X 轴标签
        if i < 4:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Position relative to TSS")

        # (d) 设置纵坐标刻度仅显示 0.0000 和 0.0001
        ax.set_yticks([0.0, 0.0001])
        ax.set_yticklabels(["0.0000", "0.0001"])

        # (e) 在 x 轴上设置主要刻度: -800, -400, 0, 400, 800
        ax.set_xticks([-800, -400, 0, 400, 800])

        # (f) 只在 x 方向启用网格(虚线), 用于画出纵向虚线
        #     linestyle=':' => 点状, 也可换成 '--' => 虚线
        ax.grid(True, axis='x', color='gray', linestyle=':', alpha=0.5)

#==================== 8) 整体标题, 自动保存并展示 ====================
fig.suptitle(r"$\mu \text{ SHAP} \pm \sigma \text{ SHAP}$", fontsize=14, fontweight='bold')

plt.savefig("shap_plot.png", dpi=300, bbox_inches="tight")
plt.show()