import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

#===========================================================
# 1) 定义 Spearman 相关计算函数
#===========================================================
def compute_spearman_corr(shap_data, channel_index, atac_data):
    """
    对指定通道的 SHAP 与 ATAC 在每个位置计算 Spearman 相关系数.
    
    参数:
      shap_data:   (N, n_channels, 2000)  (N=19095, n_channels=4或5, positions=2000)
      channel_index: 指定通道的索引 (0..3 或 0..4)
      atac_data:   (N, 2000)  (ATAC信号)
    返回:
      corr_vals:   (2000,) 每个位置的相关系数
    """
    N, _, positions = shap_data.shape
    corr_vals = np.zeros(positions)

    for p in range(positions):
        x = shap_data[:, channel_index, p]  # 该位置所有样本的 SHAP
        y = atac_data[:, p]                # 该位置所有样本的 ATAC
        r, _ = spearmanr(x, y)
        corr_vals[p] = r

    return corr_vals

#===========================================================
# 2) 加载 DNA-only 与 Full 模型的 SHAP, 以及合并后的 ATAC
#===========================================================
dna_shap  = np.load("dna_model_all_shap_link.npy")   # shape=(19095, 4, 2000)
full_shap = np.load("full_model_all_shap_link.npy")  # shape=(19095, 5, 2000)
atac_data = np.load("atac_merged.npy")          # shape=(19095, 2000)

print("dna_shap shape:", dna_shap.shape)
print("full_shap shape:", full_shap.shape)
print("atac_data shape:", atac_data.shape)

#===========================================================
# 3) 定义通道索引与颜色
#    - DNA-only: A/G/C/T => 索引0..3
#    - Full: A/G/C/T/ATAC => 索引0..4
#===========================================================
channel_idx_dna = {
    "A": 0,
    "G": 1,
    "C": 2,
    "T": 3
}
channel_idx_full = {
    "A":    0,
    "G":    1,
    "C":    2,
    "T":    3,
    "ATAC": 4
}
channel_colors = {
    "A":    "green",
    "G":    "gold",
    "C":    "blue",
    "T":    "lightcoral",
    "ATAC": "black"
}

# 位置坐标: 若 2000点对应 -1000..999
positions = np.arange(-1000, 1000)

#===========================================================
# 4) 计算各通道的 Spearman 相关曲线
#    - Full 模型: A/G/C/T/ATAC
#    - DNA-only:  A/G/C/T
#===========================================================
corr_full = {}
for ch in ["A", "G", "C", "T", "ATAC"]:
    idx = channel_idx_full[ch]
    corr_full[ch] = compute_spearman_corr(full_shap, idx, atac_data)

corr_dna = {}
for ch in ["A", "G", "C", "T"]:
    idx = channel_idx_dna[ch]
    corr_dna[ch] = compute_spearman_corr(dna_shap, idx, atac_data)

#===========================================================
# 5) 创建 5行×2列 的子图布局 (类似图 B)
#    - 第1行: 左空白, 右=Full的ATAC
#    - 第2~5行: A/G/C/T => 左=DNA-only, 右=Full
#===========================================================
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8, 10), sharex=True)
plt.subplots_adjust(top=0.92, wspace=0.3, hspace=0.4)

# 5.1) 第1行: 左空白, 右画 Full 模型的 ATAC 通道
axes[0, 0].set_visible(False)  # 左侧隐藏
ax_atac = axes[0, 1]
ax_atac.plot(positions, corr_full["ATAC"], color=channel_colors["ATAC"])
ax_atac.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax_atac.set_title("ATAC", color=channel_colors["ATAC"])

# 5.2) 第2~5行: A/G/C/T => 左列=DNA-only, 右列=Full
channel_order = ["A", "G", "C", "T"]
for row, ch in enumerate(channel_order, start=1):
    ax_left  = axes[row, 0]
    ax_right = axes[row, 1]

    # DNA-only
    ax_left.plot(positions, corr_dna[ch], color=channel_colors[ch])
    ax_left.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_left.set_title(ch, color=channel_colors[ch])

    # Full
    ax_right.plot(positions, corr_full[ch], color=channel_colors[ch])
    ax_right.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_right.set_title(ch, color=channel_colors[ch])

#===========================================================
# 6) 美化: 去除多余边框, 设置坐标
#===========================================================
for i in range(5):
    for j in range(2):
        ax = axes[i, j]
        if not ax.get_visible():
            continue  # 跳过隐藏的子图

        # 去除四周边框
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)

        # 只有最底行显示 X 轴标签
        if i < 4:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Position relative to TSS")

        # y轴范围与刻度可根据数据大小调
        ax.set_ylim([-1.0, 1.0])  # Spearman r in [-1,1]
        # 可在这里设置 y刻度 ax.set_yticks([-1.0, 0.0, 1.0])

        # 设置主要 x 轴刻度
        ax.set_xticks([-800, -400, 0, 400, 800])

        # 加网格(仅 x方向)
        ax.grid(True, axis='x', color='gray', linestyle=':', alpha=0.5)

#===========================================================
# 7) 整体标题 & 保存/展示
#===========================================================
fig.suptitle(r"$\mu \text{ SHAP} \; vs.\; \text{ATAC track Spearman } r$", 
             fontsize=14, fontweight='bold')

plt.savefig("spearman_plot.png", dpi=300, bbox_inches="tight")
plt.show()