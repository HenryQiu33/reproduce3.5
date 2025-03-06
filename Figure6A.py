import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import os

# 1. 设置图形样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['boxplot.flierprops.markersize'] = 5
plt.rcParams['boxplot.boxprops.linewidth'] = 1.5  # 增加箱体线条粗细
plt.rcParams['boxplot.whiskerprops.linewidth'] = 1.2  # 增加须线粗细
plt.rcParams['boxplot.medianprops.linewidth'] = 1.8  # 增加中位线粗细

# 自定义颜色：更精确的紫色（multiome）和橙红色（sc）
custom_palette = ['#5D3FD3', '#FF5722']  # 调整为更美观的颜色

# 2. 根据原图设置数据值
# multiome 数据 - 为每个箱体创建略微不同的值以产生箱体效果
multiome_dna = np.array([0.35, 0.36, 0.37, 0.35, 0.36, 0.34, 0.33])  # 完整箱体
multiome_atac = np.array([0.47, 0.48, 0.475, 0.465, 0.47])  # 窄箱体
multiome_dna_atac = np.array([0.51, 0.515, 0.505, 0.51, 0.512])  # 窄箱体

# sc 数据
sc_dna = np.array([0.445, 0.45, 0.44, 0.445, 0.448])  # 窄箱体
sc_atac = np.array([0.57, 0.58, 0.575, 0.565, 0.58, 0.59, 0.55])  # 窄箱体
sc_dna_atac = np.array([0.62, 0.63, 0.625, 0.615, 0.62, 0.61, 0.67])  # 完整箱体

# 3. 合并数据
data = [
    multiome_dna, multiome_atac, multiome_dna_atac,
    sc_dna, sc_atac, sc_dna_atac
]

# 4. 创建图形
fig, ax = plt.subplots(figsize=(6, 5))

# 设置背景为白色，只保留虚线横向网格
ax.set_facecolor('white')
ax.grid(True, axis='y', linestyle=':', color='gray', alpha=0.7)
ax.set_axisbelow(True)  # 网格线在下方

# 移除所有边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# 5. 绘制箱线图
positions = [0, 1, 2, 3, 4, 5]
box_width = [0.5, 0.3, 0.3, 0.3, 0.3, 0.5]  # 调整箱体宽度

# 创建自定义颜色列表
colors = [custom_palette[0]]*3 + [custom_palette[1]]*3

# 为每个箱体创建不同的中位线颜色属性
median_colors = []
for i in range(6):
    median_colors.append({'color': colors[i], 'linewidth': 1.8})

# 绘制箱线图 - 不使用 patch_artist，而是直接设置边框颜色
bplot = ax.boxplot(data, positions=positions, widths=box_width, patch_artist=False,
                  flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'white', 'markeredgecolor': 'black'},
                  boxprops={'linewidth': 1.5}, whiskerprops={'linewidth': 1.2}, 
                  medianprops=median_colors[0],  # 这个会被后面的循环覆盖
                  showfliers=True, showcaps=True)

# 设置箱体边框颜色
for i, box in enumerate(bplot['boxes']):
    box.set_color(colors[i])  # 设置箱体边框颜色
    
# 设置须线颜色
for i, whisker in enumerate(bplot['whiskers']):
    whisker.set_color(colors[i//2])  # 每个箱体有两条须线

# 设置盖子颜色
for i, cap in enumerate(bplot['caps']):
    cap.set_color(colors[i//2])  # 每个箱体有两个盖子
    
# 设置中位线颜色
for i, median in enumerate(bplot['medians']):
    median.set_color(colors[i])  # 设置中位线颜色与箱体颜色一致

# 6. 设置图形样式
ax.set_ylabel('Pearson r', fontsize=12)
ax.set_title('A', fontsize=14, loc='left', fontweight='bold')
ax.set_ylim(0.3, 0.65)
ax.set_yticks(np.arange(0.4, 0.7, 0.1))

# 添加水平虚线（替代网格线）
for y in np.arange(0.4, 0.7, 0.1):
    ax.axhline(y=y, color='gray', linestyle=':', alpha=0.7, linewidth=0.5)

# 设置 x 轴标签
labels = ['DNA\nonly', 'ATAC\nonly', 'DNA\n+\nATAC',
          'DNA\nonly', 'ATAC\nonly', 'DNA\n+\nATAC']
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=10)

# 添加图例 - 横向排列在上方
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=custom_palette[0], lw=2.5, label='multiome'),
    Line2D([0], [0], color=custom_palette[1], lw=2.5, label='sc')
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
          ncol=2, frameon=False, fontsize=10)

# 添加显著性标记（*）
ax.text(5, 0.64, '*', ha='center', va='bottom', fontsize=14, color='black')

# 7. 调整布局和显示
plt.tight_layout()
plt.savefig('figure6A.png', dpi=300, bbox_inches='tight')
plt.show()