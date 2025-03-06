import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 设置图形样式
sns.set_style("ticks")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.5

# 2. 创建与原图匹配的精确数据
normal_training = [0.622, 0.621, 0.621, 0.623, 0.622]
with_pretraining = [0.631, 0.633, 0.640, 0.629, 0.635]

# 3. 创建绘图数据
data = pd.DataFrame({
    'Method': ['normal training'] * len(normal_training) + ['with pre-training'] * len(with_pretraining),
    'Pearson r': normal_training + with_pretraining
})

# 4. 绘制箱线图
fig, ax = plt.subplots(figsize=(4, 5))

# 创建箱线图 - 红色线条，白色填充
box_plot = sns.boxplot(x='Method', y='Pearson r', data=data, width=0.5, 
                linewidth=1, boxprops=dict(facecolor='white', edgecolor='red'),
                whiskerprops=dict(color='red'), 
                medianprops=dict(color='red'),
                capprops=dict(color='red'),
                flierprops=dict(markerfacecolor='white', markeredgecolor='red'),
                ax=ax)

# 添加散点图显示实际数据点
colors = {'normal training': '#A9A9A9', 'with pre-training': '#FF0000'}
for method, color in colors.items():
    subset = data[data['Method'] == method]
    ax.scatter(x=[0 if method == 'normal training' else 1] * len(subset), 
              y=subset['Pearson r'], 
              color=color, s=30, alpha=0.9, edgecolor='none')

# 5. 设置Y轴范围和刻度
ax.set_ylim(0.618, 0.665)
ax.set_yticks([0.62, 0.63, 0.64, 0.65, 0.66])

# 6. 设置图形样式和标签
ax.grid(True, linestyle=':', alpha=0.7, axis='y', linewidth=0.5)
ax.set_title('C', fontsize=14, loc='left', pad=10, fontweight='bold')

# 添加Method标签在底部
ax.set_xlabel('Method', fontsize=10)

# 设置X轴刻度标签
ax.set_xticklabels(['normal training', 'with pre-training'])

# 7. 去除所有坐标轴线条
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# 8. 添加p值和横线作为图表外部元素
# 在顶部添加额外的空间
plt.subplots_adjust(top=0.85)

# 添加p值文本，在横线上方居中
fig.text(0.5, 0.9, "p = 0.03125", ha='center', va='center', fontsize=10)

# 添加横线，横跨整个图表宽度，保持为黑色
line_width = 0.4  # 横线宽度占图表宽度的比例
line_x_start = 0.5 - line_width/2
line_x_end = 0.5 + line_width/2
fig.add_artist(plt.Line2D([line_x_start, line_x_end], [0.88, 0.88], color='black', linewidth=1, transform=fig.transFigure))

plt.tight_layout(rect=[0, 0, 1, 0.88])  # 调整布局，但保留顶部空间给p值

# 9. 保存图形
plt.savefig('figure6C.png', dpi=300, bbox_inches='tight')
plt.show()

print("图表已生成并保存为figure6C.png")