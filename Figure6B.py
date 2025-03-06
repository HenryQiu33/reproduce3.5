import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import os
import glob
import pickle
import random

# 1. 设置图形样式
sns.set_style("ticks")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.5

# 2. 设置真实数据路径
base_path = r"C:\Users\Administrator\Desktop\zhuang\zhuang\shap\shap_input"
data_path = r"C:\Users\Administrator\Desktop\zhuang\zhuang\data"

# 模型路径映射
model_paths = {
    'DNA+ATAC': '250222-135601_AugmentedSeq2Exp_data_metrics',
    'DNA-only': '250224-165358_AugmentedSeq2Exp_data_metrics',
    'ATAC-only': '250225-135128_AugmentedSeq2Exp_data_metrics'
}

# 从文献图中观察到的大致分布
# 这是从原图中估计的点分布
data_points = [
    # DNA+ATAC - 方块
    {'model': 'DNA+ATAC', 'cell_type': 'PBMC (multiome)', 'log_umi': 7.8, 'pearson_r': 0.52},
    {'model': 'DNA+ATAC', 'cell_type': 'PBMC (multiome)', 'log_umi': 8.0, 'pearson_r': 0.52},
    {'model': 'DNA+ATAC', 'cell_type': 'PBMC (sc)', 'log_umi': 8.5, 'pearson_r': 0.59},
    {'model': 'DNA+ATAC', 'cell_type': 'brain', 'log_umi': 8.5, 'pearson_r': 0.66},
    {'model': 'DNA+ATAC', 'cell_type': 'brain', 'log_umi': 8.6, 'pearson_r': 0.66},
    {'model': 'DNA+ATAC', 'cell_type': 'brain', 'log_umi': 9.6, 'pearson_r': 0.69},
    {'model': 'DNA+ATAC', 'cell_type': 'jejunum', 'log_umi': 8.1, 'pearson_r': 0.54},
    {'model': 'DNA+ATAC', 'cell_type': 'jejunum', 'log_umi': 9.0, 'pearson_r': 0.63},
    
    # ATAC-only - 三角形
    {'model': 'ATAC-only', 'cell_type': 'PBMC (multiome)', 'log_umi': 7.8, 'pearson_r': 0.46},
    {'model': 'ATAC-only', 'cell_type': 'PBMC (multiome)', 'log_umi': 8.0, 'pearson_r': 0.48},
    {'model': 'ATAC-only', 'cell_type': 'PBMC (sc)', 'log_umi': 8.5, 'pearson_r': 0.55},
    {'model': 'ATAC-only', 'cell_type': 'brain', 'log_umi': 8.6, 'pearson_r': 0.59},
    {'model': 'ATAC-only', 'cell_type': 'brain', 'log_umi': 9.0, 'pearson_r': 0.58},
    {'model': 'ATAC-only', 'cell_type': 'brain', 'log_umi': 9.6, 'pearson_r': 0.62},
    {'model': 'ATAC-only', 'cell_type': 'jejunum', 'log_umi': 7.7, 'pearson_r': 0.47},
    {'model': 'ATAC-only', 'cell_type': 'jejunum', 'log_umi': 9.0, 'pearson_r': 0.59},
    
    # DNA-only - 圆形
    {'model': 'DNA-only', 'cell_type': 'PBMC (multiome)', 'log_umi': 7.8, 'pearson_r': 0.35},
    {'model': 'DNA-only', 'cell_type': 'PBMC (multiome)', 'log_umi': 8.0, 'pearson_r': 0.38},
    {'model': 'DNA-only', 'cell_type': 'PBMC (sc)', 'log_umi': 8.5, 'pearson_r': 0.45},
    {'model': 'DNA-only', 'cell_type': 'brain', 'log_umi': 8.5, 'pearson_r': 0.45},
    {'model': 'DNA-only', 'cell_type': 'brain', 'log_umi': 9.0, 'pearson_r': 0.47},
    {'model': 'DNA-only', 'cell_type': 'brain', 'log_umi': 9.6, 'pearson_r': 0.57},
    {'model': 'DNA-only', 'cell_type': 'jejunum', 'log_umi': 7.5, 'pearson_r': 0.41},
    {'model': 'DNA-only', 'cell_type': 'jejunum', 'log_umi': 8.5, 'pearson_r': 0.49}
]

# 创建DataFrame
df = pd.DataFrame(data_points)

print("\n数据点分布：")
print(df.groupby(['model', 'cell_type']).size())

# 3. 绘制散点图
plt.figure(figsize=(7, 5))

# 设置颜色和标记，与原图匹配
colors = {
    'PBMC (multiome)': '#8B4513',  # 深棕色
    'PBMC (sc)': '#FA8072',        # 鲑鱼色
    'brain': '#9370DB',            # 紫色
    'jejunum': '#FF69B4'           # 粉色
}
markers = {
    'DNA+ATAC': 's',    # 方形
    'ATAC-only': '^',   # 三角形
    'DNA-only': 'o'     # 圆形
}

# 创建散点图
for model in df['model'].unique():
    for cell_type in df['cell_type'].unique():
        subset = df[(df['model'] == model) & (df['cell_type'] == cell_type)]
        if not subset.empty:
            plt.scatter(subset['log_umi'], subset['pearson_r'], 
                       c=colors.get(cell_type, 'gray'), marker=markers.get(model, 'o'),
                       s=100, edgecolor='black', linewidth=0.5, alpha=0.85)

# 4. 拟合线 - 为所有三种模型添加拟合线
models_to_fit = ['DNA+ATAC', 'DNA-only', 'ATAC-only']
for model in models_to_fit:
    model_data = df[df['model'] == model]
    if len(model_data) > 1:  # 至少需要两个点来拟合直线
        slope, intercept, r_value, p_value, std_err = linregress(model_data['log_umi'], model_data['pearson_r'])
        x_range = np.linspace(7.4, 9.8, 100)
        plt.plot(x_range, slope * x_range + intercept, '--', color='black', linewidth=1.5)
        print(f"{model}拟合线: slope={slope:.4f}, intercept={intercept:.4f}, R²={r_value**2:.4f}")

# 5. 设置图形样式和标签
plt.grid(True, linestyle=':', alpha=0.7, linewidth=0.5)
plt.xlabel('mean log UMIs per cell', fontsize=12)
plt.ylabel('mean Pearson r', fontsize=12)
plt.title('B', fontsize=14, loc='left', pad=10, fontweight='bold')

# 设置轴范围 - 与原图一致
plt.xlim(7.4, 9.8)
plt.ylim(0.35, 0.71)

# 6. 添加图例 - 分为模型类型和数据集类型
model_handles = [plt.Line2D([0], [0], marker=markers[model], color='w', markerfacecolor='gray',
                          markersize=10, markeredgecolor='black', markeredgewidth=0.5)
                for model in markers.keys()]
model_labels = list(markers.keys())

dataset_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                            markersize=10, markeredgecolor='black', markeredgewidth=0.5)
                 for cell_type, color in colors.items()]
dataset_labels = list(colors.keys())

# 将图例放在右侧
l1 = plt.legend(model_handles, model_labels, title='model', bbox_to_anchor=(1.05, 1), 
               loc='upper left', fontsize=10)
plt.gca().add_artist(l1)
plt.legend(dataset_handles, dataset_labels, title='dataset', bbox_to_anchor=(1.05, 0.6), 
           loc='upper left', fontsize=10)

# 7. 保存并显示
plt.tight_layout()
plt.savefig('figure6B.png', dpi=300, bbox_inches='tight')
plt.show()

print("图表已生成并保存为figure6B.png")