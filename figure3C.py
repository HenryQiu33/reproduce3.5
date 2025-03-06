import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置路径
excel_path = "/Users/zeheng/Downloads/accessible_seq2exp-main/media-2.xlsx"

# 配色方案
colors = {
    'PBMC': '#663333',
    'brain': '#9966cc',
    'jejunum': '#cc6699'
}


# 创建DataFrame
results = {
    'dataset': [],
    'all_genes_increase': [],
    'hv_genes_increase': []
}

# 将手动设置的点添加到结果中
for dataset, values in data.items():
    for i in range(4):  # 每个数据集4个点
        results['dataset'].append(dataset)
        results['all_genes_increase'].append(values['all_genes'][i])
        results['hv_genes_increase'].append(values['hv_genes'][i])

df_results = pd.DataFrame(results)

# 绘图
plt.figure(figsize=(8, 8))

# 去除坐标轴边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# 添加网格线
plt.grid(True, color='lightgray', linestyle='-', alpha=0.3)

# 绘制散点图（使用三角形标记）
for dataset in colors.keys():
    mask = df_results['dataset'] == dataset
    plt.scatter(df_results[mask]['all_genes_increase'], 
               df_results[mask]['hv_genes_increase'],
               c=colors[dataset],
               marker='^',  # 使用三角形标记
               label=dataset,
               alpha=0.8,
               s=100)  # 增大标记大小

# 添加对角线
plt.plot([0, 50], [0, 50], 'k-', alpha=0.5)

# 设置标签和标题
plt.xlabel('evaluated on all genes', fontsize=12)
plt.ylabel('evaluated on highly variable genes', fontsize=12)
plt.title('% increase in mean Pearson r\nrelative to ATAC only', fontsize=14)

# 设置坐标轴范围
plt.xlim(0, 50)
plt.ylim(0, 50)

# 添加图例
plt.legend(title='dataset', bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
plt.savefig('figure3C.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印结果以便检查
print("\nResults:")
print(df_results)
