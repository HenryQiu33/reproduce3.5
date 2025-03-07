import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建图形
plt.figure(figsize=(12, 8))

def plot_box_with_error(data, y_pos, color='red'):
    # 确保数据是float类型
    data = np.array(data, dtype=float)
    
    # 计算统计量
    mean = np.mean(data)
    std = np.std(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    ci = 1.96 * std
    
    # 画矩形框（Q1-Q3范围）
    rect = plt.Rectangle((q1, y_pos-0.15), q3-q1, 0.3, 
                        facecolor='white', edgecolor=color, linewidth=1.0)
    plt.gca().add_patch(rect)
    
    # 画中间竖线（均值）
    plt.plot([mean, mean], [y_pos-0.15, y_pos+0.15], color=color, linewidth=1.0)
    
    # 画误差线（95%置信区间）
    plt.plot([mean-ci, q1], [y_pos, y_pos], color=color, linewidth=1.0)
    plt.plot([q3, mean+ci], [y_pos, y_pos], color=color, linewidth=1.0)
    
    # 画误差线末端
    plt.plot([mean-ci, mean-ci], [y_pos-0.08, y_pos+0.08], color=color, linewidth=1.0)
    plt.plot([mean+ci, mean+ci], [y_pos-0.08, y_pos+0.08], color=color, linewidth=1.0)

# 定义y轴位置，从上到下排列
y_labels = [
    'DNA + ATAC',           # 1.1
    'ATAC only',            # 1.2
    'DNA only',             # 1.3
    'ATAC + scrambled DNA', # 2.1
    'DNA + scrambled ATAC', # 2.2
    'scrambled DNA + scrambled ATAC (pairwise)', # 2.3
    'scrambled DNA + scrambled ATAC (separate)', # 2.4
    'ATAC + scrambled DNA', # 3.1
    'DNA + scrambled ATAC', # 3.2
    'scrambled DNA + scrambled ATAC (pairwise)', # 3.3
    'scrambled DNA + scrambled ATAC (separate)'  # 3.4
]

# 设置位置，从上到下
y_positions = np.linspace(12, 2, len(y_labels))

# 第一组：unscrambled (ablation) - 1.1, 1.2, 1.3
dna_atac = pd.read_csv('/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/evaluation/output_performance_on_26426404_dna_atac1.1.csv')
plot_box_with_error(dna_atac['pearson_r'].values[:5].astype(float), y_positions[0], 'purple')

atac = pd.read_csv('/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/evaluation/output_performance_on_26426404_atac1.2.csv')
plot_box_with_error(atac['pearson_r'].values[:5].astype(float), y_positions[1], 'red')

dna = pd.read_csv('/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/evaluation/output_performance_on_26426404_dna1.3.csv')
plot_box_with_error(dna['pearson_r'].values[:5].astype(float), y_positions[2], 'black')

# 第二组：scrambled train; unscrambled test - 2.1
df_2_1 = pd.read_csv('/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/hanwen/output_figure2_1/output1/250304-140920_AugmentedSeq2Exp_data_metrics/performance_summary2.1.csv')
plot_box_with_error(df_2_1[df_2_1['metric_type'] == 'mean']['pearson_r'].values[:5].astype(float), y_positions[3], 'red')

# 2.2 DNA + scrambled ATAC
df_2_2 = pd.read_csv('/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/hanwen/performance_summary2.2.csv')
plot_box_with_error(df_2_2[df_2_2['metric_type'] == 'mean']['pearson_r'].values[:5].astype(float), y_positions[4], 'black')

# 预留其他2.x实验组的位置
# 当有数据时，取消下面的注释并填入正确的文件路径
# """
# 2.3 scrambled DNA + scrambled ATAC (pairwise)
df_2_3 = pd.read_csv('/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/hanwen/enhanced_performance_summary2.3.csv')
plot_box_with_error(df_2_3[df_2_3['metric_type'] == 'mean']['pearson_r'].values[:5].astype(float), y_positions[5], 'blue')

# 2.4 scrambled DNA + scrambled ATAC (separate)
df_2_4 = pd.read_csv('/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/hanwen/updated_performance_summary2.4.csv')
plot_box_with_error(df_2_4[df_2_4['metric_type'] == 'mean']['pearson_r'].values[:5].astype(float), y_positions[6], 'green')
# """

# 3.1 ATAC + scrambled DNA
df_3_1 = pd.read_csv('/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/hanwen/performance_summary3.1.csv')
plot_box_with_error(df_3_1[df_3_1['metric_type'] == 'mean']['pearson_r'].values[:5].astype(float), y_positions[7], 'red')

# 3.2 DNA + scrambled ATAC
df_3_2 = pd.read_csv('/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/hanwen/performance_summary3.2.csv')
plot_box_with_error(df_3_2[df_3_2['metric_type'] == 'mean']['pearson_r'].values[:5].astype(float), y_positions[8], 'black')

# 3.3 scrambled DNA + scrambled ATAC (pairwise)
df_3_3 = pd.read_csv('/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/hanwen/performance_summary3.3.csv')
plot_box_with_error(df_3_3[df_3_3['metric_type'] == 'mean']['pearson_r'].values[:5].astype(float), y_positions[9], 'blue')

# 第三组：unscrambled train; scrambled test - 3.4
df_3_4 = pd.read_csv('/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/hanwen/output_figure2_8/performance_summary3.4.csv')
plot_box_with_error(df_3_4[df_3_4['metric_type'] == 'mean']['pearson_r'].values[:5].astype(float), y_positions[10], 'red')

# 设置y轴标签
plt.yticks(y_positions, y_labels)

# 添加分组标记线
plt.axhline(y=y_positions[2] - 0.5, color='black', linestyle='-', alpha=0.15)  # 第一组和第二组之间
plt.axhline(y=y_positions[6] - 0.5, color='black', linestyle='-', alpha=0.15)  # 第二组和第三组之间

# 添加分组标签
plt.text(0.72, y_positions[1], 'unscrambled\n(ablation)', 
         verticalalignment='center', horizontalalignment='left')
plt.text(0.72, y_positions[5], 'scrambled train;\nunscrambled test\n(DNA + ATAC)', 
         verticalalignment='center', horizontalalignment='left')
plt.text(0.72, y_positions[9], 'unscrambled train;\nscrambled test\n(DNA + ATAC)', 
         verticalalignment='center', horizontalalignment='left')

# 设置x轴范围和标签
plt.xlim(0.0, 0.7)
plt.xlabel('Pearson r')

# 添加网格线
plt.grid(True, axis='x', linestyle=':', alpha=0.3)

# 添加边框
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)

# 调整布局
plt.gcf().set_size_inches(12, 8)
plt.subplots_adjust(left=0.35, right=0.85)  # 调整左右边距

# 添加标题
plt.title('scrambling and ablation experiments (B cell)', pad=20)

# 保存图表到文件
plt.savefig('figure2_complete.png', dpi=300, bbox_inches='tight')
print("图表已保存到 figure2_complete.png")

# 显示图表
plt.show()