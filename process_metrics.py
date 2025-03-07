import os
import pickle
import pandas as pd
import numpy as np

# 获取所有metrics文件并按实验组分类
metrics_files = sorted([f for f in os.listdir('.') if f.endswith('_metrics.pkl')])

# 按实验组整理数据
grouped_results = {}
all_metrics = []  # 存储所有指标用于计算总体平均值

for file in metrics_files:
    group_num = int(file.split('.')[0])  # 获取主实验组号
    fold_num = int(file.split('.')[1].split('_')[0])  # 获取fold号
    
    if group_num not in grouped_results:
        grouped_results[group_num] = {'fold_metrics': {}}
        
    with open(file, 'rb') as f:
        metrics = pickle.load(f)
        grouped_results[group_num]['fold_metrics'][fold_num] = metrics
        all_metrics.append(metrics)  # 添加到总体列表中

# 为每个实验组生成性能文件
for group_num in grouped_results:
    # 创建性能DataFrame
    performance_df = pd.DataFrame.from_dict(grouped_results[group_num]['fold_metrics'], orient='index')
    
    # 保存每个组的性能文件
    output_filename = f'group_{group_num}_performance.csv'
    performance_df.to_csv(output_filename)
    
    # 计算该组的平均值和标准差
    grouped_results[group_num]['mean'] = performance_df.mean()
    grouped_results[group_num]['std'] = performance_df.std()

# 创建总结DataFrame
summary_df = pd.DataFrame(columns=['metric_type'] + list(performance_df.columns))

# 为每个实验组添加均值和标准差
for group_num in grouped_results:
    mean_row_name = f'Group {group_num} mean'
    std_row_name = f'Group {group_num} std'
    
    summary_df.loc[mean_row_name] = ['mean'] + list(grouped_results[group_num]['mean'])
    summary_df.loc[std_row_name] = ['std'] + list(grouped_results[group_num]['std'])

# 计算总体平均值和标准差
all_metrics_df = pd.DataFrame(all_metrics)
overall_mean = all_metrics_df.mean()
overall_std = all_metrics_df.std()

# 添加总体统计到summary
summary_df.loc['Overall mean'] = ['mean'] + list(overall_mean)
summary_df.loc['Overall std'] = ['std'] + list(overall_std)

# 保存总结文件
summary_df.to_csv('performance_summary.csv')

# 打印汇总统计
print("\n=== Summary Statistics ===")
print(summary_df)

# 打印总体平均值
print("\n=== Overall Statistics ===")
print(f"Mean values across all experiments:")
for metric, value in overall_mean.items():
    print(f"{metric}: {value:.6f} ± {overall_std[metric]:.6f}") 