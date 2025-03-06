import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

# 指定Excel文件路径
excel_file = "/Users/zeheng/Downloads/accessible_seq2exp-main/media-2.xlsx"

# 配色方案与图3A相同
colors = {
    'PBMC': '#663333',
    'brain': '#9966cc',
    'jejunum': '#cc6699'
}

# 模型标记
model_markers = {
    'DNA + ATAC': 's',       # 方形
    'DNA': 'o',              # 圆形
    'ATAC': '^'              # 三角形
}

def plot_figure3A():
    # 检查文件是否存在
    if not os.path.exists(excel_file):
        print(f"错误: Excel文件不存在: {excel_file}")
        return
    
    print(f"读取Excel文件: {excel_file}")
    
    # 读取Excel文件中的sheet名称
    xl = pd.ExcelFile(excel_file)
    sheet_names = xl.sheet_names
    print(f"Excel文件包含以下sheet: {sheet_names}")
    
    # 存储所有数据点
    results = []
    
    # 处理除Data Legend外的所有sheet
    for sheet_name in sheet_names:
        if sheet_name == 'Data Legend':
            continue
            
        # 提取数据集名称
        dataset_match = re.match(r'(PBMC|brain|jejunum)', sheet_name)
        if not dataset_match:
            continue
            
        dataset = dataset_match.group(1)
        print(f"\n处理数据集: {dataset}, Sheet: {sheet_name}")
        
        # 读取当前sheet
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # 获取唯一的训练细胞类型
        train_cell_types = df['train_cell_type'].unique()
        
        # 对每个训练细胞类型
        for train_celltype in train_cell_types:
            # 获取使用该训练细胞类型的所有行
            train_df = df[df['train_cell_type'] == train_celltype]
            
            # 对每种模型类型
            for model_type in model_markers.keys():
                # 获取该模型类型的所有行
                model_df = train_df[train_df['model'] == model_type]
                
                if model_df.empty:
                    continue
                
                # 找到相同细胞类型的行 (train_cell_type == test_cell_type)
                same_celltype_df = model_df[model_df['train_cell_type'] == model_df['test_cell_type']]
                
                # 如果没有相同细胞类型的评估，跳过
                if same_celltype_df.empty:
                    continue
                    
                # 获取相同细胞类型的平均相关系数
                same_celltype_corr = same_celltype_df['pearson_r_mean'].values[0]
                
                # 对于所有不同的测试细胞类型
                for _, other_row in model_df[model_df['train_cell_type'] != model_df['test_cell_type']].iterrows():
                    # 获取其他细胞类型的平均相关系数
                    other_celltype_corr = other_row['pearson_r_mean']
                    
                    # 添加到结果中
                    results.append({
                        'dataset': dataset,
                        'model_type': model_type,
                        'train_celltype': train_celltype,
                        'test_celltype': other_row['test_cell_type'],
                        'same_celltype_corr': same_celltype_corr,
                        'cross_celltype_corr': other_celltype_corr
                    })
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        print("错误: 无法从Excel文件中提取所需数据")
        return
    
    print(f"提取到 {len(results)} 个数据点")
    print(df_results.head())
    
    # 绘制图3A
    plt.figure(figsize=(10, 8))
    
    # 调整绘图区域，为右侧图例留出空间
    plt.subplots_adjust(right=0.85)
    
    # 设置网格线
    plt.grid(True, color='lightgray', linestyle=':', alpha=0.7)
    
    # 设置轴范围为0.35-0.67，与原图一致
    plt.xlim(0.35, 0.67)
    plt.ylim(0.35, 0.67)
    
    # 在绘制散点图之前，添加以下代码来去掉坐标轴四周的实线
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    
    # 对于每个数据集
    for dataset in colors.keys():
        df_subset = df_results[df_results['dataset'] == dataset]
        
        # 对于每种模型
        for model_type in model_markers.keys():
            df_model = df_subset[df_subset['model_type'] == model_type]
            
            if not df_model.empty:
                # 绘制散点
                plt.scatter(
                    df_model['same_celltype_corr'], 
                    df_model['cross_celltype_corr'],
                    color=colors[dataset],
                    marker=model_markers[model_type],
                    alpha=0.8,
                    s=60,
                    label=f"{dataset} - {model_type}"
                )
    
    # 添加对角线
    plt.plot([0.35, 0.67], [0.35, 0.67], 'k-', alpha=0.5, zorder=0)
    
    # 设置标题和标签
    plt.title('mean Pearson r', fontsize=14)
    plt.xlabel('evaluated on same cell type', fontsize=12)
    plt.ylabel('evaluated on other cell type', fontsize=12)
    
    # 添加"A"标记在左上角
    plt.text(0.02, 0.98, 'A', transform=plt.gca().transAxes, 
             fontsize=16, fontweight='bold', va='top')
    
    # 调整图例位置和字体大小
    # 数据集图例
    dataset_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[dataset], 
                            markersize=8, label=dataset, linestyle='none') 
                  for dataset in colors.keys()]
    
    # 模型类型图例
    model_handles = [plt.Line2D([0], [0], marker=model_markers[model], color='black', 
                          markersize=8, label=model, linestyle='none') 
                for model in model_markers.keys()]
    
    # 放置数据集图例，调整位置更靠近坐标系，增加字体大小
    first_legend = plt.legend(handles=dataset_handles, title="dataset", 
                         bbox_to_anchor=(1.02, 1), loc='upper left',
                         fontsize=10, title_fontsize=11)
    plt.gca().add_artist(first_legend)
    
    # 放置模型图例，调整位置更靠近坐标系，增加字体大小
    plt.legend(handles=model_handles, title="model", 
          bbox_to_anchor=(1.02, 0.6), loc='upper left',
          fontsize=10, title_fontsize=11)
    
    plt.tight_layout()  # 自动调整布局
    plt.savefig('figure3A_reproduction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_results

# 运行函数
if __name__ == "__main__":
    results = plot_figure3A()