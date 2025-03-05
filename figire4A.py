import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

def check_dims(npz_paths):
    """
    逐个加载 npz 文件，并打印出 'samples', 'rna', 'atac' 的形状和数据类型。
    用于排查维度是否符合预期。
    """
    for idx, npz_path in enumerate(npz_paths, start=1):
        data = np.load(npz_path, allow_pickle=True)
        
        samples = data['samples']
        rna = data['rna']
        atac = data['atac']
        
        print(f"=== File {idx}: {npz_path} ===")
        print("samples shape:", samples.shape, "| dtype:", samples.dtype)
        print("rna shape    :", rna.shape,     "| dtype:", rna.dtype)
        print("atac shape   :", atac.shape,    "| dtype:", atac.dtype)
        print("-"*60)

def load_and_bin_data_multiple(npz_paths, gex_bins=3, atac_bins=3):
    """
    加载多个 .npz 文件，并将它们沿第 0 维（样本）拼接，
    然后对 rna(一维) 和 atac(二维) 进行处理：
      - rna: 直接当作基因表达量 GEx
      - atac: 沿行方向 sum 得到一个标量
    
    最后使用 **自定义的分位数** 来分 3 档 (low/mid/high)：
      - GEx:   q=[0, 0.5284, 0.8601, 1.0]
      - auATAC: q=[0, 0.3694, 0.8561, 1.0]
    
    返回:
      samples, gex_series, auatac_series, gex_bin_series, atac_bin_series
    """
    all_samples = []
    all_rna = []
    all_atac = []
    
    # 1) 依次加载并存放到列表
    for npz_path in npz_paths:
        data = np.load(npz_path, allow_pickle=True)
        all_samples.append(data['samples'])  # shape (N,)
        all_rna.append(data['rna'])          # shape (N,)
        all_atac.append(data['atac'])        # shape (N, 2000)
    
    # 2) 沿第 0 维拼接
    samples = np.concatenate(all_samples, axis=0)  # (N1+N2, )
    rna = np.concatenate(all_rna, axis=0)          # (N1+N2, )
    atac = np.concatenate(all_atac, axis=0)        # (N1+N2, 2000)
    
    # 3) 构建 GEx 与 auATAC
    # rna 本身就是一维，每个样本一个值 => 直接用作 GEx
    gex_series = pd.Series(rna, index=samples, name='GEx')
    
    # atac 是 (N, 2000)，可先对 axis=1 求和 => 每个样本一个值
    auatac_vals = atac.sum(axis=1)  # shape (N1+N2,)
    auatac_series = pd.Series(auatac_vals, index=samples, name='auATAC')
    
    # 4) 分桶：改成你指定的分位数
    #    GEx => [0, 0.5284, 0.8601, 1.0]
    #    auATAC => [0, 0.3694, 0.8561, 1.0]
    gex_bin_series = pd.qcut(
        gex_series,
        q=[0, 0.5284, 0.8601, 1.0],
        labels=['low','mid','high']
    )
    atac_bin_series = pd.qcut(
        auatac_series,
        q=[0, 0.3694, 0.8561, 1.0],
        labels=['low','mid','high']
    )
    
    return samples, gex_series, auatac_series, gex_bin_series, atac_bin_series

def plot_sorted_curve(ax, series, bin_series=None, line_color='black', fill_colors=None):
    """
    在 ax 上绘制: x 轴为基因索引 (0 ~ n-1), y 轴为 series 排序后的数值.
    可选: 根据 bin_series (low/mid/high) 用 fill_colors 给不同区间着色。
    """
    sorted_df = pd.DataFrame({'value': series})
    if bin_series is not None:
        sorted_df['bin'] = bin_series
    sorted_df = sorted_df.sort_values('value').reset_index(drop=True)
    
    xvals = np.arange(len(sorted_df))
    yvals = sorted_df['value'].values
    ax.plot(xvals, yvals, color=line_color)
    
    if bin_series is not None and fill_colors is not None and len(fill_colors) == 3:
        n_low = (sorted_df['bin'] == 'low').sum()
        n_mid = (sorted_df['bin'] == 'mid').sum()
        ax.axvspan(0, n_low, facecolor=fill_colors[0], alpha=0.2)
        ax.axvspan(n_low, n_low + n_mid, facecolor=fill_colors[1], alpha=0.2)
        ax.axvspan(n_low + n_mid, len(sorted_df), facecolor=fill_colors[2], alpha=0.2)
    
    ax.set_xlim([0, len(sorted_df)])
    ax.set_xlabel("genes (sorted)")
    ax.set_ylabel(series.name)

def plot_sorted_curve_swapped(ax, series, bin_series=None, line_color='black', fill_colors=None):
    """
    与 plot_sorted_curve 类似，但交换 x 和 y：
    x 轴为 series 排序后的数值，y 轴为基因索引 (0 ~ n-1)。
    用于右侧分布图，使得 x,y 坐标与目标图一致。
    """
    sorted_df = pd.DataFrame({'value': series})
    if bin_series is not None:
        sorted_df['bin'] = bin_series
    sorted_df = sorted_df.sort_values('value').reset_index(drop=True)
    
    xvals = sorted_df['value'].values
    yvals = np.arange(len(sorted_df))
    ax.plot(xvals, yvals, color=line_color)
    
    if bin_series is not None and fill_colors is not None and len(fill_colors) == 3:
        n_low = (sorted_df['bin'] == 'low').sum()
        n_mid = (sorted_df['bin'] == 'mid').sum()
        ax.axhspan(0, n_low, facecolor=fill_colors[0], alpha=0.2)
        ax.axhspan(n_low, n_low + n_mid, facecolor=fill_colors[1], alpha=0.2)
        ax.axhspan(n_low + n_mid, len(sorted_df), facecolor=fill_colors[2], alpha=0.2)
    
    ax.set_xlim([min(xvals), max(xvals)])
    ax.set_xlabel(series.name)
    ax.set_ylabel("genes (sorted)")

def plot_panelA_like_target(gex_series, auatac_series, gex_bin_series, atac_bin_series,
                            output_path="panelA_like_target.png"):
    """
    绘制三面板示例：
    (A1) 上方: GEx 排序折线 (x=genes, y=GEx)
    (A2) 中间: 热图 (x=GEx bin, y=auATAC bin)
    (A3) 右侧: auATAC 排序折线 (x=auATAC, y=genes) —— 使用交换 x,y 的绘制函数
    """
    df_bins = pd.DataFrame({'GEx_bin': gex_bin_series, 'auATAC_bin': atac_bin_series})
    contingency_table = pd.crosstab(df_bins['auATAC_bin'], df_bins['GEx_bin'])
    # 设置顺序
    heatmap_y_order = ['high', 'mid', 'low']
    heatmap_x_order = ['low', 'mid', 'high']
    contingency_table = contingency_table.reindex(index=heatmap_y_order, columns=heatmap_x_order)

    fig = plt.figure(figsize=(7,5))
    gs = gridspec.GridSpec(nrows=2, ncols=2,
                           width_ratios=[3,1],
                           height_ratios=[1.5,3],
                           wspace=0.1, hspace=0.1)
    
    ax_gex = fig.add_subplot(gs[0,0])
    ax_heat = fig.add_subplot(gs[1,0])
    ax_auatac = fig.add_subplot(gs[:,1])

    # (A1)
    plot_sorted_curve(
        ax=ax_gex,
        series=gex_series,
        bin_series=gex_bin_series,
        line_color='black',
        fill_colors=('red','orange','pink')
    )
    ax_gex.set_title("GEx distribution\n(x=genes, y=GEx)")

    # (A2) 热图
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='magma',
                ax=ax_heat, cbar=False, square=True)
    ax_heat.set_xlabel("GEx bin")
    ax_heat.set_ylabel("auATAC bin")
    ax_heat.set_title("Gene count by bin")

    # (A3)
    plot_sorted_curve_swapped(
        ax=ax_auatac,
        series=auatac_series,
        bin_series=atac_bin_series,
        line_color='black',
        fill_colors=('red','orange','pink')
    )
    ax_auatac.set_title("auATAC distribution\n(x=auATAC, y=genes)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Figure saved to {output_path}")

if __name__ == "__main__":
    # 1) 先检查维度
    npz_paths = [
        "/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/data/26426404/train_0.npz",
        "/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/data/26426404/test_0.npz"
    ]
    print(">>> Checking dimensions of each file...")
    check_dims(npz_paths)
    
    # 2) 拼接并分桶（使用自定义分位数）
    samples, gex_series, auatac_series, gex_bin_series, atac_bin_series = load_and_bin_data_multiple(
        npz_paths,
        gex_bins=3,
        atac_bins=3
    )
    
    # 3) 画图
    plot_panelA_like_target(
        gex_series,
        auatac_series,
        gex_bin_series,
        atac_bin_series,
        output_path="panelA_like_target.png"
    )