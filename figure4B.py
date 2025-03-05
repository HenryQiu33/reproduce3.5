import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math

##############################################
# 1) 加载与分桶：从 train/test npz 文件中拼接数据
##############################################
def load_and_bin_data_multiple(npz_paths):
    """
    加载多个 .npz 文件，并将它们沿第 0 维（样本）拼接，
    处理方式：
      - rna: 直接作为 GEx
      - atac: 作为 ATAC 轨迹，沿行求和得到 auATAC
    使用指定分位数将 GEx 和 auATAC 分桶：
      - GEx:   q = [0, 0.5284, 0.8601, 1.0]
      - auATAC: q = [0, 0.3694, 0.8561, 1.0]
    
    返回：
      samples, gex_series, auatac_series, gex_bin_series, atac_bin_series, atac_track
    """
    all_samples = []
    all_rna = []
    all_atac = []

    for npz_path in npz_paths:
        data = np.load(npz_path, allow_pickle=True)
        all_samples.append(data['samples'])
        all_rna.append(data['rna'])
        all_atac.append(data['atac'])

    samples = np.concatenate(all_samples, axis=0)
    rna = np.concatenate(all_rna, axis=0)
    atac = np.concatenate(all_atac, axis=0)

    gex_series = pd.Series(rna, index=samples, name='GEx')
    auatac_vals = atac.sum(axis=1)
    auatac_series = pd.Series(auatac_vals, index=samples, name='auATAC')

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

    return samples, gex_series, auatac_series, gex_bin_series, atac_bin_series, atac

##############################################
# 2) 绘制 panelB：双纵坐标 + 3×3 子图（下界保留数据，但让 0 对齐）
##############################################
def plot_panelB_like_dualaxes(
    atac_track: np.ndarray,
    fullmodelshap: np.ndarray,
    gex_bin_series: pd.Series,
    atac_bin_series: pd.Series,
    output_path="panelB_like_dualaxes.png"
):
    """
    在横轴 [-1300, +1300] 范围内，仅在 [-1000, +1000] 有数据，
    使用双纵坐标绘制:
      - 左轴: ATAC track (粉色)
      - 右轴: ATAC SHAP (紫色)，右轴数据先除以 1e-4，使得数值映射到一个较小范围，
        然后平移使得右轴 0 与左轴 0 对齐。
    """

    # 原始数据对应 TSS ±1000 (2000 点)
    # 嵌入到横轴 [-1300, +1300]（共2600点），数据放在 [300:2300] 对应 [-1000, +1000]
    offset = 300
    full_len = 2600
    positions_full = np.arange(-1300, 1300)  # 2600 个点

    data_left, data_right = offset, offset + 2000  # 数据所在区间

    # 遍历 9 个类别，先计算全局范围（左轴和右轴均基于原始数据）
    gex_order  = ['low','mid','high']
    atac_order = ['low','mid','high']

    track_global_min = np.inf
    track_global_max = -np.inf
    shap_global_min = np.inf
    shap_global_max = -np.inf

    # 存储每个 bin 的均值和标准差
    data_dict = {}

    for gex_cat in gex_order:
        for atac_cat in atac_order:
            mask = (gex_bin_series == gex_cat) & (atac_bin_series == atac_cat)
            subset_idx = np.where(mask)[0]
            if len(subset_idx) == 0:
                data_dict[(gex_cat, atac_cat)] = None
                continue

            sub_atac = atac_track[subset_idx, :]       # (n, 2000)
            sub_shap = fullmodelshap[subset_idx, 4, :]   # (n, 2000) 第5通道

            mean_atac = sub_atac.mean(axis=0)
            std_atac  = sub_atac.std(axis=0)
            mean_shap = sub_shap.mean(axis=0)
            std_shap  = sub_shap.std(axis=0)

            local_track_min = (mean_atac - std_atac).min()
            local_track_max = (mean_atac + std_atac).max()
            if local_track_min < track_global_min:
                track_global_min = local_track_min
            if local_track_max > track_global_max:
                track_global_max = local_track_max

            local_shap_min = (mean_shap - std_shap).min()
            local_shap_max = (mean_shap + std_shap).max()
            if local_shap_min < shap_global_min:
                shap_global_min = local_shap_min
            if local_shap_max > shap_global_max:
                shap_global_max = local_shap_max

            data_dict[(gex_cat, atac_cat)] = (mean_atac, std_atac, mean_shap, std_shap)
        
    # 左轴范围：不强制下界为 0，因为数据可能小于0
    # track_ymin, track_ymax 根据数据并加上少量余量
    track_margin = 0.05 * (abs(track_global_min) + abs(track_global_max))
    track_ymin = track_global_min - track_margin
    track_ymax = track_global_max + track_margin

    # 右轴原始范围（在原始 SHAP 数据上）：
    shap_margin  = 0.05 * (abs(shap_global_min) + abs(shap_global_max))
    shap_ymin = shap_global_min - shap_margin
    shap_ymax = shap_global_max + shap_margin

    # 接下来，在每个子图中绘制
    fig, axes = plt.subplots(nrows=3, ncols=3,
                             figsize=(10, 9),
                             sharex=True)

    for i, gex_cat in enumerate(gex_order):
        for j, atac_cat in enumerate(atac_order):
            ax_left = axes[i, j]
            mask = (gex_bin_series == gex_cat) & (atac_bin_series == atac_cat)
            n_genes = np.sum(mask)
            info = data_dict[(gex_cat, atac_cat)]

            ax_right = ax_left.twinx()

            if info is None:
                ax_left.set_title(f"GEx={gex_cat}, ATAC={atac_cat}\n(n=0)", fontsize=11)
                ax_left.set_xlim(-1300, 1300)
                ax_left.set_ylim(track_ymin, track_ymax)
                ax_right.set_ylim(shap_ymin, shap_ymax)
                ax_left.axvline(-1000, linestyle=':', color='gray')
                ax_left.axvline(1000,  linestyle=':', color='gray')
                ax_left.axvline(0,     linestyle=':', color='gray')
                continue

            mean_atac, std_atac, mean_shap, std_shap = info

            # 构造2600长度数组，将2000点数据嵌入中间
            mean_atac_full = np.full(full_len, np.nan)
            std_atac_full  = np.full(full_len, np.nan)
            mean_shap_full = np.full(full_len, np.nan)
            std_shap_full  = np.full(full_len, np.nan)

            mean_atac_full[data_left:data_right] = mean_atac
            std_atac_full[data_left:data_right]  = std_atac
            mean_shap_full[data_left:data_right] = mean_shap
            std_shap_full[data_left:data_right]  = std_shap

            # 绘制 ATAC track（左轴）
            ax_left.plot(
                positions_full, mean_atac_full,
                color='salmon', linewidth=2
            )
            ax_left.fill_between(
                positions_full,
                mean_atac_full - std_atac_full,
                mean_atac_full + std_atac_full,
                color='salmon', alpha=0.2
            )

            # ---------------------------
            # 修改部分：对 ATAC SHAP 进行缩放并调整右轴，使0与左轴0对齐
            # 首先将 SHAP 数据除以 1e-4
            shap_scaled = mean_shap_full / 1e-4
            std_shap_scaled = std_shap_full / 1e-4

            ax_right.plot(
                positions_full, shap_scaled,
                color='purple', linewidth=2
            )
            ax_right.fill_between(
                positions_full,
                shap_scaled - std_shap_scaled,
                shap_scaled + std_shap_scaled,
                color='purple', alpha=0.2
            )

            # 原始右轴范围（缩放后）为：
            Rmin = (shap_ymin) / 1e-4
            Rmax = (shap_ymax) / 1e-4

            # 计算左轴范围（不缩放）的下界和上界
            Lmin = track_ymin
            Lmax = track_ymax
            # 计算左轴中0相对于整体的比例：
            left_zero_frac = (0 - Lmin) / (Lmax - Lmin)

            # 现在右轴的 0 点在原始比例下为：
            right_zero_frac = (0 - Rmin) / (Rmax - Rmin)
            # 为了使右轴的0与左轴0对齐，我们计算需要的平移量 d：
            d = -left_zero_frac * (Rmax - Rmin) - Rmin
            new_Rmin = Rmin + d
            new_Rmax = Rmax + d

            ax_right.set_ylim(new_Rmin, new_Rmax)
            # 设定右轴刻度为整数
            tick_min = math.floor(new_Rmin)
            tick_max = math.ceil(new_Rmax)
            ticks = list(range(tick_min, tick_max+1))
            ax_right.set_yticks(ticks)
            ax_right.set_yticklabels([str(tick) for tick in ticks])
            ax_right.set_ylabel("ATAC SHAP (×10^-4)", fontsize=11, color='purple')
            # ---------------------------
            
            # 设置左轴 x 和 y 范围
            ax_left.set_xlim(-1300, 1300)
            ax_left.set_ylim(track_ymin, track_ymax)

            # 绘制竖直虚线：-1000, 0, 1000
            ax_left.axvline(-1000, linestyle=':', color='gray')
            ax_left.axvline(1000,  linestyle=':', color='gray')
            ax_left.axvline(0,     linestyle=':', color='gray')

            # 设置左轴、右轴颜色
            ax_left.tick_params(axis='y', colors='salmon')
            ax_left.spines['left'].set_color('salmon')
            ax_right.tick_params(axis='y', colors='purple')
            ax_right.spines['right'].set_color('purple')

            ax_left.set_title(f"GEx={gex_cat}, ATAC={atac_cat}\n(n={n_genes})", fontsize=11)
            if i == 2:
                ax_left.set_xlabel("Position relative to TSS (bp)", fontsize=11)
            if j == 0:
                ax_left.set_ylabel("ATAC track", fontsize=11, color='salmon')
            if j == 2:
                ax_right.set_ylabel("ATAC SHAP (×10^-4)", fontsize=11, color='purple')

    # 统一 x 轴刻度，只显示 -500, 0, 500
    for ax in axes[2, :]:
        ax.set_xticks([-500, 0, 500])

    # 图例 (两条线: ATAC track, ATAC SHAP)
    fig.legend(
        handles=[
            Line2D([0], [0], color='salmon', lw=2),
            Line2D([0], [0], color='purple', lw=2)
        ],
        labels=['ATAC track', 'ATAC SHAP'],
        loc='upper center',
        ncol=2,
        bbox_to_anchor=(0.5, 1.05)
    )

    plt.suptitle("ATAC track (left axis) vs. ATAC SHAP (right axis)\nShared scales per axis across 9 subplots",
                 y=0.99, fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Figure saved to {output_path}")

##############################################
# 3) 主程序：加载数据、分桶、绘图
##############################################
if __name__ == "__main__":
    PATH_TRAIN = "/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/data/26426404/train_0.npz"
    PATH_TEST  = "/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/data/26426404/test_0.npz"
    PATH_FULLMODEL_SHAP = "/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/shap/full_model_all_shap_link.npy"

    npz_paths = [PATH_TRAIN, PATH_TEST]
    samples, gex_series, auatac_series, gex_bin_series, atac_bin_series, atac_track = load_and_bin_data_multiple(npz_paths)

    fullmodelshap = np.load(PATH_FULLMODEL_SHAP)

    output_fig = "panelB_like_dualaxes.png"
    plot_panelB_like_dualaxes(
        atac_track=atac_track,
        fullmodelshap=fullmodelshap,
        gex_bin_series=gex_bin_series,
        atac_bin_series=atac_bin_series,
        output_path=output_fig
    )