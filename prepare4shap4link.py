import os
import glob
import re
import numpy as np

def process_folder_keep_3d(folder_path, output_file):
    """
    从 folder_path 中读取形如 "fold.seed.npy" 的文件(5 folds × 5 seeds)，
    仅对同一 fold 的 5 个 seed 做平均 (seed 维度)，得到 (n_genes, n_channels, n_positions)，
    再将 5 个 fold 的结果在基因维度 (axis=0) 上拼接，
    最终保存一个 3D 数组 (total_genes, n_channels, n_positions) 到 output_file。
    """
    file_list = glob.glob(os.path.join(folder_path, "*.npy"))
    if not file_list:
        print(f"[警告] 在 {folder_path} 下未找到任何 .npy 文件，跳过处理。")
        return

    # 按 fold 分组，fold -> [file1, file2, ...]
    fold_dict = {}
    for file_path in file_list:
        basename = os.path.basename(file_path)
        # 假设文件名类似 "0.0.npy" => fold=0, seed=0
        match = re.match(r"^(\d+)\.(\d+)\.npy$", basename)
        if match:
            fold = int(match.group(1))
            seed = int(match.group(2))
            fold_dict.setdefault(fold, []).append(file_path)
        else:
            print(f"[提示] 无法解析文件名中的 fold.seed: {basename}")

    fold_shap_list = []
    # 遍历 fold=0..4
    for fold in sorted(fold_dict.keys()):
        seed_files = fold_dict[fold]
        if len(seed_files) != 5:
            print(f"[警告] fold={fold} 下文件数不是 5 个，而是 {len(seed_files)} 个。可能数据不完整。")

        # 收集本 fold 的 5 个 seed 数组
        seed_arrays = []
        for sf in seed_files:
            arr = np.load(sf)  # 期望 shape=(n_genes_fold, n_channels, n_positions)
            seed_arrays.append(arr)

        # 拼成 (5, n_genes_fold, n_channels, n_positions)
        seed_arrays = np.stack(seed_arrays, axis=0)

        # 在种子维度 (axis=0) 上求平均 => (n_genes_fold, n_channels, n_positions)
        fold_mean = np.mean(seed_arrays, axis=0)
        fold_shap_list.append(fold_mean)

        print(f"Fold {fold}: 合并 {len(seed_files)} seeds => shape={fold_mean.shape}")

    # 将 5 个 fold 的结果在基因维度上拼接 => (total_genes, n_channels, n_positions)
    # total_genes = sum of each fold's n_genes_fold
    all_test_shap = np.concatenate(fold_shap_list, axis=0)
    print(f"拼接所有 fold 后 shape: {all_test_shap.shape}")

    # 注意：不再对基因做平均，保留 3D 数据
    np.save(output_file, all_test_shap)
    print(f"[完成] 已保存 3D SHAP 数组 => {output_file}\n")


def main():
    """
    示例：分别处理 "full_models" 和 "dna_models" 文件夹，生成 3D SHAP 数组
    如果文件夹名不同，请自行修改。
    """
    folder_full = "/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/shap/shap_output/full_models"  # full 模型 (有 5 通道)
    folder_dna  = "/Users/qiuhongyu/Desktop/scigogogo/jjZhang/reproduce/shap/shap_output/dna_models"   # DNA-only 模型 (有 4 通道)

    # 处理 full_models
    process_folder_keep_3d(folder_full, "full_model_all_shap_link.npy")

    # 处理 dna_models
    process_folder_keep_3d(folder_dna, "dna_model_all_shap_link.npy")


if __name__ == "__main__":
    main()