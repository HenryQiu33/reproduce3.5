import numpy as np
import glob

def merge_test_atac(output_file="atac_merged.npy"):
    test_files = sorted(glob.glob("test_*.npz"))
    print("即将合并的测试文件:", test_files)

    all_atac = []
    for f in test_files:
        data = np.load(f)
        # 假设 ATAC 的键名为 "atac"
        # 若名字不同，比如 "rna"、"sequence" 等，需要先打印 data.keys() 查看
        arr = data["atac"]   # shape=(3819, 2000)
        all_atac.append(arr)

    # 在样本维度上拼接 => (5×3819, 2000) = (19095, 2000)
    merged_atac = np.concatenate(all_atac, axis=0)
    print("合并后的 ATAC shape:", merged_atac.shape)  # (19095, 2000)

    # 保存
    np.save(output_file, merged_atac)
    print(f"[完成] 已保存合并后的 ATAC => {output_file}")


if __name__ == "__main__":
    merge_test_atac("atac_merged.npy")