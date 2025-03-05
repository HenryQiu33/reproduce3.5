# Figure 2 实验配置说明

## 1. Unscrambled (ablation)
### 1.1 DNA + ATAC
```json
{
    "include_dna": true,
    "include_atac": true,
    "train_shuffle_mode": "none",
    "test_shuffle_mode": "none",
    "batch_size": 512,
    "max_epochs": 500,
    "lr": 5e-05,
    "wd": 0.001
}
```

### 1.2 ATAC only
```json
{
    "include_dna": false,
    "include_atac": true,
    "train_shuffle_mode": "none",
    "test_shuffle_mode": "none",
    "batch_size": 512,
    "max_epochs": 500,
    "lr": 5e-05,
    "wd": 0.001
}
```

### 1.3 DNA only
```json
{
    "include_dna": true,
    "include_atac": false,
    "train_shuffle_mode": "none",
    "test_shuffle_mode": "none",
    "batch_size": 512,
    "max_epochs": 500,
    "lr": 5e-05,
    "wd": 0.001
}
```

## 2. Scrambled train; unscrambled test
### 2.1 ATAC + scrambled DNA
```json
{
    "include_dna": true,
    "include_atac": true,
    "train_shuffle_mode": "dna",
    "test_shuffle_mode": "none",
    "batch_size": 512,
    "max_epochs": 500,
    "lr": 5e-05,
    "wd": 0.001
}
```

### 2.2 DNA + scrambled ATAC
```json
{
    "include_dna": true,
    "include_atac": true,
    "train_shuffle_mode": "atac",
    "test_shuffle_mode": "none",
    "batch_size": 512,
    "max_epochs": 500,
    "lr": 5e-05,
    "wd": 0.001
}
```

### 2.3 Scrambled DNA + scrambled ATAC (pairwise)
```json
{
    "include_dna": true,
    "include_atac": true,
    "train_shuffle_mode": "pairwise",
    "test_shuffle_mode": "none",
    "batch_size": 512,
    "max_epochs": 500,
    "lr": 5e-05,
    "wd": 0.001
}
```

### 2.4 Scrambled DNA + scrambled ATAC (separate)
```json
{
    "include_dna": true,
    "include_atac": true,
    "train_shuffle_mode": "separate",
    "test_shuffle_mode": "none",
    "batch_size": 512,
    "max_epochs": 500,
    "lr": 5e-05,
    "wd": 0.001
}
```

## 3. Unscrambled train; scrambled test
### 3.1 ATAC + scrambled DNA
```json
{
    "include_dna": true,
    "include_atac": true,
    "train_shuffle_mode": "none",
    "test_shuffle_mode": "dna",
    "batch_size": 512,
    "max_epochs": 500,
    "lr": 5e-05,
    "wd": 0.001
}
```

### 3.2 DNA + scrambled ATAC
```json
{
    "include_dna": true,
    "include_atac": true,
    "train_shuffle_mode": "none",
    "test_shuffle_mode": "atac",
    "batch_size": 512,
    "max_epochs": 500,
    "lr": 5e-05,
    "wd": 0.001
}
```

### 3.3 Scrambled DNA + scrambled ATAC (pairwise)
```json
{
    "include_dna": true,
    "include_atac": true,
    "train_shuffle_mode": "none",
    "test_shuffle_mode": "pairwise",
    "batch_size": 512,
    "max_epochs": 500,
    "lr": 5e-05,
    "wd": 0.001
}
```

### 3.4 Scrambled DNA + scrambled ATAC (separate)
```json
{
    "include_dna": true,
    "include_atac": true,
    "train_shuffle_mode": "none",
    "test_shuffle_mode": "separate",
    "batch_size": 512,
    "max_epochs": 500,
    "lr": 5e-05,
    "wd": 0.001
}
```

## 运行命令
对于每个配置，将对应的 JSON 内容保存为 `settings.json`，然后运行：
```bash
python 1_train_model_cv.py -m your_model -i input_dir -o output_dir -g 0 -s settings.json
```