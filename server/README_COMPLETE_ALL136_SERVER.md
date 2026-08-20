# Complete all136 伺服器訓練操作

這條流程只使用 DeepScores：

1. 已完成的 Dense all136 `best.pt`；
2. DeepScores Complete 的 136 類標註與圖片。

本階段**不讀取、不轉換、也不訓練 BPSD 資料**。Complete 不是從零訓練，
而是從 Dense all136 的最佳權重繼續訓練 30 epochs。

## 伺服器需要的內容

```text
25-omr/                         # 本 repository
yolov9/                         # WongKinYiu 官方 YOLOv9 repository
ds2_complete/
├── images/
├── deepscores-complete-0_train.json
├── ...                         # 共 103 個 train JSON
├── deepscores-complete-0_test.json
└── ...                         # 共 26 個 test JSON
weights/
└── yolov9_dense_all136_best.pt # 本機 Dense all136 的 best.pt
```

本機要上傳的 Dense 權重是：

```text
C:\OMR_work\experiments\runs\yolov9_e_dense_all136_30ep_b4_3090\weights\best.pt
```

資料集、切片、run 和 `.pt` 不要提交到 Git，應直接放在伺服器的高速
scratch/NVMe。

## 1. 建立私人環境檔

```bash
cp server/all136_env.example "$HOME/all136_env.sh"
```

編輯 `$HOME/all136_env.sh`，把 `PYTHON`、`YOLOV9_ROOT`、`WORK_ROOT`、
`COMPLETE_ROOT` 和 `INIT_WEIGHTS` 改成實際的絕對路徑。必須保留兩個不同輸出：

```bash
export ALL136_DATASET="$WORK_ROOT/datasets/deepscores_complete_all136_sharded"
export ALL136_SMOKE_DATASET="$WORK_ROOT/datasets/deepscores_complete_all136_sharded_smoke10"
```

絕對不能讓 smoke 和 full 指向同一資料夾。

依伺服器 CUDA 環境先安裝相容的 PyTorch，再安裝其餘套件：

```bash
source "$HOME/all136_env.sh"
"$PYTHON" -m pip install -r server/requirements-training.txt
bash server/check_complete_all136_server.sh "$HOME/all136_env.sh"
```

檢查程式會確認 103/26 個 JSON shards、Dense 權重、YOLOv9、Python 套件及
smoke/full 路徑隔離。若在登入節點執行，`cuda_available=False` 可以先接受；
真正的 GPU、PyTorch/CUDA 相容性必須在 GPU job 裡再次確認。

## 2. 先做伺服器 smoke

### 沒有 Slurm

```bash
source "$HOME/all136_env.sh"
SOURCE_KIND=complete MODE=smoke bash server/prepare_deepscores_all136.sh
bash server/train_yolov9_all136.sh smoke
```

這只轉換第一個 train/test shard 的各 10 頁，並訓練 1 epoch。成功條件是：

- 產生 `dataset.yaml`、`statistics.json`、`validation_report.json`；
- preflight 顯示 136 classes；
- smoke run 寫出 `weights/best.pt` 和 `weights/last.pt`；
- log 沒有 traceback、CUDA OOM 或 image/label mismatch。

### 使用 Slurm

先依中心規定在 `sbatch` 命令補上 account、partition 和 GPU 類型。例如：

```bash
sbatch --account=YOUR_ACCOUNT --partition=CPU_PARTITION \
  --export=ALL,ENV_FILE="$HOME/all136_env.sh",MODE=smoke \
  server/slurm_prepare_all136.sbatch
```

資料 job 完成後，再提交一張 GPU：

```bash
sbatch --account=YOUR_ACCOUNT --partition=GPU_PARTITION \
  --gres=gpu:1 \
  --export=ALL,ENV_FILE="$HOME/all136_env.sh",MODE=smoke \
  server/slurm_yolov9_all136.sbatch
```

若中心的 H100 資源名稱不是 `gpu:1`，依該中心規定改成例如
`--gres=gpu:h100:1`。不要猜 account/partition 名稱。

## 3. 轉換 Complete 全量資料

轉換主要使用 CPU、RAM 和磁碟，不需要占用 H100。它會逐 shard 轉換並可續跑：

```bash
source "$HOME/all136_env.sh"
SOURCE_KIND=complete MODE=full bash server/prepare_deepscores_all136.sh
```

Slurm：

```bash
sbatch --account=YOUR_ACCOUNT --partition=CPU_PARTITION \
  --export=ALL,ENV_FILE="$HOME/all136_env.sh",MODE=full \
  server/slurm_prepare_all136.sbatch
```

若 job 因時間限制中斷，提交同一條命令即可；正式資料集採分片 `--resume`。
只有看到 `ALL136 DATASET READY`，而且正式輸出內有
`validation_report.json`，才可以進入正式訓練。

## 4. Complete 正式訓練

先在 GPU allocation 內執行 preflight：

```bash
source "$HOME/all136_env.sh"
bash server/train_yolov9_all136.sh preflight
```

直接執行：

```bash
bash server/train_yolov9_all136.sh train
```

Slurm：

```bash
sbatch --account=YOUR_ACCOUNT --partition=GPU_PARTITION \
  --gres=gpu:1 \
  --export=ALL,ENV_FILE="$HOME/all136_env.sh",MODE=train \
  server/slurm_yolov9_all136.sbatch
```

預設為 YOLOv9-E、1024 輸入、30 epochs、early-stopping patience 8。H100
80GB 先用 batch 12；成功跑過數百 iterations 後可嘗試 16，OOM 就降到 8。

中斷後續訓：

```bash
source "$HOME/all136_env.sh"
export RESUME_CHECKPOINT="$WORK_ROOT/runs/$RUN_NAME/weights/last.pt"
bash server/train_yolov9_all136.sh resume
```

Slurm 續訓時將 `MODE=resume` 一起 export，並確認 `RESUME_CHECKPOINT` 已存在於
環境檔或提交環境。

## 5. 結果位置

```text
$WORK_ROOT/runs/$RUN_NAME/
├── results.csv
├── results.png
└── weights/
    ├── best.pt
    └── last.pt
```

報告時應說這是 DeepScores Complete validation 結果，不是鋼琴掃描譜或 BPSD
準確率。是否能改善真實鋼琴譜，需另外使用不參與訓練的固定頁面做視覺或標註
評估。
