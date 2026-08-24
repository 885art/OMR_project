# Complete all136 伺服器訓練操作

這條流程只使用 DeepScores：

1. 已完成的 Dense all136 `best.pt`；
2. DeepScores Complete 的 136 類標註與圖片。

本階段**不讀取、不轉換、也不訓練 BPSD 資料**。Complete 不是從零訓練，
而是從 Dense all136 的最佳權重做 continued training。H100 實測一個 train
epoch 約 30 小時，因此不再預設 30 epochs；完整 audit 與成本推導見
[`COMPLETE_ALL136_TRAINING_AUDIT.md`](COMPLETE_ALL136_TRAINING_AUDIT.md)。

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

若 Berlioz 使用 Docker，環境檔必須在 container 內載入，所有路徑都填 container
看得到的 mount 路徑；通常可設 `PYTHON=python`。不要把 host 路徑直接抄進
container，也不要在尚未知道 image／mount 設定時猜路徑。

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
COMPLETE_CONVERSION_WORKERS=1 SOURCE_KIND=complete MODE=smoke \
  bash server/prepare_deepscores_all136.sh
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

轉換主要使用 CPU、RAM 和儲存 I/O，不需要占用 H100。每個 worker 仍只處理一個
獨立 shard，但可以同時執行多個 shard converter。單工相容模式：

```bash
source "$HOME/all136_env.sh"
COMPLETE_CONVERSION_WORKERS=1 SOURCE_KIND=complete MODE=full \
  bash server/prepare_deepscores_all136.sh
```

Berlioz 有 96 logical CPU cores，但 conversion 同時會讀大型 JSON 並寫大量 PNG；
不要直接使用 96 workers。先從 4–8 開始，建議目前使用：

```bash
source "$HOME/all136_env.sh"
COMPLETE_CONVERSION_WORKERS=8 SOURCE_KIND=complete MODE=full \
  bash server/prepare_deepscores_all136.sh
```

Slurm：

```bash
sbatch --account=YOUR_ACCOUNT --partition=CPU_PARTITION \
  --export=ALL,ENV_FILE="$HOME/all136_env.sh",MODE=full \
  server/slurm_prepare_all136.sbatch
```

Slurm 使用時可在私人 `all136_env.sh` 設定
`export COMPLETE_CONVERSION_WORKERS=8`。這個變數只控制 Complete shard
conversion subprocess 數量，和 YOLO training 的 DataLoader `WORKERS` 無關。
完成 conversion 並產生通過的 `validation_report.json` 後，正式 YOLOv9
training 才使用 H100。

若 job 因時間限制中斷，提交同一條命令即可；正式資料集採分片 `--resume`。
每個 chunk 都會記錄來源 shard、class mapping、converter SHA256 與所有轉換
參數。只有 fingerprint 完全相同才會續跑或重用；不一致時程式會安全停止。
`COMPLETE_CONVERSION_WORKERS` 不屬於資料 recipe，從 1 改成 8 不會讓既有完成
chunks fingerprint mismatch。已完成的 chunk 會 reuse；中斷且 fingerprint 相同的
chunk（例如 `train_015`）會 resume，不需要刪除正式輸出或從頭轉換。
建議參數或程式改變後使用新的輸出資料夾。確定要重建既有 chunks 時才執行：

```bash
OVERWRITE_CHUNKS=1 SOURCE_KIND=complete MODE=full \
  bash server/prepare_deepscores_all136.sh
```

這會重新產生既有 chunks，不能把它當成一般續跑命令。
只有看到 `ALL136 DATASET READY`，而且正式輸出內有
`validation_report.json`，才可以進入正式訓練。

## 4. 先建立固定監控集與 Dense baseline

這一步只掃描現有 `train.txt`／`val.txt` 與 labels，建立輕量 index；不複製或
重切 179 GB tiles。固定監控集以 source page 為抽樣單位、涵蓋全部 136 類：

```bash
source "$HOME/all136_env.sh"
bash server/prepare_complete_all136_indexes.sh validation
VALIDATION_KIND=subset \
  VALIDATION_RUN_NAME=dense_best_complete_val_monitor50k \
  bash server/validate_complete_all136.sh
```

per-class baseline 位於：

```text
$WORK_ROOT/runs/validation/dense_best_complete_val_monitor50k/per_class_metrics.json
```

這個 50k subset 只用來快速比較 checkpoint，不可取代最後一次 full
validation。正式最終評估：

```bash
VALIDATION_KIND=full VALIDATION_RUN_NAME=final_complete_full \
  INIT_WEIGHTS=/path/to/final.pt bash server/validate_complete_all136.sh
```

## 5. Complete 訓練入口

先在 GPU allocation 內執行 preflight：

```bash
source "$HOME/all136_env.sh"
bash server/train_yolov9_all136.sh preflight
```

直接執行：

```bash
bash server/train_yolov9_all136.sh train
```

若只想先回答「完整 Complete 跑一輪後是否有改善」，使用獨立的 1-epoch pilot：

```bash
bash server/train_yolov9_all136.sh pilot_1epoch
```

它確實只跑一個完整 train epoch；若 `VALIDATION_POLICY=full_each_epoch`，最後再跑
一次完整 validation，估計約 34 小時。這不是把最終計畫永久限制為 1 epoch，
而是先取得可比較 checkpoint，再決定是否進一步訓練。正式 `train` 的預設仍是
2 epochs。

注意：官方 YOLOv9 不能把「已正常結束的 1-epoch run」原封不動延長為 2 epochs。
若 pilot 後要繼續，將 pilot 的 `last.pt`／`best.pt` 設成新 run 的
`INIT_WEIGHTS`；模型權重會接續，但 optimizer/scheduler 會重新初始化。因此報告
中應把它稱為第二個 continued-training stage，不可說成同一 run 的 exact resume。
若一開始就確定要保留同一 optimizer 狀態，直接用正式 2-epoch `train`。

若 baseline 顯示只有部分 classes 較差，可建立 class-aware + replay index：

```bash
export BASELINE_REPORT="$WORK_ROOT/runs/validation/dense_best_complete_val_monitor50k/per_class_metrics.json"
bash server/prepare_complete_all136_indexes.sh targeted
bash server/train_yolov9_all136.sh targeted
```

預設選出 `mAP@.5:.95 <= 0.50` 類別的 tiles，最多 400k target tiles，並加入
25% 非 target replay。入選 tile 的所有原始 136 類 labels 都保留，避免把好類別
當背景；這是第二階段實驗，不能冒充 full Complete all136 epoch。

Slurm：

```bash
sbatch --account=YOUR_ACCOUNT --partition=GPU_PARTITION \
  --gres=gpu:1 \
  --export=ALL,ENV_FILE="$HOME/all136_env.sh",MODE=train \
  server/slurm_yolov9_all136.sbatch
```

預設為 YOLOv9-E、1024 輸入、2 full epochs、`patience=0`、每完成一個 epoch
保存 snapshot，並做 full validation。Complete 專用 hyp 將 warmup 從 2.0
縮短為 0.1 epoch；其他 detection／augmentation 設定不變。H100 NVL 96GB
目前 batch 12 約使用 80.5 GB，batch 16 必須另做固定 batches throughput/OOM
測試，不能直接假設一定可用。

兩個 train epochs 加兩次 full validation 估計約 68 小時，原本 3-day Slurm
上限過於貼近；repository job 預留 4 days。若中心限制更短，不能靠假的
mid-epoch checkpoint 解決，必須取得能完成至少一整個 epoch+validation 的 allocation。

若只在最後一輪做完整 validation：

```bash
export VALIDATION_POLICY=full_final_only
export PATIENCE=0
```

預設 `DISABLE_PLOTS=1` 會傳官方 `--noplots`，避免對 291 萬 tiles 畫全量 label
統計圖。官方 run 仍會保存 `results.csv`、`opt.yaml`、`hyp.yaml` 與 weights。

中斷後續訓：

```bash
source "$HOME/all136_env.sh"
export RESUME_CHECKPOINT="$WORK_ROOT/runs/$RUN_NAME/weights/last.pt"
bash server/train_yolov9_all136.sh resume
```

Slurm 續訓時將 `MODE=resume` 一起 export，並確認 `RESUME_CHECKPOINT` 已存在於
環境檔或提交環境。

這個 resume 只保證回到上一個**已完成 epoch**。官方 checkpoint 沒有保存
current batch、AMP scaler、sampler／DataLoader worker RNG 與 prefetch queue；
epoch 中間中斷會重跑該 epoch，不能稱為 exact mid-epoch resume。repository
不提供只有存 `.pt`、實際卻從 batch 0 重跑的假功能。詳細限制與若要真正實作
所需條件見 training audit。

## 6. 3090／5080 本機測試界線

3090 或 5080 可以做 smoke、固定 subset validation 與小型 targeted index 的
1-epoch 流程測試。1024 輸入建議 RTX 3090 從 `BATCH_SIZE=4` 開始；RTX 5080
因 16 GB VRAM 從 `BATCH_SIZE=2` 開始，穩定後再試 4：

```bash
BATCH_SIZE=4 bash server/train_yolov9_all136.sh smoke  # RTX 3090
BATCH_SIZE=2 bash server/train_yolov9_all136.sh smoke  # RTX 5080
```

Windows 可使用既有 `.bat`；以 `OMR_DATASET_ROOT` 指向 smoke 或小型 targeted
index，不要指向 full dataset：

```powershell
# RTX 3090
$env:OMR_DATASET_ROOT='C:\path\to\small_index_dataset'
$env:OMR_BATCH_SIZE='4'
& 'C:\OMR_work\25-omr\server\train_yolov9_complete_all136_3090.bat' pilot1

# RTX 5080：同一入口，先從 batch 2 開始
$env:OMR_BATCH_SIZE='2'
```

它們適合驗證 CUDA／checkpoint／dataset 與顯存，不適合完整跑 233 萬 train
tiles，也不能用本機 smoke 的秒數直接推估 H100 全量成本。正式 full epoch 留給
H100。

## 7. 結果位置

```text
$WORK_ROOT/runs/$RUN_NAME/
├── results.csv
├── opt.yaml
├── hyp.yaml
└── weights/
    ├── best.pt
    ├── last.pt
    └── epoch*.pt

$WORK_ROOT/runs/_console_logs/$RUN_NAME.log
```

報告時應說這是 DeepScores Complete validation 結果，不是鋼琴掃描譜或 BPSD
準確率。是否能改善真實鋼琴譜，需另外使用不參與訓練的固定頁面做視覺或標註
評估。

目前 converter 將 Complete 官方 103 個 train shards 作為 YOLO train，26 個
官方 test shards 作為 YOLO val，供 continued pretraining 的 early stopping 與
模型選擇。因此這 26 個 shards 已不是 untouched test，不能把結果稱作正式
DeepScores test performance。若未來要發表正式 test 指標，需從官方 train 另做
固定 validation split，官方 test 只在最後評估一次；本輪伺服器訓練暫不改 split。
