# 國網中心訓練流程：YOLOv9 tiny-object v2

## 已確定的設計

- 主要模型：老師指定的官方 WongKinYiu YOLOv9-S。
- 比較模型：YOLO11n，只跑 30 epochs 對照，不先做完整訓練。
- 40 類 symbol：從原樂譜切 `512×512`，訓練時放大成 `1024×1024`。
- staccato 原始框中位數約 `6×6`，放大後約為 `12×12`。
- slur／tie 暫時維持原本 `1024×1024` 流程，避免把兩種問題混在一起。
- train／validation 依原始頁面隔離；完整資料的重複頁會先去重。

切圖大小與模型輸入大小必須成對使用。新版權重推論時必須設定：

```text
tile_size=512
overlap=128
model_input_size=1024
edge_policy=shift
```

舊版權重仍使用 `1024 / 256 / 1024`。

## 國網磁碟內容

建議在高速 scratch 建立：

```text
omr/
├─ repos/
│  ├─ 25-omr/
│  └─ yolov9/
├─ data/
│  └─ ds2_complete/
│     ├─ images/
│     ├─ deepscores-complete-73_train.json
│     └─ ...
├─ weights/
│  ├─ yolov9-s.pt
│  └─ yolo11n.pt
└─ experiments/
   ├─ datasets/
   └─ runs/
```

完整資料集、產生的 tiles、模型權重都不要提交到 GitHub。

## 1. 設定環境

複製範例：

```bash
cp server/nchc_env.example ~/nchc_env.sh
```

編輯 `~/nchc_env.sh`，填入實際的 Python、YOLOv9、DeepScores、權重及
高速磁碟路徑，然後載入：

```bash
source ~/nchc_env.sh
```

先依國網 GPU／CUDA 環境安裝相容的 PyTorch，再安裝訓練用套件：

```bash
"$PYTHON" -m pip install -r server/requirements-training.txt
```

`requirements.txt` 內含 Windows 專用套件，不建議直接拿來建立國網 Linux
訓練環境。

確認 GPU：

```bash
"$PYTHON" -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 2. 產生完整 tiny-object v2 dataset

DeepScores complete 和 dense 的 JSON 結構不同。這個步驟會：

1. 讀取 40 個目標類別的 train/test 分片。
2. 只保留 40 類需要的 annotation。
3. 以原始圖片檔名去除重複頁。
4. 移除 train/validation 重疊頁。
5. 產生 512 tiles、YOLO labels、manifest 和統計。
6. 執行完整資料驗證。

```bash
bash server/prepare_symbols_v2.sh
```

第一次正式執行前可先做小型 smoke dataset：

```bash
MAX_SHARDS=2 MAX_IMAGES_PER_SHARD=10 \
DATASET_ROOT="$WORK_ROOT/datasets/symbols_tiny_v2_smoke" \
MERGED_ROOT="$WORK_ROOT/datasets/deepscores_complete_smoke" \
bash server/prepare_symbols_v2.sh
```

smoke 與正式資料要使用不同資料夾。正式執行不要設定 `MAX_SHARDS` 或
`MAX_IMAGES_PER_SHARD`。

## 3. 訓練順序

先確認一個 epoch 可以完整跑完：

```bash
bash server/train_yolov9_symbols.sh smoke
```

再做 30 epochs pilot：

```bash
bash server/train_yolov9_symbols.sh pilot
```

先拿 pilot 的 `best.pt` 測真實 Beethoven 頁面。如果 staccato 有改善，再跑：

```bash
bash server/train_yolov9_symbols.sh full
```

正式預設為 100 epochs。單張 GPU 可使用 `BATCH_SIZE=-1` 自動估算；多 GPU
時必須手動設定總 batch size，且不要啟用 `USE_IMAGE_WEIGHTS=1`。

中斷後續訓：

```bash
RESUME_CHECKPOINT=/path/to/run/weights/last.pt \
bash server/train_yolov9_symbols.sh resume
```

單獨驗證最佳權重：

```bash
EVAL_WEIGHTS=/path/to/run/weights/best.pt \
bash server/train_yolov9_symbols.sh validate
```

## 4. YOLO11 對照

YOLOv9 pilot 完成後才需要跑：

```bash
bash server/train_yolo11_compare.sh
```

兩個模型使用同一份 tiny-v2 dataset、相同 `1024` 輸入和 30 epochs，因此結果
才可直接比較。YOLO11 暫時不跑完整 100 epochs。

## 5. Slurm 提交

如果國網使用 Slurm，先依該機器規則調整
`server/slurm_yolov9_symbols.sbatch` 的 GPU、記憶體、時間、partition 和 account：

```bash
ENV_FILE="$HOME/nchc_env.sh" MODE=pilot \
sbatch server/slurm_yolov9_symbols.sbatch
```

## 6. 把權重拿回本機測試

下載 `best.pt` 和該 dataset 的 `dataset.yaml`，執行：

```powershell
python preview_yolov9_all.py `
  --piece beethoven1 --page 1 `
  --symbol-weights C:\path\to\best.pt `
  --symbol-data-yaml C:\path\to\dataset.yaml `
  --symbol-tile-size 512 `
  --symbol-overlap 128 `
  --symbol-input-size 1024
  --symbol-edge-policy shift
```

比較時不要只看 DeepScores validation mAP；還要固定使用相同 Beethoven 頁面、
confidence threshold 和人工檢查方式，紀錄 staccato 的漏判與誤判。
