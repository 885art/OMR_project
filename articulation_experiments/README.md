# Articulation 辨識原型

本目錄包含獨立的六類 articulation 物件偵測器。此原型不會修改
`pdf2musicXML.py`、OEMER 模型或目前的 MusicXML builder。

## 固定類別順序

1. `articAccentAbove`
2. `articAccentBelow`
3. `articStaccatoAbove`
4. `articStaccatoBelow`
5. `articTenutoAbove`
6. `articTenutoBelow`

YOLO class ID 依序固定為 `0..5`。正式 mapping 位於
`dataset/class_mapping.json`，不得重新排列。

## 資料集

DeepScoresV2 的 AABB 會被轉換成 1024×1024 tiles，tile overlap 為 256
pixels。原始 source bbox 會完整保留於 manifest，確保每個訓練框都能追溯回
DeepScores annotation。

由於原始 Tenuto bbox 的高度通常只有 1–2 pixels，訓練用 bbox 會在保持中心
位置的情況下，垂直擴張到可設定的最小高度；目前正式設定為 8 pixels。原始
bbox 與訓練 bbox 會分開記錄，不會覆蓋原始標註。

```powershell
python articulation_experiments/dataset/convert_deepscores_to_yolo.py `
  --minimum-tenuto-bbox-height-pixels 8 --overwrite
python articulation_experiments/dataset/validate_yolo_dataset.py
```

正式資料集輸出位置：

```text
articulation_experiments/outputs/yolo_dataset/
```

## 訓練

可重現的 YOLO11n baseline 設定位於 `configs/baseline.yaml`。訓練開始前會對
完整資料集執行 preflight validation，並支援從 `weights/last.pt` 繼續訓練。

```powershell
python articulation_experiments/train/train_baseline.py
```

從中斷點續訓：

```powershell
python articulation_experiments/train/train_baseline.py `
  --resume articulation_experiments/outputs/runs/baseline_v1/weights/last.pt
```

Baseline 主要設定：

- 模型：YOLO11n Detect
- 輸入尺寸：1024
- Batch size：4
- Epochs：50
- Optimizer：AdamW
- 固定 random seed：`20260717`
- 關閉 mosaic、flip、scale 等可能破壞微小樂譜符號的 augmentation
- Train 與 validation 使用 DeepScoresV2 原始 split，不重新切分

正式 checkpoint 與訓練報告位於：

```text
articulation_experiments/outputs/runs/baseline_v1/
```

## Validation 深度評估

執行完整 validation error analysis：

```powershell
python articulation_experiments/evaluation/evaluate_validation.py
python articulation_experiments/evaluation/summarize_metrics.py `
  --input articulation_experiments/outputs/evaluation/baseline_v1_validation/evaluation_summary.json `
  --output articulation_experiments/outputs/evaluation/baseline_v1_validation/REPORT.md
```

評估內容包括：

- 固定 confidence／IoU 門檻下的 TP、FP、FN
- 每類 Precision、Recall、F1
- Accent、Staccato、Tenuto 語意合併指標
- Above／below 方向混淆
- Dataset-specific bbox area buckets
- False-positive／false-negative 錯誤圖
- 實測推論速度

Ultralytics AP50 與 AP50-95 仍記錄於：

```text
articulation_experiments/outputs/runs/baseline_v1/baseline_summary.json
```

詳細評估結果位於：

```text
articulation_experiments/outputs/evaluation/baseline_v1_validation/
```

## 推論

### 單張圖片

```powershell
python articulation_experiments/inference/infer_image.py `
  --input IMAGE `
  --output predictions.json `
  --annotated-output predictions.png
```

### 整頁 overlapping tiled inference

```powershell
python articulation_experiments/inference/infer_tiled_page.py `
  --input PAGE `
  --output-dir articulation_experiments/outputs/predictions/PAGE
```

整頁推論流程會：

1. 將頁面切成 1024×1024 tiles。
2. 使用 256-pixel overlap。
3. 對每個 tile 執行 YOLO11n inference。
4. 將 tile bbox 還原成 source-image coordinates。
5. 使用 class-aware NMS 合併重疊區域的重複預測。
6. 輸出 raw tile detections、merged detections、標註圖及 candidate JSON。

Candidate JSON 包含：

- 原始六類 class name
- Normalized semantic class：accent／staccato／tenuto
- Above／below side
- Source-image bbox
- Confidence
- Tile provenance
- Merge method
- 尚未配對的 note／NoteGroup association 欄位

目前 recognition 與 association 保持分離；candidate 的
`association_status` 預設為 `unmatched`。

## 自建 `yolo/` 資料夾

執行檢查：

```powershell
python articulation_experiments/evaluation/inspect_custom_yolo.py
```

目前 `25-omr/yolo/` 的二類 YOLO 標註格式正確，但資料夾中沒有 class-name
metadata，例如：

- `dataset.yaml`
- `classes.txt`
- `obj.names`
- 可證明 class mapping 的 README

因此不得只依 numeric class 0／1 猜測 articulation 語意。既有 debug overlays
也顯示這些框主要位於譜號與調號附近，與本專案的 articulation ground truth
不是同一項標註任務。

檢查程式會：

- 驗證 image-label 配對
- 驗證 YOLO label 格式與 bbox 範圍
- 統計 class 0／1 數量及 bbox 尺寸
- 使用 articulation detector 執行未評分的 domain inference
- 保留原始 numeric class，不進行猜測式 mapping

在取得真正的 articulation ground truth 與可驗證 class mapping 前，不會計算
這批資料的 articulation per-class AP。

## 主要輸出

```text
articulation_experiments/outputs/
├─ yolo_dataset/                         # 正式 train/validation tiles
├─ runs/baseline_v1/                    # Baseline checkpoint 與訓練結果
├─ evaluation/baseline_v1_validation/   # Validation 深度評估
├─ evaluation/custom_yolo/              # 自建資料檢查與 domain inference
└─ predictions/                         # 單圖及整頁推論輸出
```

## 目前範圍

已完成：

- DeepScoresV2 articulation inspection
- YOLO tile conversion 與資料驗證
- Small overfit sanity check
- YOLO11n baseline training
- Validation error analysis
- 單圖與整頁 tiled inference
- Class-aware NMS
- Association-ready candidate JSON
- 自建 YOLO 資料檢查
- Articulation-to-note／NoteGroup association
- `25-omr` 主 pipeline
- MusicXML articulation 輸出

主流程預設會執行 articulation detector，依譜表、上下方向、水平與垂直距離
將候選符號配對到 NoteGroup，並透過 music21 寫入 MusicXML。每頁同時輸出
association JSON 與標框圖，方便檢查實際結果。完整執行方式請參考專案根目錄的
`README.md`。
