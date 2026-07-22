# 17 類 YOLO 樂譜符號訓練與操作

## 已準備完成

- 模型架構：Ultralytics YOLO11n Detect
- 訓練資料：DeepScoresV2 dense
- train：4,545 tiles、23,926 個框
- validation：1,236 tiles、6,102 個框
- 完整驗證：已通過
- 資料位置：`articulation_experiments/outputs/yolo_dataset_expanded/`
- 訓練設定：`articulation_experiments/configs/expanded_symbols.yaml`

17 個 detector class 會正規化成 11 種 MusicXML 語意：accent、staccato、
tenuto、staccatissimo、marcato、fermata、caesura、trill、turn、
inverted turn、mordent。

## 開始訓練

最簡單的方法：直接雙擊專案根目錄的 `train_expanded_symbols.bat`。

它會先重新驗證資料，再開始 75 epochs 的 GPU 訓練。請不要關閉命令視窗。

也可以在 Anaconda Prompt 執行：

```powershell
cd C:\Cheewai\OMR_work\25-omr
C:\Users\885ar\anaconda3\envs\omr25-py311\python.exe articulation_experiments\train\train_baseline.py --config articulation_experiments\configs\expanded_symbols.yaml
```

## 訓練中斷後續訓

若電腦重開、程式中斷或需要稍後繼續，雙擊：

```text
resume_expanded_symbols.bat
```

它會從 `expanded_symbols_v1/weights/last.pt` 繼續。

## 訓練完成後

最佳權重會出現在：

```text
articulation_experiments/outputs/runs/expanded_symbols_v1/weights/best.pt
```

重新執行 `python pdf2musicXML.py` 時，主程式會優先使用這個 17 類權重，並依模型
類別數自動選擇 expanded mapping；不需要手動覆蓋原本的 baseline 權重。

若 expanded 權重不存在，主程式會繼續使用原本 6 類的
`baseline_v1/weights/best.pt`。

## 可調整項目

設定檔 `expanded_symbols.yaml` 中：

- `epochs`：目前 75。
- `batch`：目前 4；若顯示 CUDA out of memory，可改成 2。
- `device`：目前 0，代表第一張 NVIDIA GPU。
- `patience`：20 epochs 沒改善時提早停止。

不要在訓練途中更改 class mapping 或重新產生資料集，否則 checkpoint 的類別編號會
與新資料不一致。

## 成功判斷

看到 `Training completed successfully` 且 `best.pt` 存在，才算訓練完成。
之後先用 Beethoven 第 1 頁測試，檢查：

1. `articulations/*.articulations.jpg` 的框是否合理。
2. `articulations/*.articulations.json` 的類別與音符配對。
3. 最終 MusicXML 在 MuseScore 中是否出現新符號。
