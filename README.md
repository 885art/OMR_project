# 弦樂四重奏 OMR

本程式將樂譜 PDF 轉換為 MusicXML，主辨識流程使用 OEMER segmentation model
與既有規則，並整合 YOLO articulation detector，自動辨識 accent、staccato、
tenuto 並配對到音符。

## 環境需求

- Python 3.9 以上（目前驗證環境為 Python 3.11）
- 建議使用支援 CUDA 的 NVIDIA GPU；沒有 GPU 時可改用 CPU

安裝套件：

```powershell
pip install -r requirements.txt
```

若版本固定檔發生相依性問題，可改用：

```powershell
pip install -r requirements-no-version.txt
```

## 執行內建範例

`string_dataset/piecesToRun.json` 預設包含 `beethoven1`，直接執行：

```powershell
python pdf2musicXML.py
```

輸出位置：

```text
string_dataset/output/beethoven1/
├─ beethoven1_1.xml                    # 各頁 MusicXML
├─ all_beethoven1_*.xml                # 合併後 MusicXML
└─ articulations/
   ├─ beethoven1_1.articulations.json  # 偵測及音符配對資料
   └─ beethoven1_1.articulations.jpg   # 可視化檢查圖
```

## 執行自己的樂譜

假設曲名為 `mozart1`：

1. 將 PDF 放到 `string_dataset/pdf_data/mozart1/mozart1.pdf`。
2. 依照 `jsonTemplate.json` 建立
   `string_dataset/pdf_data/mozart1/mozart1.json`。
3. 將 `"mozart1"` 加到 `string_dataset/piecesToRun.json`。
4. 執行 `python pdf2musicXML.py`。

設定檔主要欄位：

- `numPage`：PDF 頁數
- `numPerPage`：每張 PDF 頁面包含的樂譜頁數，預設為 1
- `rotate`：是否旋轉頁面
- `tsChange`：拍號及變更位置；第一筆必須是初始拍號
- `numTrack`：聲部數量
- `track_shift`：移調設定
- `clef_options`：各聲部可使用的譜號；1 為高音、0 為中音、-1 為低音、
  -2 為次中音譜號

## Articulation 設定

Articulation 辨識預設啟用，不必修改既有曲目設定。需要調整時，可在曲目的
JSON 加入：

```json
"articulation": {
  "enabled": true,
  "device": "auto",
  "confidence": 0.25,
  "class_confidence": {
    "accent": 0.55,
    "staccato": 0.45,
    "tenuto": 0.45
  }
}
```

`device` 設為 `"auto"` 會自動選擇可用裝置，也可指定 `"0"` 使用第一張 GPU
或 `"cpu"` 強制使用 CPU。若暫時不需要 articulation，可將 `enabled` 設為
`false`。

模型權重預設位置：

```text
articulation_experiments/outputs/runs/baseline_v1/weights/best.pt
```

快速檢查既有快取頁面、不重新執行完整 OMR：

```powershell
python articulation_experiments/inference/preview_cached_page.py `
  --piece beethoven1 --page 1 --device 0
```

本專案的 segmentation model 來源為 [OEMER](https://github.com/BreezeWhite/oemer)。
