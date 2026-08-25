# 執行目前模型

本頁只說明如何使用現在整合的 Piano50 + curve-v2，不包含重新訓練。

## 1. 必要外部檔案

模型權重不在 GitHub。接手時至少要另外取得：

| 檔案 | 大小（bytes） | SHA256 |
|---|---:|---|
| Piano50 `best.pt` | 140,181,185 | `2814D9EC894816EF05E557E8E9D6F9644515F9B2F89D81A1AE737D2463D4899E` |
| curve-v2 `best.pt` | 140,103,489 | `EF683A592B89C9D66C7B69E6A7D8757B5CC14CD506227FDC7AA104D774DBAF59` |

目前原始 Windows 位置：

```text
C:\OMR_work\experiments\runs\yolov9_e_piano50_dense_parentheses_30ep_b4_3090\weights\best.pt
C:\OMR_work\experiments\runs\yolov9_e_curve_v2_fullbbox_2048_30ep_3090\weights\best.pt
```

此外需要：

- 官方 [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) checkout；
- Piano50 `dataset.yaml`（推論用類別名稱）；
- curve-v2 `dataset.yaml`；
- repository 內已提交的兩份 mapping；
- 使用文字方向時，需要 EasyOCR English model cache。

Dense all136 權重不是目前主程式必要檔案。它的 SHA256 是
`9C8BD565B572A69BD3A4A054A59D8C620AB066B81BF88F07E26B221ED4996336`，
只在 Complete continued training 或 all136 比較實驗時需要。

## 2. 環境

本機已驗證環境：

```text
Python: C:\Users\minemine\miniconda3\envs\omr\python.exe
Repository: C:\OMR_work\25-omr
YOLOv9: C:\OMR_work\yolov9
GPU: RTX 3090 24GB
```

新電腦請先安裝與 GPU/CUDA 相容的 PyTorch，再安裝 repository requirements。
RTX 5080 必須使用支援 Blackwell 的 PyTorch/CUDA build，不能沿用過舊 wheel。

## 3. 設定可攜路徑

模型模組支援以下環境變數：

```powershell
$env:OMR_PIANO50_WEIGHTS='D:\omr_assets\piano50\best.pt'
$env:OMR_PIANO50_DATA='D:\omr_assets\piano50\dataset.yaml'
$env:OMR_CURVE_V2_WEIGHTS='D:\omr_assets\curve_v2\best.pt'
$env:OMR_CURVE_V2_DATA='D:\omr_assets\curve_v2\dataset.yaml'
```

完整 MusicXML 使用曲目 JSON 內的 `weights`、`data_yaml`、`mapping`、
`yolov9_root` 與 `easyocr_model_dir`；可由
[pianoConfigTemplate.json](../../pianoConfigTemplate.json) 複製後修改。

確認權重沒有傳錯：

```powershell
Get-FileHash -Algorithm SHA256 'D:\omr_assets\piano50\best.pt'
Get-FileHash -Algorithm SHA256 'D:\omr_assets\curve_v2\best.pt'
```

## 4. 單頁偵測 smoke

```powershell
Set-Location 'C:\OMR_work\25-omr'
& 'C:\Users\minemine\miniconda3\envs\omr\python.exe' `
  '.\server\smoke_integrated_piano_page.py' `
  --image 'C:\path\to\piano-page.jpg' `
  --output-dir 'C:\OMR_work\experiments\handoff_smoke' `
  --device 0
```

它會輸出符號框、curve 框和 JSON，但不跑完整舊節奏分析，因此不能把這個輸出
當作 MusicXML accuracy。

## 5. 批次產生符號框與 JSON

```powershell
& 'C:\Users\minemine\miniconda3\envs\omr\python.exe' `
  '.\server\batch_integrated_piano_pages.py' `
  --input-dir 'C:\path\to\piano-pages' `
  --output-dir 'C:\OMR_work\experiments\handoff_batch' `
  --glob '*.jpg' --limit 20 --device 0
```

輸出包含可攜的 HTML gallery、每頁圖片及 JSON。curve 在這一步只代表視覺
候選，尚未經完整音符端點判定為 slur/tie。

## 6. 完整 PDF → MusicXML

1. PDF 放到 `string_dataset/pdf_data/<曲名>/<曲名>.pdf`。
2. 由 `pianoConfigTemplate.json` 建立同名 `<曲名>.json`，核對頁數與拍號。
3. 將曲名加入 `string_dataset/piecesToRun.json`。
4. 執行：

```powershell
& 'C:\Users\minemine\miniconda3\envs\omr\python.exe' `
  'C:\OMR_work\25-omr\pdf2musicXML.py'
```

主要輸出在 `string_dataset/output/<曲名>/`：MusicXML、articulation JSON/JPG、
slur/tie JSON/JPG。

鋼琴設定至少要有：

```json
{
  "score_mode": "piano",
  "numTrack": 2,
  "track_shift": [0, 0],
  "clef_options": [[1], [-1]]
}
```

匯入程式自動建立的 4/4 只是待人工確認的初始值；拍號錯誤會直接影響節奏與
MusicXML，不可略過。

## 7. 常見失敗

- 找不到 `best.pt`：設定上述環境變數或曲目 JSON 的絕對路徑。
- 找不到 YOLOv9：將官方 repo 放在 `25-omr` 同層，或改 `yolov9_root`。
- GPU 沒被使用：先用 Python 確認 `torch.cuda.is_available()` 和 GPU 名稱。
- 只有框、沒有 MusicXML 標籤：檢查舊 OMR 是否建立相對應 note/music21 object。
- 移動 gallery 後原圖消失：只使用新版批次工具生成的可攜輸出，不要依賴舊的
  絕對 `file:///C:/...` 連結。
