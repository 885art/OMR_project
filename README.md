# 25-OMR：樂譜 PDF 轉 MusicXML

本專案以 OEMER 的 segmentation 結果辨識音符與五線譜結構，再加入符號偵測及
rule-based 配對，輸出 MusicXML。擴充模組目前包含：

- 40 類 YOLO 樂譜符號偵測
- slur／tie 曲線偵測及端點配對
- MusicXML articulation、direction、slur、tie 寫入
- 每頁 JSON 與標註預覽圖，方便檢查偵測結果

## 環境

目前使用的 Conda 環境：

```powershell
conda activate omr25-py311
pip install -r requirements.txt
```

如果不需要固定每個套件版本，可改裝：

```powershell
pip install -r requirements-no-version.txt
```

YOLOv9 使用老師指定的官方程式庫：

```text
C:\Cheewai\OMR_work\yolov9
```

來源為 [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)。此 repo 放在
`25-omr` 的外層，不會複製進本專案。

## 一般執行方式

設定 `string_dataset/piecesToRun.json` 之後，在專案根目錄執行：

```powershell
conda activate omr25-py311
python pdf2musicXML.py
```

預設範例為 `beethoven1`。主要輸出位於：

```text
string_dataset/output/beethoven1/
├─ beethoven1_1.xml
├─ all_beethoven1_*.xml
├─ articulations/
│  ├─ beethoven1_1.articulations.json
│  └─ beethoven1_1.articulations.jpg
└─ slur_tie/
   ├─ beethoven1_1.slurs_ties.json
   └─ beethoven1_1.slurs_ties.jpg
```

要測試其他樂譜時：

1. 把 PDF 放到 `string_dataset/pdf_data/<曲名>/<曲名>.pdf`。
2. 依照既有範例建立同名 JSON 設定檔。
3. 把曲名加入 `string_dataset/piecesToRun.json`。
4. 執行 `python pdf2musicXML.py`。

鋼琴譜可以使用，但多聲部、跨譜表、密集和弦與重疊曲線仍是較困難的案例。

### 匯入 BPSD JPEG 鋼琴頁

`server/import_bpsd_piano_images.py` 會依圖片檔名最後的頁碼，自動把 BPSD
頁面分組成 OMR25 所需的資料夾、PNG 與鋼琴 JSON。原始圖片不會被移動或刪除。

先只檢查分組，不寫入檔案：

```powershell
Set-Location 'C:\OMR_work\25-omr'
& 'C:\Users\minemine\miniconda3\envs\omr\python.exe' `
  '.\server\import_bpsd_piano_images.py'
```

確認後正式匯入：

```powershell
& 'C:\Users\minemine\miniconda3\envs\omr\python.exe' `
  '.\server\import_bpsd_piano_images.py' --apply
```

預設來源是：

```text
string_dataset/pdf_data/BPSD_score_scan_jpeg/
```

匯入後會產生：

- 每首作品各自的資料夾、`<曲名>.json` 與編號 PNG。
- `string_dataset/pdf_data/bpsd_import_manifest.json`：原始檔名與輸出頁面對照。
- `string_dataset/pdf_data/piecesToRun.bpsd.json`：所有匯入作品名稱。

產生的 JSON 暫時使用 4/4，並標記 `time_signature_status` 為
`needs_manual_review`。執行 MusicXML 轉換前必須核對每首作品的 `tsChange`。
一般重跑匯入程式會保留已存在的 JSON，避免覆蓋人工修改；只有明確加上
`--overwrite` 才會重建圖片與設定。

## YOLOv9 架構

這次遷移採用官方 YOLOv9-S，分成兩個模型訓練：

| 模型 | 類別 | 訓練資料 |
|---|---:|---|
| symbol detector | 40 類 | DeepScoresV2 符號 tiles |
| curve detector | 2 類（slur、tie） | DeepScoresV2 曲線 tiles |

分開訓練是為了避免數量很多的 slur／tie 壓過其他符號，也方便分別調整 confidence。
curve dataset 已通過逐檔來源驗證：

- train：9,573 tiles、34,137 個 tile instances
- validation：2,427 tiles、8,718 個 tile instances
- 總計：12,000 tiles、42,855 個 tile instances
- 原始資料有 4 個零面積 curve bbox，已在 class mapping 和 manifest 中明確記錄並排除

YOLOv9 只負責找出曲線候選框。最後判斷 MusicXML 的 tie 或 slur，仍會參考：

- 曲線左右端點
- staff 位置
- 相鄰音符距離
- 音高是否相同
- 曲線方向與重疊關係

因此 slur／tie 是「YOLO 候選偵測 + 音樂規則配對」，不是只看 YOLO 類別名稱。

## 開始 YOLOv9 訓練

正式訓練尚未自動啟動。要開始時，只需雙擊：

[train_yolov9_all.bat](train_yolov9_all.bat)

它會依序執行：

1. 檢查官方 YOLOv9、pretrained weights、CUDA 與兩份 dataset 驗證報告。
2. 訓練 40 類 symbol detector。
3. 訓練 2 類 slur／tie detector。

目前設定為 YOLOv9-S、1024×1024、batch size 2、60 epochs、RTX 3060
（`device 0`）。兩個模型串接訓練可能需要約 10–16 小時，實際時間依 GPU 溫度、
資料讀取速度與 early stopping 而定。

若訓練被中斷，雙擊：

[resume_yolov9_all.bat](resume_yolov9_all.bat)

訓練完成後的權重位置：

```text
articulation_experiments/outputs/runs/yolov9_symbols_v1/weights/best.pt
slur_tie_experiments/outputs/runs/yolov9_curves_v1/weights/best.pt
```

兩份 `best.pt` 存在時，主程式的 `backend: "auto"` 會優先使用 YOLOv9；尚未訓練
完成之前，symbol 模組會沿用既有 Ultralytics 權重，slur／tie 模組會沿用 OpenCV
prototype，所以不會假裝 YOLOv9 已經有結果。

## 推論設定

曲目 JSON 可加入：

```json
{
  "articulation": {
    "enabled": true,
    "backend": "auto",
    "device": "0",
    "confidence": 0.25,
    "class_confidence": {
      "accent": 0.55,
      "staccato": 0.45,
      "tenuto": 0.45
    }
  },
  "slur_tie": {
    "enabled": true,
    "backend": "auto",
    "device": "0",
    "confidence": 0.25,
    "max_tie_span_units": 6.5,
    "visualize_pages": [1]
  }
}
```

`device` 可使用 `"0"`（第一張 NVIDIA GPU）、`"cpu"` 或 `null`。RTX 3060 建議用
`"0"`。

## 已知限制

- DeepScoresV2 與實際掃描樂譜存在 domain gap；合成資料驗證高分不代表所有實譜都準確。
- 很長、重疊、跨行或跨 system 的 slur／tie 仍可能合併、漏掉或連錯音。
- dynamics、文字與相似符號容易誤判，需要真實樂譜 hard-negative 資料再微調。
- OEMER 若先把音高、staff 或小節線辨識錯，後處理無法保證完全修正。
- 模型偵測框與 MusicXML 語意之間仍需要規則，不能只靠 object detection。

## Git

查看目前分支與修改：

```powershell
git branch --show-current
git status
```

這次 YOLOv9 工作位於：

```text
feature/yolov9-migration
```

大型 dataset、訓練輸出與權重不應提交到一般 Git commit；程式、設定、操作腳本與文件
才會進入版本控制。

## Tiny-object v2 與國網訓練

目前 staccato 的 DeepScores 原始框中位數約為 `6×6 px`，小於 YOLOv9
最細 detection feature 的 stride 8。新版資料流程因此使用：

```text
原圖切片 512×512 → 模型輸入 1024×1024
```

這會把 staccato 放大至約 `12×12 px`。訓練與推論必須同時使用相同設定，
不能只改訓練端。

國網正式模型使用精度優先的官方 YOLOv9-E；先前已完成的本機實驗仍保留
YOLOv9-S 紀錄，不會覆寫舊權重。

本機用 dense 資料建立新版 dataset：

[prepare_tiny_symbols_v2.bat](prepare_tiny_symbols_v2.bat)

完整 DeepScoresV2、YOLOv9 pilot／正式訓練、YOLO11 對照及 Slurm 操作請參考：

[server/README_國網訓練.md](server/README_國網訓練.md)
