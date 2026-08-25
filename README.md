# 25-OMR：鋼琴樂譜辨識與 MusicXML

本專案把既有 OMR25 音符／五線譜分析，與 YOLOv9 樂譜符號模型、曲線模型、
OCR 和音樂規則整合，目標是將印刷鋼琴譜轉成 MusicXML。

## 接手者從這裡開始

請先閱讀：

**[docs/handoff/README.md](docs/handoff/README.md)**

這一份會告訴你：

- 目前真正使用哪兩個模型；
- 現有結果可以證明什麼、不能宣稱什麼；
- 權重檔需要另外去哪裡取得；
- 如何跑單頁、批次標註和完整 MusicXML；
- 只有需要重新訓練時才要看的文件。

不需要一開始就閱讀根目錄所有歷史實驗文件。

## 目前整合狀態

目前 OMR25 預設整合：

1. **Piano50 YOLOv9-E**：articulation、dynamic、pedal、fingering、hairpin、
   tuplet 等 50 類候選。
2. **curve-v2 YOLOv9-E**：先偵測單一視覺 `curve` 類別，再依端點音符與音高
   規則判斷 slur／tie。
3. **受限字典 OCR**：處理 `cresc.`、`decresc.`、`dim.` 等文字方向。
4. **OMR25／music21 後處理**：把偵測候選連到音符並寫入 MusicXML。

一張真實鋼琴頁已完成整套 MusicXML smoke，但鋼琴多聲部、節奏、cross-staff
與曲線端點仍未達到可宣稱完整正確的程度。

老師要求的 **DeepScores Complete all136** 是另一個 continued-training 實驗，
目前 Complete tiles 已轉換完成，但 H100 正式訓練尚未完成。它不會自動取代
目前 Piano50 + curve-v2 的主程式模型。

## 最短執行方式

依 [執行目前模型](docs/handoff/RUN_CURRENT_MODELS.md) 設定外部權重後：

```powershell
& 'C:\Users\minemine\miniconda3\envs\omr\python.exe' `
  'C:\OMR_work\25-omr\server\smoke_integrated_piano_page.py' `
  --image 'C:\path\to\piano-page.jpg' `
  --output-dir 'C:\OMR_work\experiments\handoff_smoke' `
  --device 0
```

完整 PDF／多頁 MusicXML 仍由：

```powershell
& 'C:\Users\minemine\miniconda3\envs\omr\python.exe' `
  'C:\OMR_work\25-omr\pdf2musicXML.py'
```

實際曲目由 `string_dataset/piecesToRun.json` 決定。

## 文件分流

- [接手入口](docs/handoff/README.md)：一般接手者先看。
- [目前結果與限制](docs/handoff/CURRENT_RESULTS.md)：老師、報告或結果檢查。
- [執行目前模型](docs/handoff/RUN_CURRENT_MODELS.md)：需要實際跑推論的人看。
- [重新訓練入口](docs/handoff/RETRAINING.md)：只有需要重訓才看。
- [完整 AI／工程歷史](CHATGPT_PROJECT_CONTEXT.md)：除錯或追溯決策時再看。

## Git 與大型檔案

目前開發 branch 是 `feature/yolov9-migration`。模型權重、datasets、runs、推論
圖片與私人伺服器設定不存 Git；GitHub 只保存程式、設定模板、文件與小型測試。

```powershell
git branch --show-current
git status
git log -1 --oneline
```
