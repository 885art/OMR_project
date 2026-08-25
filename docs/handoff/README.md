# 25-OMR 接手入口

這是後續接手者的主要入口。先看本頁即可，不需要先讀完所有訓練紀錄。

## 一分鐘版本

- 專案目標：把印刷鋼琴譜轉成 MusicXML。
- 目前主程式使用 **Piano50 符號模型 + curve-v2 曲線模型 + OCR + 音樂規則**。
- 兩個 `.pt` 權重不在 GitHub，必須另外複製；檔案雜湊記在
  [RUN_CURRENT_MODELS.md](RUN_CURRENT_MODELS.md)。
- DeepScores validation 分數是合成來源域結果，不是鋼琴實譜準確率。
- Complete all136 tiles 已完成轉換，但 H100 正式模型尚未完成。
- 若只是使用現有成果，不必讀 Complete conversion 或 training audit。

## 你現在要做什麼？

### 只想看成果或向老師說明

閱讀 [CURRENT_RESULTS.md](CURRENT_RESULTS.md)。

### 想在新電腦跑一張鋼琴譜

閱讀 [RUN_CURRENT_MODELS.md](RUN_CURRENT_MODELS.md)。它包含外部權重、環境變數、
單頁／批次／MusicXML 指令和輸出位置。

### 想繼續訓練或使用 H100

閱讀 [RETRAINING.md](RETRAINING.md)。該頁只提供選擇入口，再連到完整訓練文件。

### 程式出問題，需要了解為什麼這樣設計

依序閱讀：

1. [OMR25_PIANO_INTEGRATION.md](../../OMR25_PIANO_INTEGRATION.md)
2. [CHATGPT_PROJECT_CONTEXT.md](../../CHATGPT_PROJECT_CONTEXT.md)
3. `AGENTS.md`（使用 ChatGPT／Codex 修改 repository 時必讀）

## 目前真正使用的模型

| 用途 | 模型 | 狀態 |
|---|---|---|
| 50 類樂譜符號 | YOLOv9-E Piano50 | 已訓練，現在的整合預設 |
| slur／tie 視覺曲線 | YOLOv9-E curve-v2，單一 `curve` 類 | 已訓練，現在的整合預設 |
| 136 類 DeepScores Dense | YOLOv9-E all136 | 已訓練，Complete 的起始權重；不是主程式預設 |
| 136 類 Complete | Dense → Complete continued training | tiles 已完成；正式 H100 模型尚未完成 |

## 絕對不要誤解的三件事

1. 方框旁的 confidence 是模型候選分數，不是整頁準確率。
2. DeepScores mAP 不能當成真實鋼琴譜 mAP。
3. 偵測到符號不代表一定能寫入 MusicXML；必須先有正確的音符、staff、節奏與
   可掛載的 music21 物件。

## GitHub 與外部檔案邊界

GitHub 保存：程式、mapping、設定模板、文件與測試。

GitHub 不保存：`.pt`、DeepScores/BPSD 圖片、轉換 tiles、runs、EasyOCR 模型、
私人伺服器路徑與帳密。接手時必須另外交付權重和需要的資料。

查看當前版本時以 Git 為準：

```bash
git branch --show-current
git log -1 --oneline
git status
```

不要把文件中歷史 commit SHA 當成固定程式設定。
