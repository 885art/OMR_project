# 目前結果與限制

更新日期：2026-11-26（Asia/Taipei）

## 結論先說

目前最適合展示與繼續整合的組合是：

- Piano50 YOLOv9-E：符號候選；
- curve-v2 YOLOv9-E：完整曲線候選；
- EasyOCR 限制字典：`cresc.`／`decresc.`／`dim.`；
- OMR25 規則：符號與音符關聯、slur/tie 判斷及 MusicXML 輸出。

這套流程已在真實鋼琴頁實際跑通並產生可解析 MusicXML，但尚未完成真實鋼琴
ground truth 的正式 accuracy 評估，因此不能說「鋼琴辨識已完成」。

## 模型結果

| 模型 | 訓練結果 | 正確解讀 |
|---|---|---|
| Piano50 | DeepScores validation 最佳 epoch 26；P/R/mAP50/mAP50-95 = 0.9898/0.9837/0.9903/0.97734 | 50 類來源域表現，不是 BPSD 鋼琴準確率 |
| curve-v2 | 最佳 epoch 29；P/R/mAP50/mAP50-95 = 0.96812/0.94299/0.97980/0.85626 | 單一 `curve` 類來源域表現；slur/tie 最後由音樂規則判斷 |
| Dense all136 | 最佳/最後 epoch 30；P/R/mAP50/mAP50-95 = 0.97496/0.94792/0.96553/0.91764 | Complete continued training 的起點；validation 只有 110 類有實例 |
| Complete all136 | 8 epochs，最佳/最後 epoch 8；P/R/mAP50/mAP50-95 = 0.95310/0.92605/0.94375/0.90012 | DeepScores Complete validation（官方 test shards 被拿來當 validation），不是 untouched test、也不是鋼琴準確率 |

## 真實鋼琴頁已驗證到哪裡

一張 BPSD 鋼琴頁已完成完整 `pdf2musicXML.py` 流程，輸出可由 music21 重新
解析。該次 smoke 包含雙 part、piano brace、articulation、dynamic、文字方向和
slur。這證明整合流程可以執行，不是 ground-truth accuracy。

同一次測試仍出現設定為 4/4 的小節被重建成 7/8，說明目前主要瓶頸不只 YOLO，
還包括舊 OMR25 的音高、節奏、多聲部與 note object 建立。

## 人工檢查觀察

目前相對可用：

- staccato 與部分 articulation；
- 常見 dynamics；
- 獨立、輪廓清楚的曲線；
- OCR 能確認的 `cresc.`／`decresc.`／`dim.`；
- 通過楔形幾何的 hairpin。

目前仍常出問題：

- fingering 與頁碼、文字、小符號互相誤判；
- tuplet recall 與指法數字混淆；
- 穿越 staff、stem、beam 或密集和弦的 slur/tie；
- 超長、跨小節或跨 system 曲線的端點；
- grand staff、多聲部、cross-staff note/beam；
- 每小節拍數與節奏修正；
- 偵測候選存在，但舊 OMR 沒建立可掛載的 music21 note。

## Complete all136 的目前狀態

Complete source 共約 255,385 images，轉成 2,914,261 tiles。H100 NVL、
YOLOv9-E、1024、batch 12 的實測約 0.557 秒/batch，完整 train epoch 約 30
小時，full validation 另估約 3.8 小時。

因此 repository 現在提供：

- 1-epoch full pilot；
- 約 50k、class-complete 的固定 validation subset；
- 依 per-class baseline 挑弱類別 tiles，並加入 25% replay；
- 正式預設 2 epochs，而不是 30；
- epoch-boundary checkpoint；不宣稱 exact mid-epoch resume。

這些是降低實驗成本的流程。

### 正式 8-epoch run（已完成）

- 從 Dense all136 `best.pt` 接續；H100、1024、batch 8、AdamW、lr0 0.0005。
- 第 8 epoch 為最佳：P 0.95310、R 0.92605、mAP50 0.94375、mAP50-95 0.90012。
- 必須一併說明的異常與限制：
  - epoch 2～8 的 `train/box_loss` 記為 `inf`，原因尚未診斷；cls/dfl loss
    與 validation mAP 仍持續改善；
  - `results.csv` 的 validation loss 全記為 0，不可當作實際 loss；
  - 472 次 duplicate-label warning，共移除 1,038 labels；
  - `tuplet2`、`tuplet9`、`clefUnpitchedPercussion` 指標為 0；`stem`、
    `ledgerLine` recall 偏低；23 類沒有 validation instances。
- 權重不在 GitHub，需另外交付。

### BPSD 243 頁預標註

8-epoch 權重已對 243 頁 BPSD 做批次推論：confidence 0.25 得 258,383 個、
0.01 得 607,480 個合併後候選框。這批沒有人工 ground truth，**不能計算
accuracy**；0.01 版本供標註者自行選 confidence 門檻。匯出工具見
`articulation_experiments/inference/`。

### 20-epoch run

伺服器上已跑完一個 20-epoch run，但結果尚未下載與比較。在同一 validation
上比完之前，不引用它、也不假設它比 8 epoch 好。

## 報告時可以／不可以怎麼說

可以說：

> 已完成 YOLOv9 樂譜符號與曲線模型、OCR、後處理和 MusicXML 的可執行整合；
> 真實鋼琴頁可產生 MusicXML，但鋼琴專用節奏、多聲部和跨譜表結構仍需改善。

不可以說：

- 「confidence 0.95 就是 95% 正確率」；
- 「DeepScores mAP 0.97 等於鋼琴譜準確率 97%」；
- 「Complete all136 mAP50-95 0.90 就是鋼琴辨識準確率 90%」；
- 「BPSD 243 頁預標註的準確率是…」（沒有 ground truth）；
- 「所有偵測框都已正確寫入 MusicXML」。

更完整的證據與歷史見
[CHATGPT_PROJECT_CONTEXT.md](../../CHATGPT_PROJECT_CONTEXT.md)。
