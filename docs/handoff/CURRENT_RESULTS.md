# 目前結果與限制

更新日期：2026-08-25（Asia/Taipei）

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
| Complete all136 | 尚無正式模型結果 | 2,329,441 train + 584,820 val tiles 已驗證；H100 正式 training 尚未完成 |

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

這些是降低實驗成本的流程，尚未產生新的 Complete 最佳權重。

## 報告時可以／不可以怎麼說

可以說：

> 已完成 YOLOv9 樂譜符號與曲線模型、OCR、後處理和 MusicXML 的可執行整合；
> 真實鋼琴頁可產生 MusicXML，但鋼琴專用節奏、多聲部和跨譜表結構仍需改善。

不可以說：

- 「confidence 0.95 就是 95% 正確率」；
- 「DeepScores mAP 0.97 等於鋼琴譜準確率 97%」；
- 「Complete all136 已訓練完成」；
- 「所有偵測框都已正確寫入 MusicXML」。

更完整的證據與歷史見
[CHATGPT_PROJECT_CONTEXT.md](../../CHATGPT_PROJECT_CONTEXT.md)。
