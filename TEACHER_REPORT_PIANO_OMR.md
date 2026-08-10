# 鋼琴 OMR 老師報告規劃

更新日期：2026-08-10（Asia/Taipei）

## 一句話結論

我們已將 Piano50 符號模型與 curve v2 曲線模型整合進 OMR25，並在輸出 MusicXML 前加入 OCR、幾何及音符關係檢查；目前真實鋼琴頁已能完整執行並產生雙五線譜 MusicXML，但舊 OMR25 的鋼琴節奏、多聲部與跨譜表結構仍需改善。

## 建議報告架構：8 張投影片

### 第 1 張：研究目標與原本問題

目標：將掃描的印刷鋼琴譜轉換成可編輯的 MusicXML。

建議放入原本辨識失敗的截圖：

- staccato、accent、tuplet、hairpin 漏掉。
- slur／tie 只抓到一部分或被切成多段。
- 曲線穿過五線譜時辨識變差。
- 括號內的力度或符號辨識不到。
- `cresc.` 被誤判為 `keyboardPedalUp`。

這一張只說明問題，不要先放模型分數。

### 第 2 張：完整系統架構

建議畫成下列流程：

```text
掃描鋼琴譜
    ↓
OMR25 staff／note／rhythm
    ↓
Piano50 小符號偵測
    ↓
curve v2 長曲線偵測
    ↓
OCR＋括號＋hairpin 幾何＋curve 端點檢查
    ↓
安全採用規則
    ↓
MusicXML
```

報告重點：YOLO 不是整套 OMR，只負責其中一部分視覺候選；最後是否採用還需要音樂結構判斷。

### 第 3 張：資料與訓練方式

Piano50：

- 使用 DeepScores Dense 的 50 類鋼琴相關符號。
- 加入括號合成 augmentation。
- 使用 YOLOv9-E 訓練。
- 最佳權重取自 30 epoch 執行中的 `best.pt`。

Curve v2：

- 將 DeepScores 的 slur 與 tie 合併成一個視覺類別 `curve`。
- 訓練資料只保留完整曲線 bbox。
- 使用較大的 2048 crop，減少長曲線被切斷。
- slur／tie 類型改由曲線兩端的音符關係決定。

### 第 4 張：模型指標及正確解釋

Piano50 最佳 source-domain 結果：

- 最佳 epoch：26
- mAP@0.5:0.95：0.97734

Curve v2 最佳結果（epoch 29）：

- precision：0.96812
- recall：0.94299
- mAP@0.5：0.97980
- mAP@0.5:0.95：0.85626

這張一定要說：

> 這些是 DeepScores／source-domain 指標，不是 BPSD 鋼琴譜的正式準確率。BPSD 與訓練資料的掃描品質、字型及排版不同，因此還需要有標註的 BPSD test set 才能計算真正的 target-domain accuracy。

圖片方框旁的 0.95 也只是單一 detection 的 confidence，不是「這一頁有 95% 正確」。

### 第 5 張：Curve v2 前後比較

使用固定 20 頁比較頁面：

`C:\OMR_work\experiments\piano50_curvev2_eval_20260809_20pages\index.html`

建議挑三種例子：

1. 舊模型只抓到部分曲線，新模型抓到完整曲線。
2. 曲線跨過五線譜，新模型仍能保留。
3. 同一長曲線出現重複框，說明後處理如何去重複。

綠色或不同顏色的框不代表模型直接知道 slur／tie。curve v2 只偵測 `curve`，最後才依端點音符分類。

### 第 6 張：為什麼需要後處理

可以使用 `cresc.` 的例子：

1. YOLO 將文字框判成 pedal。
2. 系統發現方框比例像一個單字。
3. 使用限制字典 OCR 辨識出 `cresc.`。
4. 移除重疊的 pedal 候選。
5. 只把 `cresc.` 文字寫進 MusicXML。

其他安全規則：

- hairpin 必須符合楔形幾何。
- curve 必須找到左右端音符。
- 同一對端點的重複 curve 只保留一個。
- tie 必須是相鄰、同音高且結構明確。
- tuplet 必須找到正確數量的音符。
- 不確定的候選只保留在 JSON 供人工檢查。

### 第 7 張：完整鋼琴頁實測

測試輸入：

`C:\OMR_work\experiments\piano50_eval_20260808_20pages\inputs\bpsd\Beethoven_Op010No3-01-08.jpeg`

模型整合輸出：

`C:\OMR_work\experiments\omr25_integrated_smoke_20260810\Beethoven_Op010No3-01-08`

完整 MusicXML：

`C:\OMR_work\25-omr\string_dataset\output\piano_smoke\piano_smoke_1.xml`

實測結果：

- 找到 12 個 staff，也就是 6 組鋼琴系統。
- 舊 OMR25 建立 336 個 note group。
- 105 個符號候選通過門檻，其中 85 個連到 note group。
- OCR 找到 3 個不同位置的 `cresc.` 候選。
- 重疊的 pedal 誤判已移除。
- 其中 2 個 `cresc.` 成功連到音符位置並寫入 MusicXML。
- 62 個 curve 通過 curve 後處理。
- 55 個 curve 找到左右兩端音符。
- 只有 7 條關係通過安全門檻並寫入 MusicXML。

輸出的 MusicXML 包含：

- 2 個 part，並以 Piano brace 組成雙五線譜。
- 450 個 note。
- 40 個 direction element。
- 18 個 dynamic。
- 45 個 staccato。
- 15 個 fingering。
- 4 個 tuplet tag。
- 6 個 time-modification tag。
- 14 個 slur endpoint tag，相當於 7 條 slur。
- 本頁沒有 tie 通過所有安全條件，因此沒有強行輸出 tie。

### 第 8 張：目前限制與下一步

完整測試雖然成功輸出 MusicXML，但舊節奏分析發現一個重要問題：設定為 4/4 的其中一小節，辨識後總拍數只有 7/8。

因此目前不能說「鋼琴轉譜已完成」，只能說模型整合已完成且可執行。

下一步優先順序：

1. 取得 BPSD 完整標註並建立 Opus-level train／validation／test split。
2. 計算 BPSD 每類 precision、recall、F1，而不是只看 DeepScores mAP。
3. 計算 curve 端點配對及 slur／tie 結構準確率。
4. 加入每小節拍數檢查及節奏修正。
5. 處理鋼琴同 staff 多聲部。
6. 處理 cross-staff note、beam、slur 及左右手關係。
7. 最後評估 MusicXML event accuracy。

## 建議口頭結論

可以直接這樣說：

> 圖片方框旁的數字只是模型對該候選的信心，不代表整頁準確率，也不代表一定會寫入 MusicXML。我們現在不再直接採用所有 YOLO 結果，而是讓文字、hairpin、tuplet、slur 與 tie 再通過 OCR、幾何或音符關係檢查。Piano50 與 curve v2 已經實際整合進 OMR25，一張真實鋼琴頁也成功輸出成雙五線譜 MusicXML。目前剩下的主要瓶頸已不只是符號偵測，而是鋼琴的節奏、多聲部與跨譜表結構。下一階段需要使用 BPSD ground truth 評估，並以最後的 MusicXML event accuracy 作為標準。

## 老師可能會問的問題

### 「方框旁邊 0.95 是 95% 準確率嗎？」

不是。那是單一候選的 detector confidence。真正準確率必須用有標註的 test set 計算 precision、recall、F1。

### 「為什麼 slur 和 tie 不直接分成兩類訓練？」

因為兩者外觀可能完全相同。tie 主要由相同音高及相鄰音符的音樂關係決定，而不是只看曲線形狀。

### 「為什麼有些已經偵測到的 curve 不放進 MusicXML？」

如果曲線缺少端點、連錯音符或結構不明確，寫入 MusicXML 會產生錯誤關係。保留為 review-only 比強行輸出安全。

### 「DeepScores 的 mAP 很高，是否代表鋼琴譜也很準？」

不能這樣推論。BPSD 是不同掃描 domain，目前 20 頁結果只能作為定性比較；要有 BPSD 標註才能正式計算準確率。

### 「現在可以完全取代人工校正嗎？」

還不行。完整測試仍出現小節拍數錯誤，而且一般鋼琴譜還有多聲部與 cross-staff 問題需要處理。

