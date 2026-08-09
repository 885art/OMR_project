# Slur/Tie curve v2（本機 RTX 3090）

## 目前狀態（2026-08-09）

正式 30-epoch 訓練已完成，不需要再次雙擊啟動檔。請使用：

```text
C:\OMR_work\experiments\runs\yolov9_e_curve_v2_fullbbox_2048_30ep_3090\weights\best.pt
```

最佳結果是最後一輪 epoch 29：precision 0.96812、recall 0.94299、mAP50
0.97980、mAP50-95 0.85626。這些是 DeepScores 驗證指標，不是 BPSD
鋼琴譜準確率。

同一批 20 頁的 curve-only 與 Piano50 合併結果位於：

```text
C:\OMR_work\experiments\piano50_curvev2_eval_20260809_20pages\index.html
```

## 這次和舊模型的差異

- DeepScores 的 `slur` 與 `tie` 合併成單一視覺類別 `curve`。
- 最後的 slur/tie 類型由兩端音符關係判斷；相鄰同音高才是 tie，其餘完整合法曲線是 slur。
- 資料先裁成 2048×2048，必要時增加以曲線為中心的 crop，確保每條來源曲線至少有一個完整框。
- 切到一半的曲線不會當成完整標註，也不會被抽成純背景負樣本。
- 模型輸入縮放為 1280，兼顧長曲線上下文與單張 3090 的速度/顯存。

完整資料位置：

```text
C:\OMR_work\experiments\datasets\curve_v2_fullbbox_2048
```

驗證結果：1,714 張來源頁、4,474 個 tiles、26,379 條來源曲線、52,269 個
tile instances、0 clipped labels、0 unassigned annotations。

## 需要重新訓練時

既有正式 run 已存在，直接雙擊會安全地停止。要重跑必須換 run name：

```powershell
$env:OMR_RUN_NAME='yolov9_e_curve_v2_retry'
& 'C:\OMR_work\25-omr\START_CURVE_V2_TRAIN_30EP.bat'
```

若顯存不足，可先設定 `$env:OMR_BATCH_SIZE='2'`。不要用 smoke run 的權重
判斷準確率；1-epoch smoke 僅驗證 GPU 訓練流程能執行。
