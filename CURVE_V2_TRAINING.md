# Slur/Tie curve v2（本機 RTX 3090）

## 直接開始

資料已準備並驗證完成。關閉其他會大量使用 GPU 的程式後，雙擊：

```text
C:\OMR_work\25-omr\START_CURVE_V2_TRAIN_30EP.bat
```

預設是 YOLOv9-E、1280 模型輸入、batch 3、最多 30 epochs、early-stopping
patience 10。正式輸出位於：

```text
C:\OMR_work\experiments\runs\yolov9_e_curve_v2_fullbbox_2048_30ep_3090
```

訓練完成後應使用 `weights\best.pt`，不是固定採用最後一個 epoch。

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

## 需要重建資料時

```powershell
& 'C:\OMR_work\25-omr\server\prepare_curve_v2_3090.ps1' -Mode Full -Overwrite
```

只做啟動前檢查、不開始訓練：

```powershell
cmd /d /c "C:\OMR_work\25-omr\server\train_yolov9_curve_v2_3090.bat preflight"
```

如果仍有顯存不足，可在 PowerShell 以 batch 2 啟動：

```powershell
$env:OMR_BATCH_SIZE='2'
& 'C:\OMR_work\25-omr\START_CURVE_V2_TRAIN_30EP.bat'
```

不要用 smoke run 的權重判斷準確率；1-epoch smoke 僅驗證整個 GPU 訓練流程能執行。
