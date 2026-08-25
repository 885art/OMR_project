# 鋼琴譜 OMR 改善版

> 這是早期 Dense/Piano50 階段的操作紀錄，部分「Complete 尚未處理」描述已是
> 歷史狀態。接手請以 [`docs/handoff/README.md`](docs/handoff/README.md) 和
> `CHATGPT_PROJECT_CONTEXT.md` 為準。

給 ChatGPT／Codex 接手專案時，請先讀根目錄的
`AGENTS.md` 與 `CHATGPT_PROJECT_CONTEXT.md`。後者是需要隨專案進度持續更新的
完整現況、決策與伺服器訓練交接文件。

目前只以鋼琴譜為目標。DeepScores Complete 的全量合併與訓練沒有啟動；第一輪改用現有 Dense 資料，避免先花很久處理 Complete 才知道方向是否正確。

## 已完成

- YOLOv9 staccato 保留，沒有和 YOLO11 混合。
- 40 類擴成 50 類，新增 `tuplet1`–`tuplet9` 與 `tupletBracket`。
- 括號採兩種方式處理：訓練資料合成括號，但標籤仍只框括號內的符號；推論時另跑保守的去括號影像視圖。
- `cresc. / crescendo / decresc. / decrescendo / dim. / diminuendo` 用受限字典 OCR 複核。未被 OCR 確認的低信心 pedal 不會寫入 MusicXML。
- hairpin 以低 YOLO 門檻取回候選，再用楔形線段幾何驗證；五線譜直線候選會排除。
- slur/tie 的相鄰碎框會合併，明顯直線與太小的框會排除。即使偵測分數高，沒有兩個音符端點時也只會標為 `review`，不會直接寫入 MusicXML。
- tuplet 只有在數字、同一 staff 的音群數量與信心都足夠時才寫入 MusicXML。比例不明的 1、2、4、8 保留人工複核，不猜節奏比例。
- 預覽圖顯示 `accept / review / reject`。框旁的小數是模型/後處理信心，不是「這個一定正確的機率」。

## 已驗證輸出

- 六張 Beethoven 鋼琴頁整合結果：`C:\OMR_work\experiments\piano_validation\combined_review_v2`
- DeepScores 類別瀏覽器：`C:\OMR_work\experiments\dataset_browser\complete_relevant\index.html`
- 50 類小型資料驗證：`C:\OMR_work\experiments\datasets\piano50_dense_parentheses_smoke_validated`

## 準備第一輪 Dense 50 類資料

在 PowerShell 執行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\OMR_work\25-omr\server\prepare_piano50_dense_3090.ps1"
```

完成後資料位於：

`C:\OMR_work\experiments\datasets\piano50_dense_parentheses`

## 開始 RTX 3090 訓練

預設為 YOLOv9-E、1024、batch 4、20 epochs。資料準備完成後執行：

```powershell
& "C:\OMR_work\25-omr\server\train_yolov9_piano50_3090.bat"
```

可在同一個 PowerShell 指定不同 epoch 或 batch：

```powershell
$env:OMR_EPOCHS='20'
$env:OMR_BATCH_SIZE='4'
$env:OMR_RUN_NAME='yolov9_e_piano50_dense_parentheses_20ep_b4'
& "C:\OMR_work\25-omr\server\train_yolov9_piano50_3090.bat"
```

模型結果會放在：

`C:\OMR_work\experiments\runs\<OMR_RUN_NAME>\weights\best.pt`

訓練成功後腳本會自動把 `jsonTemplate.json` 切換到該次的 `best.pt` 與 50 類 mapping；不需手動改模型路徑。

## MusicXML 採用規則

- `accept` 仍需通過音符/staff 關聯才可輸出。
- `review` 會留在 JSON 與預覽圖，不自動寫入 MusicXML。
- `reject` 不寫入 MusicXML。
- slur/tie 必須解析到完整起始音與結束音；tie 還必須符合相鄰同音高條件。
- hairpin 必須同時有 YOLO 候選與楔形幾何，並解析跨度。
- tuplet 必須有可解釋的節奏比例與完整音群跨度。
