# DeepScores 全136類訓練

> 2026-08-18 教師決定：目前階段只做 DeepScores Dense → Complete all136，
> 不使用 BPSD 作訓練或 fine-tuning。伺服器的逐步命令請以
> `server/README_COMPLETE_ALL136_SERVER.md` 為準。

更新日期：2026-08-24

## 為什麼是136類，不是208類

DeepScores JSON 的 `categories` 有208個 ID，但實際組成是：

- 136個 `annotation_set=deepscores` 正式類別，ID 1–136。
- 72個 `annotation_set=muscima++` 相容類別，ID 137–208。

MUSCIMA++ 區塊有不少名稱與前136類重複；若直接訓練208類，相同符號會被迫在兩個類別間競爭。因此「DeepScores 所有 class」採用官方136類，類別順序固定為 `YOLO ID = DeepScores ID - 1`。每次準備資料都會由來源 JSON 驗證並產生同一份 mapping。

Dense train 中有115/136類實際出現，另外21類沒有 train 標註；建立136維模型仍是必要的，後續 Complete 才能補足缺少或稀少類別。

## 正確順序

1. 在本機 RTX 3090 準備並訓練 Dense all136。
2. 取得 Dense `best.pt`。
3. 把 repo、Complete 原始 shards 與 Dense `best.pt` 放到伺服器。
4. 在伺服器用逐-shard流程準備 Complete，不建立超大型合併 JSON。
5. 用 Dense `best.pt` 初始化 Complete all136，先 H100 smoke，再正式訓練。

這是 pretraining/continued training；BPS 鋼琴標註 fine-tuning 仍是之後的目標域步驟。

## 本機：Dense all136

資料準備或重驗：

```powershell
powershell.exe -ExecutionPolicy Bypass -File `
  C:\OMR_work\25-omr\server\prepare_deepscores_dense_all136_3090.ps1 `
  -Mode Full
```

輸出資料：

```text
C:\OMR_work\experiments\datasets\deepscores_dense_all136_1024
```

目前已完成並通過完整驗證：1,714張來源頁、17,281個train tiles、4,561個validation tiles、合計2,583,651個tile instances。來源中458個零寬或零高框無法成為YOLO box，已稽核後排除；有效標註的未分配數為0。

正式30 epochs已於2026-08-13完成，輸出在：

```text
C:\OMR_work\experiments\runs\yolov9_e_dense_all136_30ep_b4_3090
```

最後一輪也是最高mAP@0.5:0.95：precision 0.97496、recall 0.94792、mAP@0.5 0.96553、mAP@0.5:0.95 0.91764。這是Dense來源域且只平均validation中有實例的110類，不是BPS鋼琴準確率；`weights\best.pt`應作為Complete all136的起始權重。

先檢查，不啟動訓練：

```powershell
cmd /c C:\OMR_work\25-omr\server\train_yolov9_dense_all136_3090.bat preflight
```

正式開始：

```powershell
cmd /c C:\OMR_work\25-omr\server\train_yolov9_dense_all136_3090.bat
```

預設 YOLOv9-E、1024輸入、batch 4、30 epochs、early stopping。起始 checkpoint 是目前50類 Dense 模型；136類偵測頭會依新類別數建立，能相容的音樂影像 backbone 權重會被載入。

中斷續跑：

```powershell
cmd /c C:\OMR_work\25-omr\server\train_yolov9_dense_all136_3090.bat resume
```

## Complete：為什麼改用sharded格式

Complete 有103個 train JSON shards、26個 test shards、255,385張來源影像。若先把所有136類合成單一 JSON，會產生非常大的檔案和記憶體尖峰。

Berlioz full conversion 已完成並通過 validation：204,308 train source pages
產生 2,329,441 train tiles；51,077 test/validation source pages 產生 584,820
validation tiles。H100 batch-12 實測一個 training epoch 約 30 小時，因此
Complete continued training 改為先做 1–2 full epochs，不沿用 Dense 的 30
epochs。完整 tiling、成本、validation 與 checkpoint audit 見
`server/COMPLETE_ALL136_TRAINING_AUDIT.md`。

新流程以 shard 為安全平行化單位：

- 每個 converter subprocess 一次只載入一個來源 JSON；`--workers N` 最多同時
  執行 N 個獨立 shard，預設 1 以維持舊行為。
- 每個 shard 都有完成標記，可中斷續跑。
- partial 與完成 chunk 都記錄來源、mapping、converter 與參數 fingerprint；
  只有完全相同才可續跑，變更 recipe 後應換新輸出目錄或明確重建 chunks。
- labels/images 留在各 chunk，master `train.txt`、`val.txt` 提供給 YOLO。
- 使用 compact manifest，避免完整 annotation trace 複製成數百GB。
- 最後檢查 label 格式、類別範圍、座標、image/label配對及 train/val來源洩漏。

Windows只建議做 smoke：

```powershell
powershell.exe -ExecutionPolicy Bypass -File `
  C:\OMR_work\25-omr\server\prepare_deepscores_complete_all136_3090.ps1 `
  -Mode Smoke
```

本機實測的Complete sharded smoke使用10張train頁與10張validation頁，產生101/116個tiles；YOLOv9-E已在RTX 3090完成1 epoch訓練、validation及`best.pt`/`last.pt`寫入，證明master文字索引與分片labels能直接訓練。這個1 epoch數值不是準確率結果。

## H100／Slurm伺服器

先複製 `server/all136_env.example` 為 Git 外的 `all136_env.sh`，填入實際路徑。資料準備主要吃 CPU、RAM、NVMe；H100 不會讓 JSON/PNG 轉換本身明顯加速。

```bash
source /absolute/path/all136_env.sh
SOURCE_KIND=complete MODE=smoke bash server/prepare_deepscores_all136.sh
bash server/train_yolov9_all136.sh preflight
bash server/train_yolov9_all136.sh smoke
```

smoke 成功後建立全量資料。單工相容模式：

```bash
COMPLETE_CONVERSION_WORKERS=1 SOURCE_KIND=complete MODE=full \
  bash server/prepare_deepscores_all136.sh
```

Berlioz 建議先使用 8 個 conversion workers：

```bash
COMPLETE_CONVERSION_WORKERS=8 SOURCE_KIND=complete MODE=full \
  bash server/prepare_deepscores_all136.sh
```

這是 CPU／RAM／storage I/O 工作，不使用 H100。即使有 96 logical cores，也不
建議直接開 96 個 workers，因為大型 JSON 讀取與 PNG 寫入會競爭記憶體和磁碟；
先從 4–8 個開始。`COMPLETE_CONVERSION_WORKERS` 和 YOLO DataLoader 的
`WORKERS` 是不同變數，也不會加入 dataset-content fingerprint。既有完成 chunks
會 reuse，相同 fingerprint 的中斷 chunk 會 resume；所有 shards 成功後才由主程序
依 train/val shard ID 順序產生 master indexes 與 validation report。完成資料轉換後
正式 YOLOv9 training 才使用 H100。

目前官方 26 個 Complete test shards 會作為 YOLO validation／early stopping，
因此結果只能稱為 validation，不能稱作 untouched official test performance。
若未來要正式報 test 指標，需從 103 個 train shards 另做固定 validation split，
官方 test 只在最後評估一次；本輪 continued pretraining 暫不更動 split。

正式提交：

```bash
sbatch --export=ALL,ENV_FILE=/absolute/path/all136_env.sh,MODE=train \
  server/slurm_yolov9_all136.sbatch
```

實際 Slurm 指令中的 account、partition、H100資源名稱要依伺服器規定補上。
目前 server 預設為 2 epochs、patience 0、warmup 0.1、每 epoch snapshot/full
validation，並關閉全量 label plotting。官方 YOLOv9 只支援 completed-epoch
resume；epoch 中間中斷會從該 epoch batch 0 重跑，不可宣稱 exact mid-epoch
resume。

## 3090夠不夠

- Dense all136：夠，建議1024、batch 4。
- Complete all136：技術上可以，腳本也提供3090入口，但資料量非常大，整體時間會比Dense長很多。
- H100 80GB：更適合 Complete。3090 smoke 的 batch 4 約使用19.4GB，因此H100預設保守使用 batch 12；確認穩定後可以再嘗試16，OOM則降到8。
- GPU只影響訓練；Complete資料轉換速度主要取決於CPU、RAM與快速儲存。

H100正式訓練的起始權重應使用本機Dense all136產生的 `best.pt`，不要再用50類 checkpoint。

## 與目前OMR25的關係

all136模型是老師要求的全類別實驗，不會自動取代目前50類符號模型和專用curve模型。136類包含 notehead、stem、beam、staff、slur、tie 等；要正式接回 OMR25，仍需另外設計全類別後處理、去重、音符結構組裝與 MusicXML 規則。
