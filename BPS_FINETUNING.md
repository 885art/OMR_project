# BPS 鋼琴譜訓練與微調

更新日期：2026-08-11

這套流程已把 BPS-OMRv01 標註轉成 YOLOv9 可用資料，並準備兩個獨立模型：

1. `symbols`：50 類 Piano50 符號模型，包含力度字母、staccato、accent、fermata、踏板、指法、hairpin、tuplet 等。
2. `curves`：一類 `curve` 模型，只偵測 slur/tie 的完整曲線；slur 或 tie 的語意要在 OMR25 依端點音符關係判斷。

`cresc.`、`decresc.` 等方向文字仍由受限制 OCR 處理，不把一般文字硬塞進 YOLO 類別。

## 已準備好的本機資料

來源標註：`C:\OMR_work\BPS-OMRv01`

- 243 頁、32 首作品、41,874 個原始框。
- 固定按作品切分，沒有同一首作品跨 train/val/test：train 170 頁、val 36 頁、test 37 頁。
- test 六首為 Op.54、78、79、90、101、109。它們已鎖定，不用來選 epoch、threshold 或改規則。
- 找到一個完全在頁面外的 tie 框；轉換器拒絕該框並留下統計紀錄，沒有改寫原始標註。

正式資料集：

| 用途 | 路徑 | train / val / test tiles |
|---|---|---:|
| BPS 符號 | `C:\OMR_work\experiments\datasets\bps_piano50_worksplit_v1` | 5,057 / 1,075 / 1,101 |
| BPS 曲線 | `C:\OMR_work\experiments\datasets\bps_curve_v2_worksplit_v1` | 340 / 72 / 74 |
| BPS 符號 + Dense replay | `C:\OMR_work\experiments\datasets\bps_piano50_dense_replay_v1` | 10,114 / 1,075 / 1,101 |
| BPS 曲線 + curve-v2 replay | `C:\OMR_work\experiments\datasets\bps_curve_v2_dense_replay_v1` | 680 / 72 / 74 |

replay 只加入 train，比例為 BPS train 1:1；val/test 都維持純 BPS。這樣能學鋼琴掃描域，又降低忘掉 DeepScores 已學符號的風險。

## Windows 3090：重建或驗證資料

下面命令可重複執行；資料已存在時只重新驗證，不會開始訓練：

```powershell
powershell.exe -ExecutionPolicy Bypass -File `
  C:\OMR_work\25-omr\server\prepare_bps_finetune_3090.ps1 -Mode Full
```

先做訓練前檢查：

```powershell
cmd /c C:\OMR_work\25-omr\server\train_yolov9_bps_piano50_finetune_3090.bat preflight
cmd /c C:\OMR_work\25-omr\server\train_yolov9_bps_curve_finetune_3090.bat preflight
```

確認後才開始正式訓練；單張 3090 建議依序跑：

```powershell
cmd /c C:\OMR_work\25-omr\server\train_yolov9_bps_piano50_finetune_3090.bat
cmd /c C:\OMR_work\25-omr\server\train_yolov9_bps_curve_finetune_3090.bat
```

預設都是 30 epochs，使用 early stopping；符號 batch 4、曲線 batch 3。初始權重分別是目前 Dense Piano50 與 curve-v2 的 `best.pt`。若同名 run 已存在，腳本會停止，避免覆寫。

中斷後從同名 run 的 `weights\last.pt` 續跑：

```powershell
cmd /c C:\OMR_work\25-omr\server\train_yolov9_bps_piano50_finetune_3090.bat resume
cmd /c C:\OMR_work\25-omr\server\train_yolov9_bps_curve_finetune_3090.bat resume
```

若 checkpoint 不在預設位置，先設定 `OMR_RESUME_CHECKPOINT` 為完整路徑。

只用 validation 選出最後配方後，才能解鎖一次最終 test：

```powershell
powershell.exe -ExecutionPolicy Bypass -File `
  C:\OMR_work\25-omr\server\evaluate_bps_test_3090.ps1 `
  -Task Symbols `
  -Weights C:\OMR_work\experiments\runs\你的run\weights\best.pt `
  -AllowFinalTest
```

曲線把 `Symbols` 改成 `Curves`。

## DeepScores Complete 對照實驗

Complete 是額外的 generic pretraining 對照，不應取代 BPS 微調。50 類 smoke 已通過：560 張來源頁、8,395 tiles、23,596 tile instances、0 個驗證錯誤，RTX 3090 preflight 也通過。

完整轉換會很久且占大量空間，所以目前沒有自動啟動。需要時執行：

```powershell
powershell.exe -ExecutionPolicy Bypass -File `
  C:\OMR_work\25-omr\server\prepare_complete_piano50_3090.ps1 -Mode Full
cmd /c C:\OMR_work\25-omr\server\train_yolov9_complete_piano50_3090.bat preflight
cmd /c C:\OMR_work\25-omr\server\train_yolov9_complete_piano50_3090.bat
```

比較順序建議為：目前 Dense checkpoint → BPS fine-tune，對照 Complete checkpoint → 同一 BPS fine-tune；最後用同一 BPS validation 指標選配方。

## Linux／Slurm 伺服器

1. 複製 `server/bps_env.example` 為 Git 外的 `bps_env.sh` 並填絕對路徑。
2. 用 `server/prepare_bps_finetune.sh` 建立及驗證資料。
3. 先跑 `preflight`，再跑一個 `smoke`。
4. smoke 成功後才提交正式 job。

```bash
source /absolute/path/bps_env.sh
MODE=full bash server/prepare_bps_finetune.sh
bash server/train_yolov9_bps_finetune.sh symbols preflight
bash server/train_yolov9_bps_finetune.sh curves preflight
bash server/train_yolov9_bps_finetune.sh symbols smoke

sbatch --export=ALL,ENV_FILE=/absolute/path/bps_env.sh,TASK=symbols,MODE=train \
  server/slurm_yolov9_bps.sbatch
```

兩張 3090 的第一輪最有效分配是每張卡各跑一個獨立實驗，例如 GPU 0 跑 symbols、GPU 1 跑 curves 或另一資料配方。兩張卡不會合成 48 GB VRAM。DDP 要等單卡 smoke 成功後再測，且 batch size 必須能被 GPU 數整除。

目前 Linux／Slurm 檔案已準備，但尚未在實際伺服器完成 smoke；在那之前不能宣稱伺服器訓練已驗證完成。

## 主要實作檔案

- `articulation_experiments/dataset/convert_bps_yolo_finetune.py`：BPS 轉 symbols/curves tiles。
- `articulation_experiments/dataset/compose_finetune_replay.py`：只對 train 加 replay。
- `articulation_experiments/dataset/validate_finetune_yolo_dataset.py`：檢查 labels、類別、座標、配對與作品切分洩漏。
- `articulation_experiments/dataset/bps_work_split_v1.json`：凍結作品切分。
- `articulation_experiments/dataset/bps_to_piano50_aliases.json`：明確且不含糊的 BPS 類別別名。
- `server/train_yolov9_bps_finetune.sh`：Linux 共用訓練／驗證入口。
