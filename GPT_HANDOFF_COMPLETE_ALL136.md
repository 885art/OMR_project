# 給筆電 GPT 的 Complete all136 專案入口

更新日期：2026-08-23

## 先讀哪些文件

在提出命令或修改程式前，依序完整閱讀：

1. `AGENTS.md`
2. `CHATGPT_PROJECT_CONTEXT.md`
3. `DEEPSCORES_ALL136_TRAINING.md`
4. `server/README_COMPLETE_ALL136_SERVER.md`

其中 `CHATGPT_PROJECT_CONTEXT.md` 是整個專案的正式交接紀錄；如果修改資料集、
class mapping、訓練策略、伺服器命令或驗證狀態，必須同步更新它。

## 目前真正要做的事

老師目前要求的訓練流程是：

```text
已完成的 DeepScores Dense all136 best.pt
                    ↓
DeepScores Complete all136 全量分片轉換（已完成並通過 validation）
                    ↓
YOLOv9-E continued training（1024，先做 1–2 full epochs）
```

目前這個階段不使用 BPSD，不做 BPSD fine-tuning，也不把 BPSD 混進 train 或
validation。BPSD 相關程式保留在 repository 供未來使用，但不是這次訓練入口。

## GitHub 有什麼、沒有什麼

GitHub 會包含：

- 136 類 mapping 產生器；
- Complete 分片、可續跑的 YOLO 資料轉換器；
- shard-level 平行轉換，`COMPLETE_CONVERSION_WORKERS` 預設 1，Berlioz 建議
  先從 4–8 開始；
- dataset validation；
- Windows RTX 3090 與 Linux/H100/Slurm 腳本；
- smoke/full 隔離與 103 train／26 test shard 完整性檢查；
- 操作說明與專案狀態。

GitHub 不包含：

- `ds2_complete` 或其他資料集；
- 產生後的 tiles；
- `best.pt`／`last.pt`；
- experiments/runs；
- 私人的 `all136_env.sh`、帳號、token 或伺服器路徑。

因此在新筆電只 clone repository 並不能直接正式訓練。還必須另外取得：

1. `ds2_complete`：`images/`、103 個 train JSON、26 個 test JSON；
2. 官方 WongKinYiu YOLOv9 repository；
3. Dense all136 的 `best.pt`；
4. 可用的 Python/PyTorch/CUDA 環境；
5. 若在伺服器跑，伺服器的 scratch 路徑與 Slurm account/partition/GPU 名稱。

若 Berlioz 使用 Docker，`PYTHON`、YOLOv9、資料、權重與輸出全部使用 container
內的 mount 路徑；不要沿用 Windows 或 host 端路徑。

## 筆電 clone

```bash
git clone --branch feature/yolov9-migration \
  https://github.com/885art/OMR_project.git
cd OMR_project
```

若已經 clone：

```bash
git switch feature/yolov9-migration
git pull --ff-only origin feature/yolov9-migration
```

不要把資料集與權重放進 Git。可以放在 repository 外，再用環境變數指定路徑。

## 可以直接貼給新 GPT 的指令

```text
請先完整閱讀 AGENTS.md、CHATGPT_PROJECT_CONTEXT.md、
DEEPSCORES_ALL136_TRAINING.md 和 server/README_COMPLETE_ALL136_SERVER.md。

目前只做 DeepScores Dense all136 best.pt → Complete all136 continued
training，不使用 BPSD。不要從零訓練，不要把 smoke 與 full 寫進同一資料夾。
Complete full conversion 已完成且 validation passed；訓練前先閱讀
server/COMPLETE_ALL136_TRAINING_AUDIT.md。chunk resume
只有在來源、mapping、converter 與轉換參數 fingerprint 完全相同時才允許；
不一致時改用新輸出資料夾，除非使用者明確要求重建 chunks。

先檢查這台筆電或伺服器的 OS、GPU、Python/PyTorch/CUDA、YOLOv9、
ds2_complete、Dense best.pt、儲存空間與 scheduler，再依
server/README_COMPLETE_ALL136_SERVER.md 執行。任何路徑差異都用環境檔處理，
不要把私人路徑或憑證 commit。完成驗證後更新 CHATGPT_PROJECT_CONTEXT.md。

Berlioz full conversion 使用：
COMPLETE_CONVERSION_WORKERS=8 SOURCE_KIND=complete MODE=full \
  bash server/prepare_deepscores_all136.sh
workers 數量不是 dataset fingerprint；既有完成 chunks 必須 reuse，相同 fingerprint
的中斷 chunk 必須 resume，不要刪除或重建正式輸出。
```

## 目前驗證邊界

- Dense all136 已在本機 RTX 3090 完成 30 epochs。
- Complete all136 的 10 train／10 validation 頁分片資料已在本機完成 1 epoch
  YOLOv9-E smoke。
- Complete 全量 255,385 張已在 Berlioz 轉換並通過 validation：2,329,441
  train tiles、584,820 validation tiles。正式 continued training 尚未完成。
- H100 NVL、YOLOv9-E、1024、batch 12 的實測約 0.557 sec/batch，單一
  training epoch 約 30 小時。30 epochs 不再是合理預設；目前預設先跑兩個
  full epochs、patience 0、每 epoch checkpoint/full validation。
- 已加入固定 class-complete validation subset、machine-readable per-class AP、
  class-aware target tiles + replay，以及獨立 `pilot_1epoch`／`targeted` 入口。
  這些只建立 indexes，不複製或重切 Complete tiles；最終 checkpoint 仍要跑
  full validation。
- RTX 3090／5080 可做 smoke、subset validation 與小型 targeted 測試；1024
  建議分別從 batch 4／2 開始。不要用它們完整跑 233 萬 train tiles。
- 上游 YOLOv9 只支援 epoch-boundary resume。epoch 中間中斷會從該 epoch
  batch 0 重跑；不要宣稱有 exact mid-epoch resume，也不要用只存 weights 的
  假 batch checkpoint。
- 保留 `/omr/runs/yolov9_e_deepscores_complete_all136_h100_retry1` 作為 log、
  config 和 benchmark 證據，不要刪除。
- Complete 官方 test shards 目前作為 YOLO validation 及 early stopping 使用，
  所以只能稱為 validation，不能再當 untouched official test performance。
