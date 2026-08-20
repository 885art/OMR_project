# 給筆電 GPT 的 Complete all136 專案入口

更新日期：2026-08-20

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
DeepScores Complete all136 全量分片轉換
                    ↓
YOLOv9-E continued training（1024，最多 30 epochs）
```

目前這個階段不使用 BPSD，不做 BPSD fine-tuning，也不把 BPSD 混進 train 或
validation。BPSD 相關程式保留在 repository 供未來使用，但不是這次訓練入口。

## GitHub 有什麼、沒有什麼

GitHub 會包含：

- 136 類 mapping 產生器；
- Complete 分片、可續跑的 YOLO 資料轉換器；
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
training，不使用 BPSD。不要從零訓練，不要把 smoke 與 full 寫進同一資料夾，
不要宣稱伺服器已可正式訓練，除非實際 Linux/GPU smoke 已成功。chunk resume
只有在來源、mapping、converter 與轉換參數 fingerprint 完全相同時才允許；
不一致時改用新輸出資料夾，除非使用者明確要求重建 chunks。

先檢查這台筆電或伺服器的 OS、GPU、Python/PyTorch/CUDA、YOLOv9、
ds2_complete、Dense best.pt、儲存空間與 scheduler，再依
server/README_COMPLETE_ALL136_SERVER.md 執行。任何路徑差異都用環境檔處理，
不要把私人路徑或憑證 commit。完成驗證後更新 CHATGPT_PROJECT_CONTEXT.md。
```

## 目前驗證邊界

- Dense all136 已在本機 RTX 3090 完成 30 epochs。
- Complete all136 的 10 train／10 validation 頁分片資料已在本機完成 1 epoch
  YOLOv9-E smoke。
- Complete 全量 255,385 張尚未轉換完成，也尚未正式訓練。
- Linux/H100/Slurm 腳本已通過本機 Bash 語法檢查，但仍要在實際伺服器完成
  smoke，才可以說伺服器流程正式可用。
- Complete 官方 test shards 目前作為 YOLO validation 及 early stopping 使用，
  所以只能稱為 validation，不能再當 untouched official test performance。
