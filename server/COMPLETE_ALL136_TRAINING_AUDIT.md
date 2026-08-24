# DeepScores Complete all136 訓練成本與資料切片 audit

更新基準：2026-08-24，branch `feature/yolov9-migration`。

本文件記錄 Complete full conversion 完成後的正式訓練決策。它不代表重新切資料，
也不改變目前 136-class dataset recipe。

## 已知資料與 H100 benchmark

- Complete source：204,308 train pages、51,077 test/validation pages，共 255,385。
- 寫出的 tiles：2,329,441 train、584,820 validation，共 2,914,261。
- 平均：train 11.402 tiles/page、validation 11.450 tiles/page、全體
  11.411 tiles/page。
- model：YOLOv9-E；`imgsz=1024`；`batch=12`。
- Dense all136 `best.pt` 初始化；不是 scratch training。
- H100 NVL 實測：約 80.5 GB、2056 batches / 19:05，約
  0.557 sec/training batch。
- 一個 train epoch 有 `ceil(2,329,441 / 12) = 194,121` batches，單算
  training 約 30.0 小時。
- 官方 YOLOv9 validation 使用兩倍 batch（目前為 24），完整 validation 有
  `ceil(584,820 / 24) = 24,368` batches。若每 batch 與 training 同速，約
  3.77 小時；實際值必須在 H100 量測。
- 30 epochs 單算 training 約 901 小時／37.5 天，尚未含 validation、dataset
  scan 與排隊時間，因此不再視為合理預設。

請保留伺服器上的
`/omr/runs/yolov9_e_deepscores_complete_all136_h100_retry1`。它雖然沒有可用的
epoch checkpoint，但 log、設定和 benchmark 仍是比較基準。

## 為何一頁通常有 12 個候選 tiles

正式 recipe 是：tile 1024、overlap 256，所以 stride 是 768；edge policy 是
`shift`。`tile_utils.py::tile_starts()` 先保留 stride grid，再把最後一張 tile
貼齊右／下邊界。

常見 1960×2772 page 因而得到：

```text
x = 0, 768, 936
y = 0, 768, 1536, 1748
3 × 4 = 12 candidate tiles
```

最後兩個 x tiles 的 overlap 是 856 px（83.6%）；最後兩個 y tiles 是
812 px（79.3%）。常見 page 若 12 張都寫出，tile area／source area 為 2.316，
即一個 source pixel 平均被看約 2.32 次。小 bbox 在一般 overlap 區可出現在
2–4 tiles；在 `shift` 造成的三重覆蓋窄帶，理論最高可出現在 9 tiles。

這不是無限重複或錯誤座標，但確實是 `shift` 的可觀 redundancy。另一方面，
它確保所有輸出都是完整 1024×1024 source crop，不會像 `pad` 在右／下緣放入
大量白色 padding。

converter 對 bbox 的規則是：bbox center 在 tile 內，**或**保留面積比例至少
0.6 就收下。center 在 tile 內時，即使 bbox 被切掉超過 40% 仍會收下；正式
Complete wrapper 沒有開 `--require-full-bbox` 或 target-centered crop。因此降低
overlap 前必須特別測 slur/tie、hairpin、beam 等長符號的 boundary recall。

negative sampling 是每個 shard 在所有 positive tiles 決定後，選
`floor(positive_tile_count × 0.05)`，若可用 negatives 不足則再降低。因此寫出
資料中的 negative fraction 上限約 4.762%；2,914,261 tiles 中約最多 138.8k
是 negatives，至少約 2.775M 是 positives。精確數字請用下一節工具讀取各
chunk 的既有 `statistics.json`，不必重切 179 GB dataset。

## 在 Berlioz 執行精確 audit

一般統計只讀 129 個小型 chunk statistics，速度快：

```bash
cd /path/to/OMR_project
source /path/to/all136_env.sh

python articulation_experiments/dataset/audit_complete_all136.py \
  --dataset-root "$ALL136_DATASET" \
  --batch-size 12 \
  --seconds-per-batch 0.557 \
  --epochs 30 \
  --output-json "$WORK_ROOT/audits/complete_all136_audit.json"
```

它會回報：

- candidate／positive／selected negative tiles；
- 每 source page 的 tiles；
- `tile_instance_count / source_target_instance_count`，也就是同一 source
  annotation 平均進入幾個 tile labels；
- 被分配到多張 tile 的 annotation 比例與額外 assignment 數；
- clipped bbox assignments；
- 依 benchmark 推算的 train／validation 成本。

YOLO 的 `duplicate labels removed` 是「同一個 `.txt` 裡出現完全相同的
class+box row」，不是同一 annotation 出現在不同 tiles。若要精確掃描 291 萬個
label files，可額外執行（很慢）：

```bash
python articulation_experiments/dataset/audit_complete_all136.py \
  --dataset-root "$ALL136_DATASET" \
  --scan-label-duplicates \
  --output-json "$WORK_ROOT/audits/complete_all136_with_label_duplicates.json"
```

目前 converter 會逐一處理 source `ann_ids`，但不會對輸出的 `label_lines` 做
`unique`。DeepScores source 本身確實存在不同 annotation IDs、相同 DeepScores
class 與相同 bbox 的記錄；這會產生 YOLO warning。Overlap 只會把 annotation
放到不同 label files，不會觸發該 warning。正式決定是否清除前，仍應用上述
exact scan 量化，不能把所有 warning 一概當成 tiling bug。

## 訓練方案排序

### 1. 推薦的第一個正式結果：目前 dataset，不重切，1–2 full epochs

研究意義最接近原定 Complete all136 continued training，且不用再付 conversion
成本。repository 預設先改成 2 epochs、每個 epoch 做 full validation、
`patience=0`、每 epoch 留 snapshot。

原本 hyperparameter 的 `warmup_epochs=2.0` 在本資料等於約 388k batches／60
小時，對 1–2 epoch continued training 不合理。Complete 專用 hyp 只把 warmup
改為 0.1 epoch（約 19.4k batches／3 小時），其他 detection 和 augmentation
recipe 保持一致。

若只需要 final 指標，可設：

```bash
export VALIDATION_POLICY=full_final_only
export PATIENCE=0
```

這會使用官方 `--noval`，只在最後 epoch 做 full validation。它不能用 intermediate
metric 選 best epoch，所以正式比較優先保留 `full_each_epoch`。

### 2. 固定 representative validation subset

可節省每 epoch validation，但 subset 必須：

- 按 source page 選，不可把同頁重疊 tiles 當獨立樣本亂抽；
- deterministic；
- 覆蓋全部 136 classes，並特別保留 rare classes；
- 只作 monitoring／checkpoint selection；正式結果仍跑一次 full validation。

目前 repository 已提供 `build_complete_all136_indexes.py validation`：按 source
page deterministic 抽樣、強制每類至少涵蓋指定頁數，預設約 50k tiles。它只寫
index/YAML/statistics，不複製 images/labels。`validate_complete_all136.sh` 會輸出
machine-readable `per_class_metrics.json`。此 subset 只供快速監控，最終模型仍需
跑 full 584,820-tile validation。

### 3. 針對表現差類別做 class-aware fine-tuning

先用 Dense best.pt 在固定 Complete validation 做 per-class baseline。之後可提高
「含差類別 tiles」的抽樣率，但入選 tile 上的**全部 136 類 labels 必須保留**，
並混入普通／表現好類別 replay tiles。不可把好類別 labels 刪掉，否則 YOLO
會把真實符號當 background，造成 catastrophic forgetting。

若老師要求所有 classes 都經過 Complete training，先做一個 full all136 epoch，
再做 targeted stage，是最清楚的兩階段實驗。

目前 `build_complete_all136_indexes.py targeted` 會由 baseline mAP 選弱類別，
挑出含這些類別的完整 tiles，並混入 non-target replay。它不刪任何 tile 內的好
類別 labels，也不重寫 179 GB dataset，因此可安全做成獨立實驗。預設門檻與
tile 上限是起點，必須在看到 baseline class distribution 後再確認。

### 4. Batch size

不改 dataset 或解析度，是低風險的 throughput 調整。batch 12 已約 80.5 GB，
batch 16 可能 OOM；只能用固定 500–2000 batches 比較 images/sec，不能只看
seconds/batch。官方 trainer 用 nominal batch 64 做 gradient accumulation，所以
batch 12 與 16 的 optimizer dynamics 不會完全等於 raw batch size。

### 5. 降 overlap／重新切 dataset

這會改變 training distribution，應是獨立 ablation，不是目前第一步。尤其只把
overlap 改成 128、但保留現行 `shift` 時，1960 px 寬度會得到
`x=0,896,936`，最後兩張反而重疊 984 px；不能只改一個數字就宣稱消除重複。
若將來重切，應一起設計 adaptive starts，並對 tiny articulation 與跨 tile 長符號
做 boundary recall 測試。

### 6. 降 imgsz

不推薦。它會把 1024 tile 再縮小，直接傷害 staccato、accent、tenuto、fingering
等 tiny symbols。降低 overlap 不會降低符號 pixel resolution；降低 imgsz 會。

## Checkpoint、pause、resume 的真實限制

目前官方 YOLOv9 可以在**完成一個 epoch 後**保存 model、EMA、optimizer、epoch
與 options。scheduler 依 epoch 重建。它沒有保存：

- current batch／sampler position；
- AMP GradScaler；
- Python、NumPy、Torch、CUDA RNG 全部狀態；
- DataLoader shuffle generator；
- 16 個 worker RNG；
- 已 prefetch、但尚未訓練的 batches。

而且 trainer 使用 persistent `InfiniteDataLoader`。因此目前的 `resume` 是
epoch-boundary resume；若在 epoch 中間停止，下次會從該 epoch 開頭重新跑。
repository 不提供假的 `--checkpoint-every-N-batches`。若要 bitwise/exact
mid-epoch resume，必須 fork trainer、實作 stateful sampler，處理 worker prefetch
與 per-sample deterministic augmentation，再做「中斷前後 batch sequence、loss、
weights 一致」的 integration test；這不是安全的小改。

正式 run 目前會保存：

```text
runs/<RUN_NAME>/
  weights/last.pt
  weights/best.pt
  weights/epoch*.pt
  results.csv
  opt.yaml
  hyp.yaml

runs/_console_logs/<RUN_NAME>.log
```

`DISABLE_PLOTS=1` 會傳入官方 `--noplots`，避免每次對全量 labels 畫圖。
`train` 會拒絕覆蓋既有 run；`resume` 只使用同一正式 run 的 `last.pt`，不建立
`retry1`、`retry2`。

## 目前推薦正式 plan

1. 保留既有 2.914M-tile dataset，先不重切。
2. 先執行 audit，取得 exact positive/negative 與 annotation duplication factor。
3. Dense best.pt 先跑固定 validation baseline，取得 per-class 表現。
4. 第一階段可先用 `pilot_1epoch` 跑 1 個 full epoch；正式入口仍預設 2、
   warmup 0.1、patience 0、imgsz 1024。已正常完成的 1-epoch run 若再以其
   weights 開新 stage，optimizer/scheduler 會重建，不等同 exact resume；若確定
   要同一 optimizer 連跑兩輪，應直接使用正式 2-epoch入口。
5. 預設每 epoch full validation；若伺服器實測 validation 太慢，再改 final-only
   或建立 class-complete deterministic subset。
6. 每個完成 epoch 保存 `last/best/epochN`；Slurm wall time 必須容納至少一整個
   train epoch 加 validation。中途 batch exact resume 目前不支援。
7. 依 per-class 結果再決定是否做 targeted class-aware + replay stage。
8. 最終模型跑一次完整 584,820-tile validation，報告時稱為 Complete validation；
   現行 26 個 official test shards 已被當 validation 使用，不是 untouched test。

成本估計：1 epoch + full validation 約 34 小時；2 epochs + 每輪 full validation
約 68 小時；3 epochs 約 101 小時。實際以獨占 H100 的 validation benchmark 修正。
