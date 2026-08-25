# 重新訓練入口

如果只是要使用或展示目前結果，不需要看這一頁。

## 先決定你要訓練什麼

| 目的 | 應閱讀的文件 |
|---|---|
| Complete all136／H100 | [server/README_COMPLETE_ALL136_SERVER.md](../../server/README_COMPLETE_ALL136_SERVER.md) |
| Complete 成本、tiles、checkpoint 限制 | [server/COMPLETE_ALL136_TRAINING_AUDIT.md](../../server/COMPLETE_ALL136_TRAINING_AUDIT.md) |
| Dense → Complete all136 全流程 | [DEEPSCORES_ALL136_TRAINING.md](../../DEEPSCORES_ALL136_TRAINING.md) |
| BPSD 目標域 fine-tuning | [BPS_FINETUNING.md](../../BPS_FINETUNING.md) |
| curve-v2 重訓 | [CURVE_V2_TRAINING.md](../../CURVE_V2_TRAINING.md) |

## Complete all136 現況

- 103 train shards + 26 test shards 已完成轉換與 validation。
- 2,329,441 train tiles；584,820 validation tiles。
- 起始權重是 Dense all136 `best.pt`，不是 scratch。
- H100 batch 12 實測 train epoch 約 30 小時。
- 正式預設 2 epochs；可先用 `pilot_1epoch` 跑一個完整 epoch。
- 可先建立約 50k 的固定 class-complete validation subset。
- 可依 per-class baseline 建立 weak-class tiles + 25% replay。
- 最終 checkpoint 仍必須做 full validation。
- 上游 YOLOv9 只能安全地從完成的 epoch checkpoint resume，沒有 exact
  mid-epoch resume。

## 最短推薦順序

```text
Dense all136 best.pt
        ↓
固定 50k Complete validation baseline
        ↓
Complete full 1-epoch pilot
        ↓
比較每類改善幅度
        ↓
需要時做 targeted + replay
        ↓
最終 full validation
```

不要因為「好類別不用訓練」就刪掉同一 tile 上的好類別 labels；否則它們會被
當成背景。targeted 工具會保留入選 tile 的全部 labels。

## 3090／5080 與 H100 分工

- RTX 3090：1024 從 batch 4 開始，適合 smoke/subset。
- RTX 5080：16GB，1024 從 batch 2 開始；先確認 Blackwell CUDA 支援。
- H100：完整 233 萬 train tiles 與 full validation。

3090／5080 的小型 smoke 只能驗證程式、CUDA、資料與 checkpoint，不能直接
推算完整 H100 訓練時間或正式準確率。

## 研究結果命名

現行 Complete 的 26 個 official test shards 被當成 validation 使用，因此結果
只能稱為 **Complete validation**，不能稱為 untouched official test performance。

任何 DeepScores 指標都不能直接稱為真實鋼琴譜準確率。真實鋼琴效果需在凍結、
不參與調參的 BPSD work-level split 上另外評估。
