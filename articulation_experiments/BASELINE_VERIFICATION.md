# Articulation baseline 驗證紀錄

## 可運作範圍

目前 pipeline 支援：

```text
PDF／圖片 → YOLO articulation detection → NoteGroup association
→ MusicXML staccato／accent／tenuto output
```

模型使用 `outputs/runs/baseline_v1/weights/best.pt`，主流程整合於
`pdf2musicXML.py`，association 與 MusicXML helper 位於
`omr/articulation.py`。

## 驗證結果

- Beethoven 26 個逐頁 MusicXML：全部可由 XML parser 開啟
- 合併 MusicXML：可解析
- 合併結果：staccato 1,358、accent 4、tenuto 2
- 26 份 page-level association JSON：共 1,369 個 matched relations
- Runtime integration tests：3／3 通過
- 保留 5 張代表性 debug 圖：第 1、7、13、14、15 頁

## 執行

完整流程：

```powershell
python pdf2musicXML.py
```

快速驗證既有快取頁面：

```powershell
python articulation_experiments/inference/preview_cached_page.py `
  --piece beethoven1 --page 1 --device 0
```

測試：

```powershell
python -m unittest articulation_experiments.tests.test_runtime_integration -v
```

## 已知問題

- 真實掃描譜和 DeepScoresV2 合成資料存在 domain gap。
- Accent 容易和 hairpin 混淆。
- Tenuto 訓練樣本較少，真實掃描譜 recall 偏低。
- 目前 Beethoven 測試沒有人工 ground truth，因此上述數量不是正式準確率。
