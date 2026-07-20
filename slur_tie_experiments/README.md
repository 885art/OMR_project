# Slur／Tie prototype

本功能使用 OpenCV heuristic 尋找五線譜附近的細長弧形 component，再將曲線
左右端點配對到同一 staff 的 NoteGroup。

目前分類規則：

- 兩端音高集合重疊、音符相鄰且跨度合理：`tie`
- 兩端音高不同，或同音高但跨過其他音符：`slur`
- 任一端音高未知：`unknown_curve`

這是 prototype 規則，不代表相同音高的曲線一定是 tie。跨 system 曲線目前不處理。

快速測試既有 Beethoven 頁面：

```powershell
python slur_tie_experiments/preview_cached_page.py --piece beethoven1 --page 1
```

使用既有 OMR 快取更新完整曲目，不重跑 segmentation 與 articulation：

```powershell
python slur_tie_experiments/refresh_cached_piece.py --piece beethoven1 --visualize-pages 1
```

輸出包含 `slurs_ties.json`、可選 debug 圖與 MusicXML。完整 `pdf2musicXML.py`
流程只在設定的 `visualize_pages` 保存 debug 圖，其餘頁面只保留 JSON 與 XML。
