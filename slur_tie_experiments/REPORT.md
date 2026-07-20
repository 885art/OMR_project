# Slur／Tie prototype 驗證報告

## 完成內容

- 確認 OEMER 現有輸出沒有獨立的 slur／tie channel，因此採用 OpenCV heuristic。
- 以 staff spacing 正規化尺寸，移除五線與長直線，篩選薄、平滑、有曲率的 component。
- 將曲線左右端點配對到同一 staff 的 NoteGroup；不允許跨 system 配對。
- 以端點音高、相鄰性及水平跨度共同分類 tie／slur，音高未知則保留為 unknown curve。
- 第二版要求 tie 兩端必須是相鄰音符且跨度不超過 6.5 個 staff unit；同音高長弧
  或跨過其他音符的曲線改判為 slur。
- 曲線修補只沿水平方向進行，避免把上下靠近的巢狀曲線黏成單一 component。
- 每個候選皆保存 bbox、端點、方向、特徵、分類理由與 confidence。
- 只有端點完整、結構明確且 confidence 至少 0.45 的關係會寫入 MusicXML。
- Tie 同時輸出 `<tie>` 與 `<tied>` start／stop；slur 輸出有配對編號的 start／stop。

## 驗證結果

第一版曾由正式 `pdf2musicXML.py` 完整執行成功；第二版使用既有 OMR 快取，透過
`refresh_cached_piece.py` 重新處理 `beethoven1` 26 頁並成功結束，exit code 為 0。
共輸出 26 份 slur/tie JSON 與 1 張正式 debug 圖。合併 MusicXML 可由 XML parser
正常讀取，包含 68 組 `<tie>`／`<tied>` 與 681 組 `<slur>`，而且三者的
start／stop 數量都完全相等。68 組 tie 中有 3 組是 music21 處理跨小節時值時
自動產生；曲線偵測本身有 65 組符合門檻。原本的 articulation 內容亦仍存在。
新舊整合測試共 12 個，全部通過，其中包含兩層巢狀 slur 的 MusicXML 測試。

偵測器共保留 2852 個曲線候選，其中 1889 個完成左右端點配對。保守門檻後有
687 組 slur 與 65 組 tie 符合 MusicXML 寫入條件。相較第一版，抽查頁面的 tie
分類明顯下降：第 1 頁由 53 降為 25，第 13 頁由 20 降為 7，減少同音高長弧被
誤當 tie 的情況。實際 slur XML 略少於合格關係，原因是既有 rhythm tuning 可能
刪除端點音符。

## 已知限制

- beam 邊緣、hairpin、文字底線及音符輪廓仍可能成為 JSON 候選；低信心結果不寫 XML。
- 相鄰音符、音高及跨度分類仍是 heuristic，不能涵蓋所有聲部與樂理情況。
- chord tie 目前不冒險寫入，僅支援兩端皆為單音且音高一致的 tie。
- 跨 system slur／tie 暫不處理。
