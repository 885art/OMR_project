# Slur／Tie prototype 驗證報告

## 完成內容

- 確認 OEMER 現有輸出沒有獨立的 slur／tie channel，因此採用 OpenCV heuristic。
- 以 staff spacing 正規化尺寸，移除五線與長直線，篩選薄、平滑、有曲率的 component。
- 將曲線左右端點配對到同一 staff 的 NoteGroup；不允許跨 system 配對。
- 兩端音高相同分類為 tie，不同分類為 slur，音高未知則保留為 unknown curve。
- 每個候選皆保存 bbox、端點、方向、特徵、分類理由與 confidence。
- 只有端點完整、結構明確且 confidence 至少 0.45 的關係會寫入 MusicXML。
- Tie 同時輸出 `<tie>` 與 `<tied>` start／stop；slur 輸出有配對編號的 start／stop。

## 驗證結果

完整 `beethoven1` 26 頁由正式 `pdf2musicXML.py` 執行成功，exit code 為 0。
共輸出 26 份 slur/tie JSON 與 1 張正式 debug 圖。合併 MusicXML 可由 XML parser
正常讀取，包含 468 個 `<tie>`、468 個 `<tied>`、956 個 `<slur>`，而且三者的
start／stop 數量都完全相等；原本的 articulation 內容亦仍存在。新舊整合測試共
10 個，全部通過。

偵測器共保留 2854 個曲線候選，其中 1874 個完成左右端點配對。保守門檻後有
481 組 slur 與 259 組 tie 符合 MusicXML 寫入條件；實際 XML 關係略少，原因是
既有 rhythm tuning 可能刪除音符，或多條候選落到同一音符而 MusicXML tie 欄位只能
保留單一狀態。

## 已知限制

- beam 邊緣、hairpin、文字底線及音符輪廓仍可能成為 JSON 候選；低信心結果不寫 XML。
- 「同音高為 tie、不同音高為 slur」只是第一版規則，不能涵蓋所有樂理情況。
- chord tie 目前不冒險寫入，僅支援兩端皆為單音且音高一致的 tie。
- 跨 system slur／tie 暫不處理。
