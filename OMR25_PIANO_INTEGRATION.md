# OMR25 鋼琴譜整合技術說明

更新日期：2026-08-10（Asia/Taipei）

## 一、目前啟用的模型

只要下列檔案存在，OMR25 會優先使用目前訓練完成的模型：

- Piano50 符號模型：
  `C:\OMR_work\experiments\runs\yolov9_e_piano50_dense_parentheses_30ep_b4_3090\weights\best.pt`
- curve v2 曲線模型：
  `C:\OMR_work\experiments\runs\yolov9_e_curve_v2_fullbbox_2048_30ep_3090\weights\best.pt`

設定檔用途：

- `jsonTemplate.json`：四重奏／四聲部設定範例。
- `pianoConfigTemplate.json`：雙五線譜鋼琴設定範例。

若要移到伺服器執行，必須先複製兩個模型權重及相對應的 dataset，
再修改樂曲設定檔中的 `weights`、`data_yaml`、`mapping` 與
`yolov9_root` 路徑。

模型權重、dataset、訓練輸出及大量推論結果都不應提交到 Git。

## 二、目前的辨識與 MusicXML 決策流程

### 1. 一般音樂符號

Piano50 使用 0.05 的低原始門檻進行偵測，先保留可能的小符號，再由各類別自己的門檻篩選。

文字方向 OCR 會保留擴張框作為辨識稽核資料，但輸出的 `direction_text`
改用實際墨跡緊框，避免 `cresc.`／`decresc.` 的過大搜尋框影響水平位置。
`filter_outside_music_region` 預設開啟，譜外頁碼、標題與出版資訊候選會移至
`outside_music_region_candidates`，不會直接進入 MusicXML；原始 `dynamicS`
候選仍保存在 `dynamic_letter_detections`，批次檢查圖以紫色 `raw_s` 顯示。

例如：

- `staccato`：0.45
- `accent`：0.55
- `tenuto`：0.45
- `pedal_start`／`pedal_stop`：0.78

畫面方框旁顯示的數字是模型對該方框的 confidence，並不是整頁準確率，也不代表一定會寫進 MusicXML。

### 2. 括號處理

系統會同時使用原圖與移除可信括號線條後的影像進行辨識。

目的是讓 `(sf)`、括號內力度、括號內指法或其他被括號干擾的符號仍有機會被模型找到。原圖結果仍會保留，因此不會只依賴括號清除後的影像。

### 3. `cresc.`、`decresc.` 與 pedal 誤判

寬度像文字，或 confidence 不夠高的 pedal 候選，必須再通過限制字典的 OCR。

OCR 字典目前包含：

- `cresc.`／`crescendo`
- `decresc.`／`decrescendo`
- `dim.`／`diminuendo`
- `Ped.`

處理規則如下：

1. 如果 OCR 辨認為 `cresc.`、`decresc.` 或 `dim.`，候選會改成文字方向記號。
2. 原本重疊的 pedal 候選會被移除。
3. 如果方框很像文字，但 OCR 無法確認，它會留在 audit JSON 中，不會寫成 pedal MusicXML。
4. 只有幾何、confidence 或 OCR 足以確認的 pedal 才會被採用。

### 4. Hairpin

YOLO 偵測到的 crescendo／diminuendo hairpin 必須通過楔形線條檢查。

系統也加入 OpenCV 楔形偵測，補足 YOLO 完全漏掉 hairpin 的情況。兩個來源重疊時會去除重複候選。

2026-08-10 的三首鋼琴譜檢查又加入兩項通用規則：切圖重疊造成的 hairpin
片段會先合併成完整框，再重新依楔形判斷 crescendo／diminuendo；已確認
hairpin 的左右鄰域也會做一次窄範圍幾何搜尋，用來補回緊接在旁邊、但 YOLO
漏掉的反向 hairpin。這兩項規則不是針對單一頁面的固定座標。

通過檢查後，hairpin 的左右端還必須連到兩個不同的音符群組，才會建立 MusicXML crescendo／diminuendo spanner。

### 5. Slur 與 tie

curve v2 只偵測一個視覺類別：`curve`。

原因是 slur 和 tie 的外觀可能相同，單靠圖片分類不可靠。現在的流程是：

1. 偵測完整曲線方框。
2. 偵測階段仍會清除明顯的模型重複框，但進入端點配對後不再因端點相同或 confidence 較低而刪除候選。
3. 曲線穿過五線譜線時不再直接刪除。
4. 與已確認 hairpin 重疊的 curve 會被排除。
5. 尋找曲線左端與右端對應的音符。
6. 相鄰、同音高、單音的關係才可能判定為 tie。
7. 其他曲線會保留為 slur；若原本看似 tie、但和弦內無法確定對應音高，也先以 slur 寫出，避免建立錯誤或不完整的 tie。

目前預設使用 `"acceptance_policy": "all_detected"`。這是依人工檢查結果採用的「全部保留」模式：

- articulation 不再因距離、側向位置、跨 staff 或同一音符已有相同類別而變成灰色 review 項目，而是配到代價最低的音符群組。
- curve 不再因 confidence、端點距離或重複端點關係而被排除；一般端點搜尋失敗時，會改用最接近的有序音符對。
- tuplet 不再因 confidence 偏低而排除，並會在局部範圍不足時使用同 staff 上最接近的音符範圍。

如需回到原本較嚴格的模式，可在 articulation 與 slur/tie 設定中改為 `"acceptance_policy": "conservative"`。

注意：「候選被保留」不等於「最終 MusicXML 一定有對應標籤」。若舊 OMR25 的音高／節奏解析沒有為端點建立 music21 音符物件，候選仍會完整留在 JSON 和檢查圖，但 MusicXML 沒有可掛載的音符。這種情況需要修正鋼琴音符解析，而不是再降低 YOLO 門檻。

### 6. Tuplet

Tuplet 數字會嘗試尋找同一個 staff 上對應的音符範圍。

只有在下列條件成立時才寫入 MusicXML：

- 找到有效的 tuplet 數字。
- 能找到預期數量的連續音符群組。
- 節奏比例可合理推導。
- confidence 高於 `tuplet_xml_confidence`。

不明確的 tuplet 仍會顯示在候選 JSON 中，但不會強行產生錯誤的 time-modification。

外觀相同的數字 `3` 可能是指法，也可能是三連音。現在只有在同一 staff、
同一高度出現至少三個、且各組間距規律並符合 beamed triplet 群組尺度時，才會把
模型的 `fingering_3` 重新分類為 `tuplet_3`；零散的數字仍保留為指法。這修正了
月光奏鳴曲伴奏中「每三個音一個 3」被全部當成指法的情況。

力度字母的後處理也改為按水平閱讀順序組合，並檢查候選左右是否仍有同一單字的
字母墨跡。因此 `sempre` 內誤抓出的 `mp`、`decresc.`／`senza`／`sordino`
內的孤立 `s` 不會成為最終力度事件；原始偵測仍留在 audit JSON 供檢查。

## 三、本機測試方式

### 自動化測試

```powershell
& 'C:\Users\minemine\miniconda3\envs\omr\python.exe' -m unittest discover `
  -s 'C:\OMR_work\25-omr\articulation_experiments\tests' -p 'test_*.py'

& 'C:\Users\minemine\miniconda3\envs\omr\python.exe' -m unittest discover `
  -s 'C:\OMR_work\25-omr\slur_tie_experiments\tests' -p 'test_*.py'
```

目前結果：

- articulation／piano：38 個測試通過。
- slur／tie：14 個測試通過。

### 單張鋼琴譜模型整合測試

```powershell
& 'C:\Users\minemine\miniconda3\envs\omr\python.exe' `
  'C:\OMR_work\25-omr\server\smoke_integrated_piano_page.py' `
  --image 'C:\path\to\piano-page.jpeg' `
  --output-dir 'C:\OMR_work\experiments\omr25_integrated_smoke' `
  --device 0
```

這個測試會實際載入 Piano50 和 curve v2，執行 OCR、hairpin、curve 幾何處理並輸出檢查圖，但不會測試完整的舊節奏分析。

### 批次產生鋼琴符號框與 JSON

需要同時檢查多張鋼琴譜、但暫時不跑完整 MusicXML 時，可使用批次工具。模型只載入一次，輸出會包含每頁原圖、符號框、curve 框、兩種 JSON 與 HTML 總覽：

```powershell
& 'C:\Users\minemine\miniconda3\envs\omr\python.exe' `
  'C:\OMR_work\25-omr\server\batch_integrated_piano_pages.py' `
  --input-dir 'C:\path\to\piano_pages' `
  --output-dir 'C:\OMR_work\experiments\piano_symbol_review' `
  --glob '*.jpeg' --limit 10 --device 0
```

這個模式沒有執行舊音符／節奏解析，所以 JSON 不包含 NoteGroup 端點配對，也不能直接當成 MusicXML 接受結果或準確率。

### 完整 OMR25 主程式

OMR25 仍使用：

```powershell
& 'C:\Users\minemine\miniconda3\envs\omr\python.exe' `
  'C:\OMR_work\25-omr\pdf2musicXML.py'
```

實際處理的樂曲由 `string_dataset/piecesToRun.json` 決定。

鋼琴設定必須包含：

```json
{
  "score_mode": "piano",
  "numTrack": 2,
  "track_shift": [0, 0],
  "clef_options": [[1], [-1]]
}
```

輸出的兩個 part 會以 Piano brace 組成一組雙五線譜。

## 四、目前鋼琴主程式的能力邊界

一張 BPSD 鋼琴頁已成功完成整套 `pdf2musicXML.py`，並輸出雙 part、帶 brace 的 MusicXML。這表示模型與主程式已能實際串接執行。

套用全部保留模式後，同一頁的 105 個 articulation 候選全部完成音符群組配對；68 個 curve 候選全部成為可輸出的關係。實際 MusicXML 寫入 51 個 staccato、17 個 fingering、27 個 dynamic、3 個文字方向，以及 60 條 slur。其餘候選仍保留在 JSON 和標示圖；未寫入 MusicXML 的主因是對應 note group 沒有被舊音符解析轉成可掛載的 music21 音符。

但這不代表鋼琴轉譜已完全正確。該次測試中，舊節奏分析把設定為 4/4 的其中一小節算成 7/8。

目前鋼琴核心仍需要處理：

- 同一 staff 的多聲部拆分。
- 左右手及 grand staff 關係。
- cross-staff note／beam／slur。
- 每小節總拍數驗證及自動修正。
- 延音線跨小節、跨系統的連接。
- BPSD 標註資料上的正式 precision、recall、F1 與 MusicXML event accuracy。

目前正確的說法是：

> Piano50、curve v2、OCR、後處理及 MusicXML 安全規則已整合進 OMR25，並可在真實鋼琴頁完整執行；鋼琴專用節奏與多聲部結構仍需要後續改善。
