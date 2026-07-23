# 40 類 YOLO 樂譜符號操作說明

## 這個版本新增什麼

原本 17 類模型保留不變，新增 23 個原始 YOLO 類別，共 40 類：

- 力度字母：`p`、`m`、`f`、`s`、`z`、`r`，推論後會組成 `pp`、`mf`、`sfz` 等記號。
- 漸強與漸弱 hairpin。
- 上弓、下弓、琶音、踏板開始與踏板結束。
- 指法 0–5。
- 1–4 條顫音線。

Tuplet、八度線、slur 與 tie 不放進這個模型。它們需要節奏或跨音符關係；slur/tie 已由獨立模組處理。

## 開始訓練

雙擊專案根目錄的：

```text
train_extended_symbols.bat
```

批次檔會先檢查資料集，再由現有 17 類最佳權重開始訓練 40 類模型。正式訓練共 60 epochs。

若訓練中斷，雙擊：

```text
resume_extended_symbols.bat
```

訓練完成後最佳權重在：

```text
articulation_experiments/outputs/runs/extended_symbols_v2/weights/best.pt
```

主程式會優先自動載入這個權重；若它還不存在，仍會使用原本 17 類模型。因此在新模型訓練完成前，新符號不會真的被偵測到。

## 目前資料集

- Train：8,836 tiles
- Validation：2,323 tiles
- YOLO 原始類別：40
- 來源：DeepScoresV2 dense，沿用官方 train/validation 分割

## 已接入 MusicXML 的功能

- 力度記號會寫成 MusicXML dynamics。
- Hairpin 會寫成 crescendo/diminuendo wedge。
- 上下弓、指法、顫音與琶音會掛在對應 note/chord。
- 踏板目前寫成 `Ped.` 與 `*` 方向文字，MuseScore 可直接顯示。

DeepScores 是合成資料，實拍或不同排版的鋼琴譜仍可能有 domain gap。新模型完成後應先用數頁實際樂譜檢查，再調 confidence 與音符關聯規則。
