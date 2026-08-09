# OMR25 piano integration

Updated: 2026-08-10 (Asia/Taipei)

## What is active

OMR25 now prefers the locally trained models when their files exist:

- Piano50 symbols: `C:\OMR_work\experiments\runs\yolov9_e_piano50_dense_parentheses_30ep_b4_3090\weights\best.pt`
- curve v2: `C:\OMR_work\experiments\runs\yolov9_e_curve_v2_fullbbox_2048_30ep_3090\weights\best.pt`

`jsonTemplate.json` is the four-part example. `pianoConfigTemplate.json` is the
two-staff piano example. On a server, copy the two checkpoints and datasets,
then change the `weights`, `data_yaml`, and `yolov9_root` paths in the piece
configuration. Model files, datasets, and generated runs must not enter Git.

## Decision pipeline

1. Piano50 detects small symbols at a raw threshold of 0.05.
2. Per-class thresholds decide which candidates continue.
3. Parenthesis-cleaned image views recover symbols inside parentheses.
4. Wide or weak pedal proposals require constrained OCR. Recognized
   `cresc./decresc./dim.` become text; unresolved word-shaped boxes are
   suppressed and cannot create pedal MusicXML.
5. YOLO hairpins must pass wedge geometry; an OpenCV wedge fallback covers
   hairpins missed by YOLO.
6. Curve v2 detects one visual `curve` class. Full-box duplicates are collapsed
   without unioning nested curves. Crossing staff lines is not an automatic
   rejection.
7. Curve endpoints must attach to two ordered note groups. Same-pitch adjacent
   single notes may become ties; other complete relations become slurs.
8. Incomplete, duplicate, ambiguous, or low-confidence relations stay in JSON
   for review and do not enter MusicXML.
9. Tuplets enter MusicXML only when the numeral resolves to the expected note
   groups and its ratio is unambiguous.

The number beside a box is detector confidence, not page accuracy and not final
MusicXML correctness.

## Local checks

```powershell
& 'C:\Users\minemine\miniconda3\envs\omr\python.exe' -m unittest discover `
  -s 'C:\OMR_work\25-omr\articulation_experiments\tests' -p 'test_*.py'
& 'C:\Users\minemine\miniconda3\envs\omr\python.exe' -m unittest discover `
  -s 'C:\OMR_work\25-omr\slur_tie_experiments\tests' -p 'test_*.py'
```

One-page model/postprocessing smoke:

```powershell
& 'C:\Users\minemine\miniconda3\envs\omr\python.exe' `
  'C:\OMR_work\25-omr\server\smoke_integrated_piano_page.py' `
  --image 'C:\path\to\piano-page.jpeg' `
  --output-dir 'C:\OMR_work\experiments\omr25_integrated_smoke' `
  --device 0
```

Full OMR25 still uses `pdf2musicXML.py` and the piece in
`string_dataset/piecesToRun.json`. A piano piece uses `score_mode: piano`,
`numTrack: 2`, `track_shift: [0,0]`, and `clef_options: [[1],[-1]]`.

## Current boundary

A BPSD page completed the whole program and produced a two-part braced
MusicXML file. This proves the integration is executable. It does not prove
correct piano transcription: the same run reported a 7/8 reconstructed bar in
a configured 4/4 passage. Voice separation, cross-staff notation, and bar-level
rhythm completeness remain the main piano-core work.

