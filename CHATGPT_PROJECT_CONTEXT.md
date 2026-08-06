# ChatGPT / Codex project context: piano OMR

Last updated: 2026-08-07 (Asia/Taipei)

This is the canonical handoff document for a new ChatGPT/Codex session. Read it
before proposing server training or modifying the OMR pipeline. Update it in the
same commit whenever the project state materially changes.

## 1. Goal

The target is printed **piano score** OMR that eventually produces MusicXML.
The immediate focus is reliable detection and structural use of:

- staccato and other point articulations;
- dynamics, pedal, hairpins, tuplets, and parenthesized symbols;
- textual directions such as `cresc.`, `decresc.`, and `dim.`;
- complete slur/tie curves and their start/end note relationships.

The teacher requested YOLOv9 for staccato. Other tasks do not have to use the
same model when another method is technically better.

## 2. Repository and Git state

- GitHub: `https://github.com/885art/OMR_project.git`
- Working branch: `feature/yolov9-migration`
- The stable source of truth for the current commit is `git log -1`; do not copy
  a commit hash from this document into scripts.
- Large data, experiment output, OCR models, and `.pt` files are intentionally
  not stored in Git.

## 3. Local development environment

- OS: Windows
- Repository: `C:\OMR_work\25-omr`
- Python: `C:\Users\minemine\miniconda3\envs\omr\python.exe`
- Local GPU: one NVIDIA RTX 3090, 24 GB VRAM
- Official YOLOv9 checkout: `C:\OMR_work\yolov9`
- Experiments: `C:\OMR_work\experiments`
- Pretrained YOLOv9-E: `C:\OMR_work\weights\yolov9-e.pt`
- EasyOCR English model: `C:\OMR_work\weights\easyocr`

These paths are examples for the local Windows machine. Server scripts must use
environment variables and Linux paths, never hard-code these values.

## 4. Available local data and weights

### DeepScores

- Dense source exists locally and was used for the current symbol and curve
  experiments.
- The full 50-class Dense symbol dataset has now been converted and validated:
  `C:\OMR_work\experiments\datasets\piano50_dense_parentheses`.
  It contains 25,529 training images and 6,662 validation images after adding
  4,440 parenthesized training tiles. The underlying validated base dataset has
  27,751 tiles and 87,065 instances.
- Complete source also exists locally, but blindly training all Complete data is
  **not** the agreed final strategy.
- DeepScores is useful as generic pretraining data, not as a substitute for the
  target piano domain.

### BPSD piano pages

- 243 Beethoven piano JPEG pages are in:
  `C:\OMR_work\BPSD_score_scan_jpeg`
- The user says BPSD has already been fully annotated.
- The annotations are **not currently on this machine or in Git**. The next
  agent must request the complete annotation export, class list, image-name
  mapping, and relationship/mask information before building the final dataset.
- Accept COCO JSON, YOLO, CVAT, Label Studio, Pascal VOC, MusicXML-aligned data,
  masks, or polylines; inspect before deciding the converter.

### Existing local models (not in Git)

- Current 40-class YOLOv9 symbol model:
  `C:\OMR_work\experiments\runs\yolov9_e_dense_20ep_b4_3090\weights\best.pt`
- Current two-class YOLOv9 slur/tie model:
  `slur_tie_experiments\outputs\runs\yolov9_curves_v1\weights\best.pt`
- These are preliminary models, not the final BPSD-trained piano models.

## 5. What is implemented

### Symbol dataset and training

- Tiny-object tiling supports a 512 px crop enlarged to 1024 model input.
- `class_mapping_piano.json` expands the existing 40 detector classes to 50 by
  adding tuplet numbers 1-9 and `tupletBracket`.
- Parenthesis augmentation draws parentheses while keeping the label on the
  inner musical symbol.
- A static DeepScores dataset browser can show class counts and crop examples.
- Windows RTX 3090 preparation/training scripts exist for Dense 50-class smoke
  and full runs. The full augmented 50-class dataset passed validation and
  single-GPU YOLOv9 preflight locally on 2026-08-07. No full 50-class training
  run has been started yet.
- Local one-click launcher (outside Git):
  `C:\OMR_work\START_PIANO50_TRAIN_20EP.bat`. It starts YOLOv9-E for 20 epochs,
  batch size 4, image size 1024, and writes the run under
  `C:\OMR_work\experiments\runs\yolov9_e_piano50_dense_parentheses_20ep_b4_3090`.

### Piano inference/postprocessing

- Parenthesis-robust inference runs the original page and a conservative
  parenthesis-suppressed view.
- `cresc./decresc./dim.` currently use dictionary-constrained EasyOCR.
- Low-confidence pedal proposals can be reclassified as text directions.
- Hairpins use low-threshold YOLO candidates plus wedge geometry validation.
- Curve fragments can be merged; obvious multi-staff-line and tiny/flat false
  boxes are rejected.
- A review output assigns `accept`, `review`, or `reject` instead of treating the
  displayed confidence as a correctness probability.
- Tuplet relations can be attached conservatively to note groups and written to
  MusicXML when the ratio and span are unambiguous.
- Slurs and ties are written only after both endpoints are resolved. A tie also
  requires structurally compatible adjacent same-pitch notes.

### Representative local review output

- Six-page integrated review:
  `C:\OMR_work\experiments\piano_validation\combined_review_v2`
- Eight additional cross-staff pages:
  `C:\OMR_work\experiments\piano_validation\more_cross_staff_review`
- YOLO versus original OpenCV curve comparison:
  `C:\OMR_work\experiments\piano_validation\more_cross_staff_hybrid`
- DeepScores class browser:
  `C:\OMR_work\experiments\dataset_browser\complete_relevant\index.html`

These outputs are local diagnostics and are not committed.

## 6. Known limitations and evidence

### Text directions

DeepScores does not provide YOLO classes for printed words such as `cresc.` and
`decresc.`. They should normally be handled by pretrained OCR plus a restricted
music-direction dictionary, not by pretending an untrained YOLO class exists.

The current OCR trigger still relies partly on low-confidence pedal proposals.
The desired improvement is staff-aware OCR over inter-staff text bands so a word
can be found even when YOLO proposes no pedal box.

### Slur/tie domain shift

The current curve model was trained for 20 epochs from DeepScores Dense:

- 12,345 source slurs;
- 14,034 source ties;
- 42,855 tiled curve instances;
- 5,557 bounding boxes clipped by tiling;
- tie height median approximately 11 px and 95th percentile approximately
  13 px.

This helps explain the failure mode on dense piano scores: clipped training
labels teach fragments, and extremely flat tie boxes resemble staff lines and
beams. String-quartet pages are visually sparser and therefore looked better.

On piano pages, YOLO works best for isolated curves outside the staff. Curves
crossing staff lines, stems, beams, or dense chords are frequently fragmented or
missed. The original OpenCV staff-removal/curve-fit method recovers some curves
but also creates many beam/staff false positives. Do not union all OpenCV and
YOLO results.

## 7. Agreed final modeling strategy

### A. Point symbols and small musical objects

Use YOLOv9. Compare two controlled experiments on the same held-out BPSD split:

1. generic YOLOv9 weights -> BPSD fine-tune;
2. DeepScores Dense/Complete pretraining -> BPSD fine-tune.

Select using BPSD validation; do not assume Complete is better merely because
it is larger. If mixing data, ensure BPSD is not overwhelmed, and always finish
with a BPSD-focused fine-tune.

### B. Slur and tie

Preferred design:

1. detect one visual class, `curve`;
2. resolve the two note endpoints;
3. classify structurally: adjacent same-pitch notes -> tie, otherwise a complete
   valid curve -> slur;
4. write MusicXML only for complete endpoint pairs.

If BPSD supplies masks or polylines, prefer instance segmentation or a curve
tracing model. If BPSD supplies bounding boxes only, start with a one-class
YOLOv9 curve baseline. Use curve-aware crops that retain whole curves; do not
train clipped fragments as complete objects. Include both small-detail and
larger staff-system context.

The OpenCV result may raise confidence when it agrees with YOLO, but OpenCV-only
candidates remain review-only unless structural endpoint checks pass.

### C. Text directions

Use staff-aware OCR plus a dictionary for `cresc`, `crescendo`, `decresc`,
`decrescendo`, `dim`, and `diminuendo`. Do not add YOLO text classes unless a
measured OCR recall problem justifies a separate text-region detector.

## 8. BPSD split and evaluation rules

- Split by sonata/Opus, never randomly by page or tile.
- Target split: about 70% train, 15% validation, 15% test.
- All pages and tiles from one work must remain in one split.
- Pages already examined repeatedly (including examples from Op.13, Op.27,
  Op.31, Op.53, and Op.57) should be train/validation, not the final untouched
  test set.
- Freeze the test manifest before training decisions.
- Report per-class precision/recall/F1, missed long/cross-staff curves, endpoint
  pair accuracy, slur-versus-tie structural accuracy, and final MusicXML event
  accuracy. mAP alone is insufficient.

## 9. Server-training status

The user plans to train on a server and has mentioned a possible two-RTX-3090
machine. Exact server OS, GPU count/model, CUDA/PyTorch environment, storage
paths, and scheduler are not yet confirmed.

Existing Linux scripts in `server/` support the older 40-class DeepScores flow:

- `prepare_symbols_v2.sh`
- `train_yolov9_symbols.sh`
- `slurm_yolov9_symbols.sbatch`
- `nchc_env.example`

Important: the current Linux script expects 40 classes. The newer 50-class
piano helper is currently Windows-oriented. Do **not** tell the user the final
BPSD server pipeline is ready. It cannot be finalized until the BPSD annotation
format and server environment are known.

For two 3090s, first use one GPU per experiment to compare BPSD-only versus
generic-pretrain-plus-BPSD in parallel. After the data recipe is selected, test
YOLOv9 DistributedDataParallel on a smoke dataset before using both GPUs for the
final run. Two cards do not combine into one 48 GB memory pool.

## 10. Required next inputs

Ask the user for:

1. the complete BPSD annotation export;
2. the class list and image filename mapping;
3. whether slur/tie annotations are boxes, masks, polylines, endpoints, or note
   relationships;
4. server OS and access workflow;
5. GPU model/count, CUDA/PyTorch versions, CPU/RAM, and fast-storage path;
6. whether the scheduler is Slurm and the required account/partition fields.

Never request or store passwords/tokens in Git. Use an ignored environment file
for private server values.

## 11. Validation already completed

- 18 focused `unittest` integration tests passed locally.
- `pip check` reported no broken requirements in the local `omr` environment.
- The 50-class smoke dataset passed dataset validation.
- The 50-class smoke dataset passed YOLOv9 RTX 3090 preflight.
- The full 50-class Dense base dataset passed validation with 27,751 tiles and
  87,065 instances.
- The augmented full dataset passed its derived-dataset checks with 25,529
  training images, 6,662 validation images, 4,440 parenthesized images, and 50
  classes. YOLOv9 RTX 3090 preflight also passed against this dataset.
- EasyOCR and `music21` are installed locally; tuplet MusicXML output was tested.

Useful command:

```powershell
& 'C:\Users\minemine\miniconda3\envs\omr\python.exe' -m unittest discover `
  -s 'C:\OMR_work\25-omr\articulation_experiments\tests' -v
```

## 12. Important files

- `PIANO_OMR_CHANGES.md`: concise local Windows usage notes.
- `articulation_experiments/dataset/class_mapping_piano.json`: current 50-class
  DeepScores mapping.
- `articulation_experiments/dataset/augment_parentheses.py`: parenthesis data
  augmentation.
- `articulation_experiments/dataset/browse_deepscores_dataset.py`: static class
  browser.
- `articulation_experiments/inference/review_piano_pages.py`: integrated review
  output.
- `articulation_experiments/inference/review_curve_hybrid.py`: YOLO/OpenCV curve
  comparison.
- `omr/articulation.py`: symbol inference, thresholds, association.
- `omr/slur_tie.py`: curve detection, endpoint association, MusicXML rules.
- `omr/text_directions.py`: constrained OCR.
- `omr/hairpin.py`: wedge validation.
- `omr/tuplet.py`: tuplet association and MusicXML attachment.
- `jsonTemplate.json`: current local runtime defaults.
- `server/`: Linux/NCHC and Windows training helpers.

## 13. Handoff prompt for another ChatGPT

The user can paste this:

> Read `AGENTS.md` and `CHATGPT_PROJECT_CONTEXT.md` completely. We are building
> printed piano OMR and will train on a server. Do not propose blindly training
> all DeepScores Complete. First inspect the BPSD annotation export and server
> environment, preserve an Opus-level untouched test split, then prepare Linux
> smoke, BPSD-only, generic-pretrain-plus-BPSD, and final evaluation workflows.
> Update `CHATGPT_PROJECT_CONTEXT.md` with every material decision or result.

## 14. Change log

- 2026-08-07: Prepared and validated the full local DeepScores Dense piano-50
  dataset with parenthesis augmentation. Recorded exact counts, passed the
  single-RTX-3090 preflight, and created a local one-click 20-epoch launcher.
  Training was intentionally not started so the user can launch it manually.
- 2026-08-06: Created the canonical AI handoff document and repository update
  policy. Recorded current implementation, known curve-domain failure modes,
  agreed BPSD-first strategy, server blockers, and validation status.
