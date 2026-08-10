# ChatGPT / Codex project context: piano OMR

Last updated: 2026-08-10 (Asia/Taipei)

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
- Curve v2 training completed on 2026-08-09. It uses one visual `curve` class
  and full-curve crops. The selected checkpoint is
  `C:\OMR_work\experiments\runs\yolov9_e_curve_v2_fullbbox_2048_30ep_3090\weights\best.pt`.
  The local dataset is
  `C:\OMR_work\experiments\datasets\curve_v2_fullbbox_2048`.
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
  single-GPU YOLOv9 preflight locally on 2026-08-07. The local 30-epoch run
  completed successfully on 2026-08-08; the selected `best.pt` is active in
  `jsonTemplate.json`.
- Local one-click launcher (outside Git):
  `C:\OMR_work\START_PIANO50_TRAIN_30EP.bat`. It starts YOLOv9-E for a maximum
  of 30 epochs with early-stopping patience 8,
  batch size 4, image size 1024, and writes the run under
  `C:\OMR_work\experiments\runs\yolov9_e_piano50_dense_parentheses_30ep_b4_3090`.
  On the DeepScores validation split, the best checkpoint was epoch 26 with
  precision 0.9898, recall 0.9837, mAP@0.5 0.9903, and mAP@0.5:0.95 0.97734.
  These are source-domain validation metrics, not BPSD piano test accuracy.

### Piano inference/postprocessing

- Parenthesis-robust inference runs the original page and a conservative
  parenthesis-suppressed view.
- `cresc./decresc./dim.` currently use dictionary-constrained EasyOCR.
- Low-confidence pedal proposals can be reclassified as text directions.
- Hairpins use low-threshold YOLO candidates plus wedge geometry validation.
- Curve fragments can be merged; obvious multi-staff-line and tiny/flat false
  boxes are rejected.
- Detector confidence remains a candidate score, not a correctness probability.
- At the user's direction after visual review, the default runtime policy is now
  `all_detected`: displayed articulation, tuplet, and curve candidates are kept
  and assigned to the lowest-cost available note group or ordered endpoint pair.
- A tie still requires structurally compatible adjacent same-pitch single notes.
  Ambiguous chord ties are preserved as slurs so that the curve is not discarded
  and an invalid pitch-specific tie is not created.
- `conservative` remains available as an explicit `acceptance_policy` setting.
- Candidate retention and MusicXML insertion are distinct: if legacy note parsing
  does not create a music21 object for an associated note group, the record remains
  in JSON/debug output but has no MusicXML object to attach to.

### Representative local review output

- Six-page integrated review:
  `C:\OMR_work\experiments\piano_validation\combined_review_v2`
- Eight additional cross-staff pages:
  `C:\OMR_work\experiments\piano_validation\more_cross_staff_review`
- YOLO versus original OpenCV curve comparison:
  `C:\OMR_work\experiments\piano_validation\more_cross_staff_hybrid`
- DeepScores class browser:
  `C:\OMR_work\experiments\dataset_browser\complete_relevant\index.html`
- New 50-class model domain test on ten BPSD piano pages and ten string-quartet
  pages:
  `C:\OMR_work\experiments\piano50_eval_20260808_20pages\index.html`

These outputs are local diagnostics and are not committed.

The 20-page domain test used confidence 0.25 with parenthesis-robust inference
and hairpin geometry validation. BPSD produced 1,314 detections (mean model
confidence 0.878); the string pages produced 1,710 (mean 0.875). The source
folders do not contain matching ground-truth annotations, so these confidence
values are not accuracy. Visual review found useful staccato and dynamic-symbol
detections, but frequent high-confidence fingering false positives, some
fermata/page-number and text/pedal confusion, weak tuplet recall, and incomplete
hairpin recall. Do not cite the DeepScores validation mAP as piano-domain
accuracy.

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

The replacement curve-v2 dataset was prepared and validated on 2026-08-09:

- DeepScores slur 121 and tie 123 both map to YOLO class 0, `curve`;
- 1,714 source pages, 4,474 output tiles, and 26,379 source curves;
- 52,269 duplicated tile instances for overlapping-context coverage;
- 0 clipped training labels and 0 unassigned source annotations;
- 2048 full-curve crops with 1024 overlap plus target-centered crops where the
  regular grid cannot contain the complete curve.

The local training launcher uses YOLOv9-E at model input 1280, batch 3, maximum
30 epochs, and early-stopping patience 10. The formal run completed all 30
epochs. The best checkpoint is epoch 29 with precision 0.96812, recall 0.94299,
mAP@0.5 0.97980, and mAP@0.5:0.95 0.85626. These remain DeepScores pretraining
metrics, not BPSD accuracy.

The curve-v2 checkpoint was run on the same fixed ten BPSD and ten string pages
as the piano50 domain review. At confidence 0.25 it produced 490 BPSD and 985
string curve boxes. Visual review shows substantially better long/cross-staff
curve and small-tie coverage than the old two-class model. Remaining problems
include duplicate/overlapping boxes on some very long curves, hairpin confusion,
and the old multi-staff-line geometry rule flagging some legitimate long curves.
The comparison gallery preserves both accepted and flagged proposals at
`C:\OMR_work\experiments\piano50_curvev2_eval_20260809_20pages\index.html`.
These pages have no matching ground truth, so counts and model confidence are
not accuracy.

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
3. classify structurally: adjacent same-pitch single notes -> tie, otherwise a
   curve -> slur;
4. under the current user-selected `all_detected` runtime policy, retain every
   displayed candidate and use the lowest-cost ordered endpoint pair when the
   normal endpoint gate fails; write MusicXML when both resolved note groups
   produced attachable music21 objects.

If BPSD supplies masks or polylines, prefer instance segmentation or a curve
tracing model. If BPSD supplies bounding boxes only, start with a one-class
YOLOv9 curve baseline. Use curve-aware crops that retain whole curves; do not
train clipped fragments as complete objects. Include both small-detail and
larger staff-system context.

The OpenCV result may raise confidence when it agrees with YOLO. Historical
comparison galleries may still contain review-only candidates; the integrated
runtime now follows the explicit `acceptance_policy` described above.

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
- The local YOLOv9-E 30-epoch run completed with 30 result rows and produced
  `weights/best.pt` and `weights/last.pt`. The best source-domain mAP@0.5:0.95
  was 0.97734 at epoch 26.
- EasyOCR and `music21` are installed locally; tuplet MusicXML output was tested.
- The full one-class curve-v2 dataset passed source/manifest/label validation:
  4,474 tiles, 52,269 tile instances, 0 clipped labels, and 0 unassigned source
  annotations. YOLOv9-E RTX 3090 preflight passed. A separate 1-epoch 1280,
  batch-4 smoke completed training, validation, and checkpoint writing.
- On 2026-08-10, 26 articulation/piano tests and 14 slur/tie tests passed after
  the OMR25 integration and all-detected policy changes.
- A real BPSD page completed the full legacy `pdf2musicXML.py` pipeline in
  two-staff piano mode. After enabling `all_detected`, it produced a braced
  two-part MusicXML with 451 notes, 30 directions, 51 staccato marks, 17
  fingerings, four tuplet tags, and 60 slurs. The run also exposed a 7/8
  reconstructed bar in a configured 4/4 passage, so it is an executable
  integration smoke and retention check, not a piano-accuracy result.
- The local environment additionally required `onnxruntime-gpu==1.18.0`,
  `scikit-learn==1.7.0`, and `pdf2image==1.17.0` for the legacy main program.
  ONNX Runtime could not load its CUDA provider DLL and fell back to CPU; both
  YOLOv9-E detectors did run on the RTX 3090.

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
- `articulation_experiments/inference/build_curve_v2_comparison_gallery.py`:
  fixed-page previous-symbol, curve-only, and combined visual comparison.
- `omr/articulation.py`: symbol inference, thresholds, association.
- `omr/slur_tie.py`: curve detection, endpoint association, MusicXML rules.
- `omr/text_directions.py`: constrained OCR.
- `omr/hairpin.py`: wedge validation.
- `omr/tuplet.py`: tuplet association and MusicXML attachment.
- `jsonTemplate.json`: current local runtime defaults.
- `server/`: Linux/NCHC and Windows training helpers.
- `CURVE_V2_TRAINING.md`: curve-v2 local data recipe, start command, output,
  and recovery notes.
- `OMR25_PIANO_INTEGRATION.md`: runtime rules, model paths, smoke command, and
  the current piano boundary.
- `TEACHER_REPORT_PIANO_OMR.md`: evidence-backed teacher presentation outline
  and speaking points.
- `pianoConfigTemplate.json`: two-staff piano configuration and local model
  settings.
- `server/smoke_integrated_piano_page.py`: one-page real-model integration
  smoke without the complete legacy note/rhythm pipeline.

## 13. Handoff prompt for another ChatGPT

The user can paste this:

> Read `AGENTS.md` and `CHATGPT_PROJECT_CONTEXT.md` completely. We are building
> printed piano OMR and will train on a server. Do not propose blindly training
> all DeepScores Complete. First inspect the BPSD annotation export and server
> environment, preserve an Opus-level untouched test split, then prepare Linux
> smoke, BPSD-only, generic-pretrain-plus-BPSD, and final evaluation workflows.
> Update `CHATGPT_PROJECT_CONTEXT.md` with every material decision or result.

## 14. Change log

- 2026-08-10: Added `server/import_bpsd_piano_images.py` to prepare the 243
  BPSD JPEG pages for the legacy OMR25 input layout without changing source
  files. The importer validates the final numeric page suffix, rejects gaps and
  duplicates, groups the local pages into 32 pieces, creates real PNG files and
  piano JSON configs, and records a source-to-output manifest plus a separate
  piece list. A dry run and full local import both succeeded: 32 pieces, 243
  pages, and zero missing outputs. Generated configs deliberately mark the
  initial 4/4 `tsChange` as needing manual review; normal reruns preserve edited
  configs unless `--overwrite` is explicitly requested. Generated pages/configs
  remain ignored dataset artifacts and are not committed.
- 2026-08-10: Switched the integrated piano runtime defaults to the user-selected
  `all_detected` policy. All displayed articulation candidates are now associated,
  including far/duplicate candidates; tuplets receive a nearest same-staff fallback;
  and curve candidates bypass confidence/distance/duplicate relation suppression,
  with a nearest ordered endpoint fallback. Ambiguous chord ties are exported as
  slurs rather than discarded or emitted as invalid ties. On the same BPSD smoke
  page, articulation association changed from 85/105 to 105/105 and curve relations
  changed to 68/68 XML-eligible. The actual MusicXML contains 51 staccatos, 17
  fingerings, 27 dynamics, three text directions, and 60 slurs; seven articulation
  records and eight curve relations remain only in JSON/debug output because legacy
  note parsing did not create attachable music21 note objects. This is retention
  behavior, not a ground-truth accuracy result.
- 2026-08-10: Integrated the Piano50 and curve-v2 checkpoints into OMR25's
  preferred runtime path. Added constrained pedal/text arbitration, direct
  geometry hairpin fallback, full-box curve duplicate collapse, staff-crossing
  curve retention, duplicate endpoint-relation suppression, conservative
  MusicXML gates, and two-staff piano brace output. Completed focused tests, a
  real-model BPSD smoke, and a full one-page OMR25 MusicXML run. The run proved
  executable integration while exposing a remaining 4/4-versus-7/8 rhythm
  error; BPSD ground-truth accuracy is still unavailable. Added technical and
  teacher-report handoff documents. Both handoff documents were subsequently
  rewritten in Traditional Chinese for the local team and teacher review.
- 2026-08-09: Completed the 30-epoch curve-v2 run; selected epoch 29 `best.pt`
  and recorded its DeepScores metrics. Re-ran the fixed 20-page domain set and
  generated a three-way symbol/curve/combined gallery. Updated candidate export
  to represent the many-to-one detector output as `curve` instead of arbitrarily
  reporting one source label. Visual review found much better curve coverage,
  with long-curve duplicates and geometry/hairpin filtering still unresolved.
- 2026-08-09: Implemented and validated the one-class full-curve v2 DeepScores
  pipeline. Added many-to-one source-class mapping, full-bbox-only labels,
  target-centered recovery crops, partial-overlap negative suppression, a local
  RTX 3090 preparation script, preflight, and one-click 30-epoch launcher. The
  full dataset has 4,474 tiles and 52,269 instances with zero clipped or
  unassigned annotations. A separate 1280/batch-4 GPU smoke completed; the
  formal launcher defaults to batch 3 and training was intentionally not
  started.
- 2026-08-08: Ran the new piano50 model on ten distributed BPSD piano pages and
  ten distributed string-quartet PDF pages. Created a local 20-page gallery and
  recorded the domain-shift findings: staccato/dynamics are useful, while
  fingering, text/pedal, fermata, tuplet, and hairpin behavior still needs
  target-domain labels and postprocessing.
- 2026-08-08: Completed the local YOLOv9-E piano50 Dense training run. Recorded
  the best source-domain metrics and confirmed `jsonTemplate.json` points to
  the resulting `best.pt`. BPSD evaluation is still required before claiming
  target-domain accuracy.
- 2026-08-07: Changed the local one-click run from 20 to a maximum of 30 epochs
  with early-stopping patience 8; the selected output remains `best.pt`.
- 2026-08-07: Prepared and validated the full local DeepScores Dense piano-50
  dataset with parenthesis augmentation. Recorded exact counts, passed the
  single-RTX-3090 preflight, and created a local one-click 20-epoch launcher.
  Training was intentionally not started so the user can launch it manually.
- 2026-08-06: Created the canonical AI handoff document and repository update
  policy. Recorded current implementation, known curve-domain failure modes,
  agreed BPSD-first strategy, server blockers, and validation status.
