# ChatGPT / Codex project context: piano OMR

Last updated: 2026-08-25 (Asia/Taipei)

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
- One legacy 5.6 MB baseline checkpoint under
  `articulation_experiments/outputs/runs/baseline_v1/weights/best.pt` was already
  tracked historically and is still referenced by legacy defaults. The global
  `*.pt` ignore prevents new checkpoints but does not untrack that existing
  exception; do not add further model weights.

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
- On 2026-08-11 the teacher additionally requested an all-class DeepScores
  experiment. The canonical detector has 136 classes (DeepScores IDs 1--136),
  not all 208 category-table rows: IDs 137--208 are the MUSCIMA++ compatibility
  annotation set and duplicate several names.
- The full Dense all136 dataset is validated at
  `C:\OMR_work\experiments\datasets\deepscores_dense_all136_1024`: 1,714 source
  pages, 21,842 tiles, and 2,583,651 tile instances. Dense train contains
  annotations for 115 of 136 classes; Complete is needed for missing/rare classes.
- Complete source has 103 train shards, 26 test shards, and 255,385 images.
  Complete all136 must use the new sharded converter rather than one enormous
  merged JSON.
- The Berlioz full Complete conversion is now finished and validation passed:
  204,308 train source pages produced 2,329,441 train tiles; 51,077 official
  test pages (used here as validation) produced 584,820 validation tiles.
  The converted dataset remains on server storage and is not committed.
- DeepScores is useful as generic pretraining data, not as a substitute for the
  target piano domain.

### BPSD piano pages

- 243 Beethoven piano JPEG pages are in:
  `C:\OMR_work\BPSD_score_scan_jpeg`
- The full YOLO annotation export is available at `C:\OMR_work\BPS-OMRv01`:
  243 images, 243 label files, 244 source classes, and 41,874 raw boxes.
- A frozen work-level split is stored in
  `articulation_experiments/dataset/bps_work_split_v1.json`: 170 train pages
  from 21 works, 36 validation pages from five works, and 37 final-test pages
  from six works. There is no work leakage.
- BPS supplies bounding boxes, not curve masks, polylines, endpoints, or note
  relationships. Slur/tie endpoint and semantic decisions therefore remain an
  OMR25 postprocessing responsibility.
- One source tie box in `Beethoven_Op007-01-06.txt` is fully outside the page.
  The converter rejects and audits it without changing the source annotation.

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
- Dense all136 visual comparison on the same fixed ten BPSD piano pages and ten
  string-quartet pages:
  `C:\OMR_work\experiments\dense_all136_eval_20260813_20pages\index.html`.
  It places the prior Piano50+curve-v2 output beside all136 overview, core
  notation, expressive/curve, and layout-structure views. The folder includes
  portable source/previous images and merged prediction JSON.

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

### D. Teacher-requested DeepScores all-class experiment

Train YOLOv9-E on all 136 canonical DeepScores classes in this order:

1. Dense all136 on the local RTX 3090;
2. upload the Dense all136 `best.pt` to the server;
3. continue training on Complete all136, preferably on one H100;

As of 2026-08-18, the teacher explicitly requested that this training stage not
use BPSD. BPSD conversion, replay mixing, and fine-tuning are therefore outside
the active Complete all136 server workflow unless the teacher changes that
decision later. Evaluation claims must still distinguish DeepScores
source-domain metrics from real piano-score accuracy.

This experiment does not automatically replace the production 50-class symbol
plus one-class curve pipeline. It adds noteheads, stems, beams, staffs, rests,
clefs, slur, tie, and other structural classes whose OMR25 association and
MusicXML logic still need a separate integration design.

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

The user plans to train Complete all136 on a server that may have an H100.
Exact H100 memory, server OS, CUDA/PyTorch environment, storage paths, and
scheduler configuration are not yet confirmed.

The repository now includes both Windows and Linux/Slurm preparation and
training entry points for the 50-class BPS symbol fine-tune, the one-class BPS
curve fine-tune, and a separate 50-class DeepScores Complete experiment. The
local BPS datasets pass validation and both Windows fine-tune launchers pass
RTX 3090 preflight. The Linux/Slurm scripts have not yet completed a smoke on
the actual server, so do **not** claim server readiness until that succeeds.

The all136 server path uses `convert_deepscores_complete_sharded.py`, bounded to
one source shard in CPU memory and resumable per shard. Its master train/val
  indexes passed a real local one-epoch RTX 3090 smoke. The full Berlioz
  conversion later passed validation at 2,329,441 train and 584,820 validation
  tiles. H100 defaults use image size 1024 and batch 12; the measured Complete
  throughput is about 0.557 seconds/batch and roughly 30 hours per train epoch.
  Data conversion is CPU/RAM/NVMe-bound and is not accelerated materially by the
  H100.

The server workflow now keeps `ALL136_SMOKE_DATASET` separate from the formal
`ALL136_DATASET`, rejects a full conversion unless all expected 103 train and 26
test shards are present, provides a CPU-only Slurm preparation job, and provides
  a server file/environment checker. The formal run continues from the local
  Dense all136 `best.pt`, does not reference BPSD, and now defaults to two full
  epochs rather than 30. Exact mid-epoch resume is not supported by upstream
  YOLOv9; only completed-epoch checkpoints are safe restart points.

Complete conversion now supports bounded shard-level concurrency through
`--workers N` and server variable `COMPLETE_CONVERSION_WORKERS`, both defaulting
to 1. Workers launch independent `convert_deepscores_to_yolo.py` subprocesses;
only the main process performs deterministic validation and master aggregation
after every required conversion succeeds. Worker count is deliberately excluded
from the content fingerprint, so Berlioz can change from 1 to 8 and safely reuse
the roughly 15 completed chunks or resume a matching interrupted `train_015`.
Start Berlioz at 4–8 conversion workers rather than its full 96 logical cores;
this phase is CPU/RAM/storage-I/O bound and does not use the H100.

Complete chunk resume is now recipe-safe: each partial and completed chunk
records the source fingerprint, mapping and base-converter SHA256 values, and
the tiling/sampling/compression/limit parameters. A mismatch fails closed and
requires a new output directory or explicit `--overwrite-chunks`; legacy chunks
without the new progress fingerprint are not silently reused. The current
engineering pretraining split still maps the 26 official Complete test shards
to YOLO validation for early stopping, so those metrics must not be reported as
untouched official test performance. A publication-grade experiment would hold
validation out of the 103 official train shards and evaluate official test only
after model selection; that split change is intentionally deferred.

For two 3090s, first use one GPU per experiment to compare BPSD-only versus
generic-pretrain-plus-BPSD in parallel. After the data recipe is selected, test
YOLOv9 DistributedDataParallel on a smoke dataset before using both GPUs for the
final run. Two cards do not combine into one 48 GB memory pool.

## 10. Required next inputs

Still required for actual server execution:

1. server OS and access workflow;
2. GPU model/count, CUDA/PyTorch versions, CPU/RAM, and fast-storage path;
3. whether the scheduler is Slurm and its required account/partition fields.

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
- Both formal BPS target datasets and their 1:1 replay mixes passed validation.
  Symbols contain 5,057/1,075/1,101 target tiles for train/val/test; the mixed
  symbol train contains 10,114 tiles. Curves contain 340/72/74 target tiles;
  the mixed curve train contains 680 tiles. Replay is train-only, while BPS
  validation and final test remain pure target-domain data.
- Both BPS fine-tune Windows launchers passed RTX 3090 preflight against the
  validated 50-class and one-class datasets. No BPS fine-tuning run has been
  started yet.
- A 50-class DeepScores Complete smoke converted 560 source pages into 8,395
  tiles and 23,596 tile instances with zero validation errors; its RTX 3090
  preflight passed. The very large full Complete conversion has not been
  started automatically.
- The formal Dense all136 conversion passed full trace validation: 17,281 train
  tiles plus 4,561 validation tiles, 2,583,651 tile instances, zero unassigned
  annotations, and 458 audited nonpositive source boxes dropped. The validated
  dataset passed RTX 3090 136-class preflight.
- The formal Dense all136 YOLOv9-E run completed all 30 epochs on 2026-08-13 at
  `C:\OMR_work\experiments\runs\yolov9_e_dense_all136_30ep_b4_3090`.
  The final/best epoch reported precision 0.97496, recall 0.94792, mAP@0.5
  0.96553, and mAP@0.5:0.95 0.91764. The highest mAP@0.5:0.95 occurred at
  epoch 30, while precision/recall/mAP@0.5 peaked at epochs 27/26/28. Training
  losses decreased throughout. These are Dense source-domain metrics averaged
  over the 110 validation classes with instances, not BPS piano accuracy and
  not evidence for the 26 absent validation classes. `best.pt` contains 136
  names/classes and is the checkpoint to initialize Complete all136.
- The Complete all136 sharded smoke used ten pages from one train shard and ten
  pages from one test shard, produced 101/116 tiles, passed master validation,
  and completed a real one-epoch YOLOv9-E RTX 3090 train/validation/checkpoint
  smoke. It transferred 2,160/2,172 compatible tensors from the existing
  Piano50 checkpoint and peaked at about 18.6 GB reported training GPU memory.
  Smoke accuracy is intentionally meaningless after one epoch.
- After the resume-safety update, 45 articulation/piano dataset and integration
  tests plus 14 slur/tie tests passed. A real one-page-per-split Complete micro
  conversion produced 6/12 train/validation tiles, reused both chunks with an
  identical recipe, and rejected reuse after overlap changed from 256 to 128.
  A separate micro run confirmed that explicit `--overwrite-chunks` rebuilt
  both splits under the changed overlap and updated the stored fingerprint.
- After shard concurrency was added, all 53 articulation/piano/dataset tests
  passed, including fake-subprocess coverage for sequential and parallel jobs,
  reuse, resume, mismatch refusal, fail-closed publication, deterministic
  aggregation, worker validation, and smoke limits. A real two-worker Complete
  micro conversion concurrently loaded one train and one test shard, produced
  6/12 tiles, and completed the master dataset; rerunning the same output with
  eight workers reused both chunks without a fingerprint mismatch.
- The local YOLOv9-E 30-epoch run completed with 30 result rows and produced
  `weights/best.pt` and `weights/last.pt`. The best source-domain mAP@0.5:0.95
  was 0.97734 at epoch 26.
- EasyOCR and `music21` are installed locally; tuplet MusicXML output was tested.
- The full one-class curve-v2 dataset passed source/manifest/label validation:
  4,474 tiles, 52,269 tile instances, 0 clipped labels, and 0 unassigned source
  annotations. YOLOv9-E RTX 3090 preflight passed. A separate 1-epoch 1280,
  batch-4 smoke completed training, validation, and checkpoint writing.
- On 2026-08-10, 30 articulation/piano tests and 14 slur/tie tests passed after
  the OMR25 integration, all-detected policy, importer, and batch-review changes.
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
- A separate one-page hybrid all136 smoke completed on BPSD
  `Beethoven_Op007-01-03` at
  `C:\OMR_work\experiments\musicxml_all136_hybrid_op007_p03`. It deliberately
  kept legacy OMR25 for note/pitch/rhythm reconstruction, routed only all136
  expressive classes through the existing articulation/MusicXML stage, and
  disabled all136 slur/tie because the visual review showed severe Dense-domain
  tie false positives. The MusicXML reparsed successfully with music21: two
  parts, 60 measures across parts, 205 notes, 101 chords, and 15 rests. Of 1,282
  post-threshold all136 boxes, the semantic gate rejected 1,226 structural
  boxes; 44 of 56 supported expressive candidates associated to note groups.
  The output included 13 staccatos, 10 dynamic directions, and 21 fingerings.
  This proves conflict-free execution, not transcription accuracy; the imported
  4/4 configuration is still marked for manual time-signature review.

Useful command:

```powershell
& 'C:\Users\minemine\miniconda3\envs\omr\python.exe' -m unittest discover `
  -s 'C:\OMR_work\25-omr\articulation_experiments\tests' -v
```

## 12. Important files

- `docs/handoff/README.md`: concise human-maintainer start page. It routes
  result review, current-model inference, external assets, and retraining into
  separate short documents so successors do not need to read every experiment
  record first.
- `docs/handoff/CURRENT_RESULTS.md`: current model results, evidence boundary,
  Complete status, and teacher-safe wording.
- `docs/handoff/RUN_CURRENT_MODELS.md`: external checkpoint hashes and current
  single-page, batch, and full MusicXML execution instructions.
- `docs/handoff/RETRAINING.md`: optional training decision page linking to the
  detailed Complete/BPSD/curve recipes.
- `PIANO_OMR_CHANGES.md`: concise local Windows usage notes.
- `articulation_experiments/dataset/class_mapping_piano.json`: current 50-class
  DeepScores mapping.
- `articulation_experiments/dataset/generate_deepscores_all_mapping.py`:
  deterministic official 136-class mapping and schema validation. It maps
  expressive all136 names onto OMR25 semantic classes while leaving structural
  note/rest/stem/beam/staff classes unsupported by the articulation stage, so
  they cannot duplicate the legacy note/rhythm reconstruction in hybrid mode.
- `articulation_experiments/dataset/convert_deepscores_complete_sharded.py`:
  memory-bounded, per-shard resumable Complete conversion and master indexes.
- `articulation_experiments/dataset/convert_bps_yolo_finetune.py`: BPS symbols
  and whole-curve tiled conversion.
- `articulation_experiments/dataset/compose_finetune_replay.py`: train-only
  target/replay composition.
- `articulation_experiments/dataset/validate_finetune_yolo_dataset.py`: labels,
  coordinates, pairing, and work-leakage validation.
- `articulation_experiments/dataset/bps_work_split_v1.json`: frozen BPS split.
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
- `BPS_FINETUNING.md`: Traditional-Chinese data, training, final-test, Complete,
  and server instructions.
- `DEEPSCORES_ALL136_TRAINING.md`: Dense-first, Complete/H100 all136 workflow.
- `GPT_HANDOFF_COMPLETE_ALL136.md`: laptop clone requirements and a ready-to-paste
  prompt for another GPT to continue the no-BPSD Complete all136 workflow.
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
- `server/batch_integrated_piano_pages.py`: multi-page symbol/curve JSON and
  HTML gallery generation with one model load.

## 13. Handoff prompt for another ChatGPT

The user can paste this:

> Read `AGENTS.md` and `CHATGPT_PROJECT_CONTEXT.md` completely. We are building
> printed piano OMR. The teacher now requires a 136-class DeepScores experiment:
> train Dense all136 locally first, then continue on sharded Complete all136 on
> the server/H100. Preserve the separate BPS Opus-level test split and remember
> that the production 50-class plus curve pipeline is not automatically replaced.
> Update `CHATGPT_PROJECT_CONTEXT.md` with every material decision or result.

## 14. Change log

- 2026-08-25: Replaced the stale, training-heavy root README with a concise
  project summary and one human handoff entry. Added `docs/handoff/` pages for
  current results, external model assets with verified SHA256 values, inference
  commands, and optional retraining. Existing research/training documents were
  preserved for traceability but are no longer the default reading path. The
  handoff explicitly distinguishes the active Piano50 + curve-v2 integration
  from the unfinished Complete all136 training experiment.
- 2026-08-24: Added index-only Complete all136 experiment tooling without
  recutting or copying the 179 GB tiles: a deterministic source-page-grouped,
  class-complete ~50k validation subset; machine-readable per-class YOLOv9 AP;
  and class-aware weak-class train selection with full per-tile labels plus
  non-target replay. Added separate `pilot_1epoch` and `targeted` launch modes.
  A completed one-epoch pilot can initialize a later stage, but official YOLOv9
  rebuilds optimizer/scheduler and this must not be described as exact resume.
  RTX 3090/5080 are suitable for smoke/subset checks (start at batch 4/2 for
  1024); full 2.329M-tile epochs remain H100 work.
- 2026-08-24: Audited the completed 2.914M-tile Complete all136 dataset and the
  H100 training cost. The observed YOLOv9-E batch-12 throughput is about 0.557
  seconds/batch, or about 30 hours per 2.329M-tile train epoch; 30 epochs would
  exceed 37 days before validation. Added a read-only chunk-statistics audit
  tool, documented the `shift` edge redundancy and exact duplicate-label scan,
  and changed the continued-training default to two full epochs, patience zero,
  per-epoch snapshots, plotting disabled, and a Complete-specific 0.1-epoch
  warmup. Exact mid-epoch resume is explicitly not supported: upstream YOLOv9
  does not checkpoint batch position, AMP scaler, RNG/sampler/worker or prefetch
  state. The server retry1 run must be retained as benchmark evidence.
- 2026-08-23: Added bounded parallel DeepScores Complete conversion at the shard
  level with a backward-compatible one-worker default, fail-closed child-process
  handling, best-effort child termination on interruption, and deterministic
  single-process final aggregation. Added a separate
  `COMPLETE_CONVERSION_WORKERS` server setting, documented a Berlioz starting
  point of eight workers, and preserved all existing resume fingerprints so
  completed and interrupted server chunks continue without a rebuild.
- 2026-08-23: Moved Complete master readiness invalidation ahead of shard
  planning, so a planning-time fingerprint mismatch cannot leave a stale
  successful `validation_report.json` visible to server preflight. Added a
  regression test that first builds a successful tiny master dataset and then
  verifies mismatch refusal also removes the old readiness report.
- 2026-08-20: Hardened Git ignores for model checkpoints and root-level data/run
  directories. Added fail-closed Complete chunk resume fingerprints covering
  source metadata, mapping/converter hashes, and conversion parameters,
  including interrupted partial chunks; explicit overwrite is required after a
  recipe change. Documented that official Complete test shards currently serve
  as YOLO validation and therefore are not an untouched reported test set.
- 2026-08-20: Added a laptop/GPT entry document that identifies the canonical
  files to read, the no-BPSD Dense-to-Complete workflow, required external data
  and weights, Git clone commands, validation boundaries, and a ready-to-paste
  handoff prompt. Added private all136/BPS environment filenames to `.gitignore`.
- 2026-08-18: Recorded the teacher's decision to exclude BPSD from the active
  all136 training stage. Hardened the Complete server workflow by separating
  smoke and full output directories, checking the expected 103/26 source
  shards before full conversion, selecting the correct dataset by training
  mode, adding a CPU-only Slurm conversion job and server environment checker,
  and documenting the Dense-best-to-Complete 30-epoch procedure. Local Bash
  syntax checks passed; an actual server/H100 smoke remains pending.
- 2026-08-13: Added OMR25 semantic normalization for expressive all136 classes
  while intentionally leaving note/rest/stem/beam/staff classes unsupported in
  the articulation stage. Added environment-selectable OMR dataset/piece-list
  roots so isolated MusicXML trials do not mutate the normal project inputs.
  Completed and reparsed a one-page Op. 7 hybrid MusicXML smoke; all136
  structural detections were gated out and therefore did not duplicate the
  legacy note/rhythm pipeline.
- 2026-08-13: Ran the completed Dense all136 checkpoint on the same fixed 20
  BPSD/string pages used for the earlier Piano50 and curve-v2 comparisons.
  Generated a portable six-view gallery with per-class colors, legends, source
  images, and merged JSON. At confidence 0.25 the raw cross-domain review has
  11,870 BPSD and 17,068 string-page boxes. These dense counts and confidence
  values are not accuracy because the 20 pages lack matching ground truth.
- 2026-08-13: The local RTX 3090 Dense all136 run completed all 30 epochs. The
  selected `best.pt` is the final epoch by mAP@0.5:0.95 (0.91764); final
  precision/recall/mAP@0.5 were 0.97496/0.94792/0.96553. Curves remained stable
  and training losses continued decreasing. Metrics cover only the 110 Dense
  validation classes with instances and must not be presented as BPS accuracy.
- 2026-08-11: Added the teacher-requested all-class DeepScores workflow. Defined
  the canonical target as 136 official DeepScores categories, excluding 72
  MUSCIMA++ compatibility IDs that duplicate names. Converted and fully
  validated Dense all136 into 21,842 1024px tiles with 2,583,651 tile instances;
  audited 458 nonpositive source boxes and found no unassigned valid annotation.
  Added an RTX 3090 launcher with preflight and resume support.
- 2026-08-11: Added a per-shard resumable Complete all136 converter, compact
  manifests, master train/val indexes, label validation, Windows smoke helpers,
  and Linux/H100/Slurm launchers. A real 10+10-page sharded dataset completed a
  one-epoch YOLOv9-E RTX 3090 training, validation, and checkpoint smoke at
  about 18.6 GB reported GPU memory. Actual server smoke is still pending.
- 2026-08-11: Located and inspected the 243-page `BPS-OMRv01` YOLO export,
  froze a 21/5/6-work train/validation/test split, and added reproducible
  converters for 50-class symbols and whole slur/tie curves. Added train-only
  1:1 Dense/curve-v2 replay, strict validation including work-leakage checks,
  conservative fine-tuning hyperparameters, Windows RTX 3090 launchers, and
  Linux/Slurm templates. All four formal BPS datasets passed validation and
  both Windows fine-tune preflights passed; training was intentionally not
  started. Added a locked final-test evaluator so the six test works are not
  used during model selection.
- 2026-08-11: Added a 50-class DeepScores Complete preparation and training
  path. Its smoke run merged 560 source pages and produced 8,395 valid tiles
  with 23,596 instances and zero validation errors; RTX 3090 preflight passed.
  Full Complete conversion remains an explicit long-running comparison step,
  not the default BPS strategy.
- 2026-08-11: Made detector-review galleries portable by copying source pages
  into each gallery's `inputs/` directory and emitting relative image links,
  so moving the result folder to another computer does not break originals.
- 2026-08-10: Tightened OCR direction-word output boxes independently from the
  expanded recognition crop, added a staff-relative page-furniture filter with
  an audit trail, and exposed raw `dynamicS` detections in purple for detector
  review. The same ten-page BPSD batch retained 979 displayed symbol candidates,
  moved 186 pre-threshold outside-region candidates into audit records, exposed
  87 raw `s` detections, and retained all 448 curve candidates. Across the base
  conversion plus parenthesis augmentation, the effective tiled training data
  contains 1,153 `dynamicS` instances versus 16,927 `dynamicF` instances. The
  imbalance and domain shift make target-domain fine-tuning on BPSD dynamic
  tokens preferable to training every unrelated DeepScores class.
- 2026-08-10: Clarified and corrected detector-only batch visualization. The
  batch page is explicitly described as a wrapper around the existing Piano50
  and curve-v2 inference path, not a separate model or training tool. Symbol
  candidates now retain their class colors even when NoteGroup parsing is
  intentionally skipped, curve candidates are green and labeled `detected`,
  and the HTML gallery includes a color legend. These colors indicate raw
  detections only; slur/tie classification and MusicXML eligibility still
  require the full OMR25 note-association path.
- 2026-08-10: Added a reusable batch mode for detector-only piano review by
  factoring `run_page` from the single-page smoke and adding
  `server/batch_integrated_piano_pages.py`. It produces symbol/curve JPEGs and
  JSON for every page, a machine-readable batch summary, and an HTML gallery,
  while reusing both loaded YOLOv9 models. The fixed ten-page BPSD set completed
  in about 37 seconds and produced ten symbol JSON/JPEG pairs plus ten curve
  JSON/JPEG pairs: 1,133 displayed symbol candidates and 448 retained curve
  candidates. Output is at
  `C:\OMR_work\experiments\piano_symbols_json_20260810_10pages`. This mode
  intentionally skips legacy NoteGroup/rhythm analysis; the counts are review
  candidates, not MusicXML acceptance counts or ground-truth accuracy.
- 2026-08-10: Completed a one-page full ensemble-mode conversion of the original
  Beethoven Op. 18 No. 1 string-quartet source at 3/4, using four tracks and the
  current Piano50/curve-v2 detectors. The first uncached run took about 167
  seconds. All 151 displayed articulation candidates were associated, and all
  73 curve relations were both candidate-eligible and represented in the output
  as 68 slurs plus five ties. The four-part merged MusicXML parsed successfully
  with `music21` and contains 547 `<note>` elements, 64 directions, 41 dynamics,
  83 staccatos, one fingering, and two tuplet tags. Unlike the three-page piano
  smoke, this page emitted no zero-beat-bar warning. Visual review still shows
  text/direction false positives in title and instrument-label regions, so the
  result remains a functional comparison smoke rather than accuracy evidence.
- 2026-08-10: Added general piano postprocessing for the three-piece visual
  review set. Dynamic letters now cluster in horizontal reading order; dynamic
  fragments with ordinary word ink on both sides are rejected to audit JSON;
  and raw S debug boxes are displayed only beside a plausible F. Regular runs
  of at least three widely spaced fingering-3 detections on one staff baseline
  are reclassified as `tuplet_3`, correcting the repeated triplet numerals in
  Beethoven Op. 27 No. 2 while leaving isolated fingerings unchanged. Hairpin
  tile fragments are unioned before wedge reclassification, and narrow
  geometry searches beside accepted hairpins recover immediately adjacent
  inverse wedges. The ten-page gallery was regenerated: the false `mp` inside
  `sempre` disappeared, the true adjacent `pp` became one token, the missing
  decrescendo beside a crescendo was recovered, and the reported duplicate
  hairpin became one union box. Raw detections remain available in JSON for
  audit. Focused test totals are now 39 articulation/piano plus 14 curve tests.
- 2026-08-10: Expanded the piano review gallery from 10 to 20 pages with ten
  pages sampled across five additional Beethoven sonatas. The larger sample
  exposed weak 0.40-0.50 arpeggio predictions on rests and straight strokes;
  confirmed vertical wavy arpeggios in the same review set were around 0.95.
  The final arpeggio threshold is now 0.75. The batch HTML now has a sticky,
  collapsible full color legend covering every displayed symbol class.
- 2026-08-10: Completed a full three-page piano conversion of the imported
  `Beethoven_Op027No2-01` score after visually verifying the initial cut-time
  signature as 2/2. The first uncached run took about 455 seconds and produced
  three page MusicXML files plus one merged two-part, piano-braced MusicXML.
  `music21` successfully parsed the merged file. Page-level symbol candidates
  were 56, 80, and 51 (the one page-1 candidate outside the ordinary association
  count was an XML-eligible tuplet); curve relations were 20, 25, and 27, all
  candidate-eligible. The merged XML contains 787 `<note>` elements, 44
  directions, 107 fingerings, one staccato, 34 tuplet tags, and 92 slur endpoint
  tags (46 written slur relations), with no written ties. This remains a smoke
  result rather than accuracy evidence: the runtime warned about five zero-beat
  bars, not every eligible curve had two exported note objects, and visual review
  still shows symbol-class false positives requiring correction. The legacy ONNX
  stages used CPU fallback while both YOLOv9 models used the RTX 3090.
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
