# Piano OMR teacher report plan

Updated: 2026-08-10 (Asia/Taipei)

## One-sentence result

We integrated a Piano50 symbol detector and a one-class curve detector into
OMR25, added structural checks before MusicXML, and exported a real two-staff
piano page; symbol/curve integration works, while the legacy piano rhythm core
still needs validation and correction.

## Recommended presentation (8 slides)

1. **Goal and failure cases** — printed piano score to editable MusicXML; show
   missing staccato/accent/tuplet/hairpin, clipped slur/tie, parentheses, and
   `cresc.` incorrectly labeled as pedal.
2. **Architecture** — OMR25 page/notes, Piano50 local symbols, curve v2 long
   curves, OCR/geometry/endpoint rules, then conservative MusicXML.
3. **Data and training** — Piano50 Dense plus parenthesis augmentation; curve
   v2 merges DeepScores slur/tie into one visual class and uses complete boxes.
4. **Model results with caveat** — Piano50 source mAP@0.5:0.95 = 0.97734
   (epoch 26); curve v2 epoch 29 precision 0.96812, recall 0.94299, mAP@0.5
   0.97980, mAP@0.5:0.95 0.85626. These are source-domain metrics, not BPSD
   piano accuracy.
5. **Before/after evidence** — use the fixed 20-page gallery and compare
   previous clipped/missing curves with curve-v2 complete boxes.
6. **Why postprocessing matters** — confidence is not accuracy. Show `cresc.`
   pedal proposals becoming text, hairpin validation, duplicate suppression,
   and endpoint-based slur/tie decisions.
7. **End-to-end smoke** — show the input, overlay, and XML statistics below.
8. **Limitations and next work** — BPSD ground-truth evaluation, bar/rhythm
   correction, multi-voice and cross-staff handling.

## End-to-end evidence from 2026-08-10

Input:
`C:\OMR_work\experiments\piano50_eval_20260808_20pages\inputs\bpsd\Beethoven_Op010No3-01-08.jpeg`

Outputs:

- Model/postprocess smoke:
  `C:\OMR_work\experiments\omr25_integrated_smoke_20260810\Beethoven_Op010No3-01-08`
- Full MusicXML:
  `C:\OMR_work\25-omr\string_dataset\output\piano_smoke\piano_smoke_1.xml`
- Fixed 20-page comparison:
  `C:\OMR_work\experiments\piano50_curvev2_eval_20260809_20pages\index.html`

Observed results:

- 12 staves (six piano systems) and 336 legacy OMR25 note groups.
- 105 symbol candidates survived thresholds; 85 attached to note groups.
- OCR recognized three distinct `cresc.` candidates. Overlapping pedal
  proposals were removed; two attached `cresc.` words reached MusicXML.
- 62 curves survived postprocessing; 55 got two note endpoints. Only seven
  relations passed the MusicXML safety threshold.
- XML contains two parts under a piano brace, 450 notes, 40 direction elements,
  18 dynamics, 45 staccato marks, 15 fingerings, four tuplet tags, six
  time-modification tags, and 14 slur endpoint tags (seven slurs).
- No tie was written because none passed every tie constraint on this page.
- The legacy stage warned that one 4/4 bar totaled only 7/8. This is an
  integration result, not a finished accuracy claim.

## Suggested spoken conclusion

> The number shown beside a box is only detector confidence. We do not directly
> turn every box into MusicXML. Text, hairpins, tuplets, slurs, and ties must
> pass OCR, geometry, or note-relationship checks. The new modules now run
> inside OMR25 and a real piano page exports as a braced two-staff score. The
> remaining bottleneck is piano-specific rhythm, voices, and cross-staff
> structure. The next evaluation must use BPSD ground truth and measure final
> MusicXML events.

## Questions to expect

- **Is 0.95 beside a box 95% accuracy?** No. Accuracy needs labeled test pages
  and aggregate precision/recall/F1.
- **Why one curve class?** Slur and tie can look identical; endpoint pitch and
  adjacency are more reliable for musical type.
- **Why not accept every curve?** Missing endpoints create invalid MusicXML;
  review-only output is safer than a wrong relation.
- **Does source mAP prove piano performance?** No. The fixed BPSD gallery is
  qualitative until the BPSD annotations are available locally.
- **Can it replace manual correction?** Not yet. The smoke exposes a bar-length
  error and does not solve general multi-voice or cross-staff notation.

