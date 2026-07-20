import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from music21 import note, stream

from omr.slur_tie import (
    add_slurs_to_stream,
    apply_ties_to_music21,
    associate_curve_candidates,
    detect_curve_candidates,
    finalize_ties,
    register_slur_endpoints,
    register_tie_endpoints,
)


class FakeStaff:
    def __init__(self):
        self.left = 20
        self.right = 480
        self.ys = [90, 100, 110, 120, 130]


class FakeStem:
    def __init__(self, pitch):
        self.pitchSoprano = pitch
        self.isOrnament = False


class FakeGroup:
    def __init__(self, x, pitch):
        self.boundingBox = [x - 5, 92, x + 5, 108]
        self.noteBoxes = [self.boundingBox]
        self.noteStemList = [] if pitch is None else [FakeStem(pitch)]
        self.slur_ties = []


def candidate():
    return {
        "candidate_id": 0,
        "bbox_xyxy": [95, 55, 305, 98],
        "left_endpoint": [100, 97],
        "right_endpoint": [300, 97],
        "curve_direction": "above",
        "staff_index": 0,
        "confidence": 0.9,
        "features": {},
    }


class SlurTieTests(unittest.TestCase):
    def test_detects_synthetic_thin_curve(self):
        image = np.full((220, 500), 255, dtype=np.uint8)
        cv2.ellipse(image, (250, 80), (80, 15), 0, 180, 360, 0, 2)
        detected, _ = detect_curve_candidates(image, [FakeStaff()])
        self.assertGreaterEqual(len(detected), 1)
        self.assertEqual(detected[0]["curve_direction"], "above")

    def test_same_pitch_is_tie_and_different_pitch_is_slur(self):
        groups = [FakeGroup(100, 5), FakeGroup(300, 5)]
        result = associate_curve_candidates([candidate()], groups, [FakeStaff()], "tie")
        self.assertEqual(result["relations"][0]["predicted_type"], "tie")
        self.assertTrue(result["relations"][0]["xml_eligible"])

        groups = [FakeGroup(100, 5), FakeGroup(300, 6)]
        result = associate_curve_candidates([candidate()], groups, [FakeStaff()], "slur")
        self.assertEqual(result["relations"][0]["predicted_type"], "slur")

    def test_unknown_pitch_is_not_attached_for_xml(self):
        groups = [FakeGroup(100, None), FakeGroup(300, 6)]
        result = associate_curve_candidates([candidate()], groups, [FakeStaff()], "unknown")
        self.assertEqual(result["relations"][0]["predicted_type"], "unknown_curve")
        self.assertEqual(groups[0].slur_ties, [])
        self.assertEqual(groups[1].slur_ties, [])

    def test_musicxml_contains_tie_and_tied(self):
        start_group = FakeGroup(100, 5)
        stop_group = FakeGroup(300, 5)
        associate_curve_candidates([candidate()], [start_group, stop_group], [FakeStaff()], "tie")
        start = apply_ties_to_music21(note.Note("C4"), start_group)
        stop = apply_ties_to_music21(note.Note("C4"), stop_group)
        score = stream.Score(stream.Part([stream.Measure([start, stop])]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tie.musicxml"
            score.write("musicxml", fp=str(path))
            xml = path.read_text(encoding="utf-8")
        self.assertIn('<tie type="start"', xml)
        self.assertIn('<tie type="stop"', xml)
        self.assertIn('<tied type="start"', xml)
        self.assertIn('<tied type="stop"', xml)

    def test_incomplete_tie_is_removed_before_export(self):
        start_group = FakeGroup(100, 5)
        stop_group = FakeGroup(300, 5)
        associate_curve_candidates([candidate()], [start_group, stop_group], [FakeStaff()], "tie")
        start = apply_ties_to_music21(note.Note("C4"), start_group)
        registry = {}
        register_tie_endpoints(registry, start, start_group)
        self.assertEqual(finalize_ties(registry), 0)
        self.assertIsNone(start.tie)

    def test_shared_tie_start_keeps_only_highest_confidence_pair(self):
        start, first_stop, second_stop = note.Note("C4"), note.Note("C4"), note.Note("C4")
        registry = {
            "low": {"confidence": 0.5, "start": start, "stop": first_stop},
            "high": {"confidence": 0.9, "start": start, "stop": second_stop},
        }
        self.assertEqual(finalize_ties(registry), 1)
        self.assertEqual(start.tie.type, "start")
        self.assertIsNone(first_stop.tie)
        self.assertEqual(second_stop.tie.type, "stop")

    def test_musicxml_contains_numbered_slur_start_and_stop(self):
        start_group = FakeGroup(100, 5)
        stop_group = FakeGroup(300, 6)
        associate_curve_candidates([candidate()], [start_group, stop_group], [FakeStaff()], "slur")
        start, stop = note.Note("C4"), note.Note("D4")
        part = stream.Part([stream.Measure([start, stop])])
        score = stream.Score(part)
        registry = {}
        register_slur_endpoints(registry, start, start_group)
        register_slur_endpoints(registry, stop, stop_group)
        self.assertEqual(add_slurs_to_stream(score, registry), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slur.musicxml"
            score.write("musicxml", fp=str(path))
            xml = path.read_text(encoding="utf-8")
        self.assertIn('<slur number="1" placement="above" type="start"', xml)
        self.assertIn('<slur number="1" type="stop"', xml)


if __name__ == "__main__":
    unittest.main()
