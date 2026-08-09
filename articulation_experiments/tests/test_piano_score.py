from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from music21 import note, stream

from omr.piano_score import apply_piano_staff_group


class PianoScoreTest(unittest.TestCase):
    def test_two_parts_export_as_braced_piano_group(self):
        score = stream.Score(
            [
                stream.Part([stream.Measure([note.Note("C5")])]),
                stream.Part([stream.Measure([note.Note("C3")])]),
            ]
        )
        self.assertTrue(apply_piano_staff_group(score))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "piano.musicxml"
            score.write("musicxml", fp=str(path))
            xml = path.read_text(encoding="utf-8")
        self.assertIn('<group-symbol>brace</group-symbol>', xml)
        self.assertIn('<group-barline>yes</group-barline>', xml)

    def test_non_two_part_score_is_not_modified(self):
        score = stream.Score([stream.Part([note.Note("C4")])])
        self.assertFalse(apply_piano_staff_group(score))


if __name__ == "__main__":
    unittest.main()
