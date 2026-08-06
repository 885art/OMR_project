from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from omr.curve_postprocess import merge_curve_fragments
from omr.hairpin import validate_yolo_hairpins
from omr.parentheses import suppress_parentheses
from omr.text_directions import normalize_direction_word
from omr.tuplet import associate_tuplet_candidates


class FakeStaff:
    left = 0
    right = 500
    ys = (80, 90, 100, 110, 120)

    def get_yOne_float(self):
        return 10.0


class FakeGroup:
    def __init__(self, x: int):
        self.boundingBox = (x, 95, x + 10, 105)
        self.noteBoxes = [self.boundingBox]
        self.tuplets = []


class PianoPostprocessTest(unittest.TestCase):
    def test_parenthesis_view_erases_bowed_components(self):
        image = Image.new("RGB", (160, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.arc((35, 20, 55, 80), 70, 290, fill="black", width=2)
        draw.arc((105, 20, 125, 80), 250, 110, fill="black", width=2)
        cleaned, records = suppress_parentheses(image, min_height=20, max_height=80)
        self.assertGreaterEqual(len(records), 1)
        self.assertNotEqual(image.tobytes(), cleaned.tobytes())

    def test_hairpin_geometry_accepts_wedge_and_rejects_staff(self):
        wedge = Image.new("RGB", (240, 100), "white")
        draw = ImageDraw.Draw(wedge)
        draw.line((20, 30, 210, 48), fill="black", width=2)
        draw.line((20, 70, 210, 52), fill="black", width=2)
        prediction = {
            "raw_class_name": "dynamicDiminuendoHairpin",
            "class_id": 39,
            "bbox_xyxy": [15, 25, 215, 75],
            "confidence": 0.8,
        }
        accepted, rejected = validate_yolo_hairpins(wedge, [prediction])
        self.assertEqual(len(accepted), 1)
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["class_name"], "diminuendo")

        staff = Image.new("RGB", (240, 100), "white")
        staff_draw = ImageDraw.Draw(staff)
        for y in (30, 40, 50, 60, 70):
            staff_draw.line((10, y, 225, y), fill="black", width=2)
        accepted, rejected = validate_yolo_hairpins(staff, [prediction])
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["decision"], "reject")

    def test_curve_fragments_merge_only_when_aligned(self):
        merged = merge_curve_fragments(
            [
                {"raw_class_name": "slur", "bbox_xyxy": [10, 20, 80, 40], "confidence": 0.8},
                {"raw_class_name": "slur", "bbox_xyxy": [76, 22, 140, 42], "confidence": 0.7},
                {"raw_class_name": "slur", "bbox_xyxy": [200, 80, 260, 100], "confidence": 0.9},
            ]
        )
        self.assertEqual(len(merged), 2)
        combined = next(item for item in merged if item.get("fragment_count") == 2)
        self.assertEqual(combined["bbox_xyxy"], [10.0, 20.0, 140.0, 42.0])

    def test_curve_validation_rejects_multiple_staff_lines(self):
        from omr.curve_postprocess import validate_curve_candidates

        image = Image.new("L", (240, 80), 255)
        draw = ImageDraw.Draw(image)
        draw.line((10, 30, 230, 30), fill=0, width=2)
        draw.line((10, 40, 230, 40), fill=0, width=2)
        accepted, rejected = validate_curve_candidates(
            image,
            [{"bbox_xyxy": [5, 20, 235, 50], "confidence": 0.9, "raw_class_name": "slur"}],
        )
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["decision_reason"], "multiple_straight_staff_lines_in_curve_bbox")

    def test_direction_dictionary_tolerates_ocr_noise(self):
        self.assertEqual(normalize_direction_word("(cresc.)")["direction_type"], "crescendo")
        self.assertEqual(normalize_direction_word("decresc.p.")["direction_type"], "diminuendo")
        self.assertIsNone(normalize_direction_word("allegro"))

    def test_tuplet_three_gets_conservative_span(self):
        groups = [FakeGroup(x) for x in (90, 120, 150)]
        document = {
            "candidates": [
                {
                    "class_name": "tuplet_3",
                    "bbox_xyxy": [118, 65, 128, 78],
                    "confidence": 0.9,
                }
            ]
        }
        result = associate_tuplet_candidates(document, groups, [FakeStaff()])
        self.assertEqual(result["xml_eligible_count"], 1)
        self.assertEqual(result["relations"][0]["note_group_ids"], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
