from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from omr.curve_postprocess import collapse_curve_duplicates, merge_curve_fragments
from omr.hairpin import validate_yolo_hairpins
from omr.parentheses import suppress_parentheses
from omr.articulation import filter_outside_music_region_candidates
from omr.text_directions import normalize_direction_word, recognize_pedal_word_candidates
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

    def test_curve_validation_keeps_staff_crossing_curve_for_endpoint_review(self):
        from omr.curve_postprocess import validate_curve_candidates

        image = Image.new("L", (240, 80), 255)
        draw = ImageDraw.Draw(image)
        draw.line((10, 30, 230, 30), fill=0, width=2)
        draw.line((10, 40, 230, 40), fill=0, width=2)
        accepted, rejected = validate_curve_candidates(
            image,
            [{"bbox_xyxy": [5, 20, 235, 50], "confidence": 0.9, "raw_class_name": "slur"}],
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(
            accepted[0]["decision_reason"],
            "staff_crossing_curve_requires_note_endpoints",
        )

    def test_full_curve_duplicate_collapse_does_not_union_boxes(self):
        kept, suppressed = collapse_curve_duplicates(
            [
                {"bbox_xyxy": [10, 20, 210, 50], "confidence": 0.9},
                {"bbox_xyxy": [15, 22, 205, 49], "confidence": 0.8},
                {"bbox_xyxy": [30, 55, 190, 75], "confidence": 0.7},
            ]
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(suppressed), 1)
        self.assertEqual(kept[0]["bbox_xyxy"], [10, 20, 210, 50])

    def test_direction_dictionary_tolerates_ocr_noise(self):
        self.assertEqual(normalize_direction_word("(cresc.)")["direction_type"], "crescendo")
        self.assertEqual(normalize_direction_word("decresc.p.")["direction_type"], "diminuendo")
        self.assertIsNone(normalize_direction_word("allegro"))

    def test_wide_pedal_box_is_reclassified_as_crescendo(self):
        class FakeReader:
            def recognize(self, *_args, **_kwargs):
                return [([0, 0, 1, 1], "cresc.", 0.95)]

        image = Image.new("RGB", (240, 100), "white")
        candidates = [
            {
                "class_name": "pedal_stop",
                "bbox_xyxy": [20, 30, 150, 50],
                "confidence": 0.88,
            }
        ]
        with patch("omr.text_directions._reader", return_value=FakeReader()):
            directions, audits = recognize_pedal_word_candidates(
                image, candidates, model_dir="unused", gpu=False
            )
        self.assertEqual(directions[0]["direction_type"], "crescendo")
        self.assertTrue(candidates[0]["ocr_reclassified"])
        self.assertEqual(audits[0]["decision"], "recognized")

    def test_direction_output_uses_tight_ink_box_not_expanded_ocr_crop(self):
        class FakeReader:
            def recognize(self, *_args, **_kwargs):
                return [([0, 0, 1, 1], "cresc.", 0.95)]

        image = Image.new("RGB", (300, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.text((55, 35), "cresc.", fill="black")
        candidates = [
            {
                "class_name": "pedal_stop",
                "bbox_xyxy": [52, 31, 78, 48],
                "confidence": 0.70,
            }
        ]
        with patch("omr.text_directions._reader", return_value=FakeReader()):
            directions, audits = recognize_pedal_word_candidates(
                image, candidates, model_dir="unused", gpu=False
            )
        tight = directions[0]["bbox_xyxy"]
        expanded = audits[0]["expanded_bbox_xyxy"]
        self.assertGreater(tight[0], expanded[0])
        self.assertLess(tight[2], expanded[2])
        self.assertEqual(directions[0]["bbox_xyxy"], audits[0]["tight_bbox_xyxy"])

    def test_music_region_filter_removes_page_furniture_and_keeps_staff_symbol(self):
        document = {
            "candidates": [
                {
                    "class_name": "fermata",
                    "bbox_xyxy": [20, 5, 35, 18],
                    "confidence": 0.98,
                },
                {
                    "class_name": "staccato",
                    "bbox_xyxy": [120, 72, 126, 78],
                    "confidence": 0.92,
                },
            ]
        }
        filter_outside_music_region_candidates(document, [FakeStaff()])
        self.assertEqual([item["class_name"] for item in document["candidates"]], ["staccato"])
        self.assertEqual(document["music_region_filter"]["rejected_count"], 1)
        self.assertEqual(
            document["outside_music_region_candidates"][0]["music_region_reason"],
            "too_far_from_staff",
        )

    def test_unresolved_word_shaped_pedal_is_suppressed(self):
        class FakeReader:
            def recognize(self, *_args, **_kwargs):
                return []

        image = Image.new("RGB", (240, 100), "white")
        candidates = [
            {
                "class_name": "pedal_stop",
                "bbox_xyxy": [20, 30, 150, 50],
                "confidence": 0.91,
            }
        ]
        with patch("omr.text_directions._reader", return_value=FakeReader()):
            directions, audits = recognize_pedal_word_candidates(
                image, candidates, model_dir="unused", gpu=False
            )
        self.assertEqual(directions, [])
        self.assertTrue(candidates[0]["ocr_suppressed"])
        self.assertEqual(audits[0]["decision"], "suppressed_unresolved_pedal")

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
