from __future__ import annotations

import unittest

from articulation_experiments.dataset.generate_deepscores_all_mapping import (
    build_mapping,
)


class DeepScoresAllMappingTest(unittest.TestCase):
    def test_uses_only_canonical_deepscores_ids(self) -> None:
        categories = {
            str(index): {
                "name": f"class{index}",
                "annotation_set": "deepscores",
            }
            for index in range(1, 137)
        }
        categories[137] = {
            "name": "class1",
            "annotation_set": "muscima++",
        }
        categories["75"]["name"] = "articTenutoAbove"
        categories["76"]["name"] = "articTenutoBelow"
        categories["73"]["name"] = "articStaccatoAbove"
        categories["94"]["name"] = "dynamicP"
        categories["110"]["name"] = "arpeggiato"
        categories["113"]["name"] = "tuplet3"
        categories["115"]["name"] = "fingering0"
        categories["121"]["name"] = "slur"
        categories["125"]["name"] = "dynamicCrescendoHairpin"

        mapping = build_mapping({"categories": categories})

        self.assertEqual(136, len(mapping["yolo_names"]))
        self.assertEqual(0, mapping["deepscores_to_yolo"]["1"])
        self.assertEqual(135, mapping["deepscores_to_yolo"]["136"])
        self.assertNotIn("137", mapping["deepscores_to_yolo"])
        self.assertEqual("tenuto", mapping["classes"][74]["normalized_semantic_class"])
        self.assertEqual("above", mapping["classes"][74]["side"])
        self.assertEqual("staccato", mapping["classes"][72]["normalized_semantic_class"])
        self.assertEqual("dynamic_letter_p", mapping["classes"][93]["normalized_semantic_class"])
        self.assertEqual("arpeggio", mapping["classes"][109]["normalized_semantic_class"])
        self.assertEqual("tuplet_3", mapping["classes"][112]["normalized_semantic_class"])
        self.assertEqual("fingering_0", mapping["classes"][114]["normalized_semantic_class"])
        self.assertEqual("slur", mapping["classes"][120]["normalized_semantic_class"])
        self.assertEqual("class25", mapping["classes"][24]["normalized_semantic_class"])
        self.assertEqual("crescendo", mapping["classes"][124]["normalized_semantic_class"])


if __name__ == "__main__":
    unittest.main()
