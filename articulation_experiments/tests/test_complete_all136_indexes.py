import json
import tempfile
import unittest
from pathlib import Path

import yaml

from articulation_experiments.dataset.build_complete_all136_indexes import (
    build_targeted_subset,
    build_validation_subset,
    parse_args,
)


class CompleteAll136IndexesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "full"
        self.dataset.mkdir()
        self.names = ["good", "rare", "bad"]
        train = [
            self.tile("train", 0, "pageA", 1, 0, ["0 0.5 0.5 0.1 0.1", "2 0.6 0.6 0.1 0.1"]),
            self.tile("train", 0, "pageA", 1, 1, ["2 0.5 0.5 0.1 0.1"]),
            self.tile("train", 0, "pageB", 2, 0, ["0 0.5 0.5 0.1 0.1"]),
            self.tile("train", 0, "pageC", 3, 0, ["1 0.5 0.5 0.1 0.1"]),
            self.tile("train", 0, "pageD", 4, 0, []),
        ]
        val = [
            self.tile("val", 0, "pageV1", 11, 0, ["0 0.5 0.5 0.1 0.1"]),
            self.tile("val", 0, "pageV1", 11, 1, ["2 0.5 0.5 0.1 0.1"]),
            self.tile("val", 0, "pageV2", 12, 0, ["1 0.5 0.5 0.1 0.1"]),
            self.tile("val", 0, "pageV3", 13, 0, ["0 0.5 0.5 0.1 0.1"]),
        ]
        (self.dataset / "train.txt").write_text(
            "".join(f"{path.as_posix()}\n" for path in train), encoding="utf-8"
        )
        (self.dataset / "val.txt").write_text(
            "".join(f"{path.as_posix()}\n" for path in val), encoding="utf-8"
        )
        (self.dataset / "dataset.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": self.dataset.as_posix(),
                    "train": "train.txt",
                    "val": "val.txt",
                    "names": {index: name for index, name in enumerate(self.names)},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def tile(self, split, shard, page, image_id, offset, labels):
        chunk = self.dataset / "chunks" / f"{split}_{shard:03d}"
        image_dir = chunk / "images" / split
        label_dir = chunk / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{page}__id{image_id}__x{offset:04d}_y0000"
        image = image_dir / f"{stem}.png"
        image.write_bytes(b"png")
        (label_dir / f"{stem}.txt").write_text(
            "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
        )
        return image.resolve()

    def test_validation_subset_is_page_grouped_and_class_complete(self):
        output = self.root / "val_subset"
        args = parse_args(
            [
                "validation",
                "--dataset-root",
                str(self.dataset),
                "--output-dir",
                str(output),
                "--max-tiles",
                "2",
                "--min-pages-per-class",
                "1",
            ]
        )
        statistics = build_validation_subset(args)
        lines = (output / "val.txt").read_text(encoding="utf-8").splitlines()
        # Mandatory class coverage may exceed max-tiles, and both pageV1 tiles stay together.
        self.assertGreaterEqual(len(lines), 3)
        self.assertEqual(statistics["selected"]["class_statistics"]["1"]["tile_count"], 1)
        self.assertTrue(any("pageV1__id11__x0000" in line for line in lines))
        self.assertTrue(any("pageV1__id11__x0001" in line for line in lines))

    def test_targeted_subset_keeps_all_labels_and_adds_replay(self):
        baseline = self.root / "baseline.json"
        baseline.write_text(
            json.dumps(
                {
                    "per_class": [
                        {"class_id": 0, "map50_95": 0.95},
                        {"class_id": 1, "map50_95": 0.90},
                        {"class_id": 2, "map50_95": 0.20},
                    ]
                }
            ),
            encoding="utf-8",
        )
        output = self.root / "targeted"
        args = parse_args(
            [
                "targeted",
                "--dataset-root",
                str(self.dataset),
                "--output-dir",
                str(output),
                "--baseline-report",
                str(baseline),
                "--maximum-map",
                "0.5",
                "--max-target-tiles",
                "2",
                "--min-tiles-per-class",
                "1",
                "--replay-ratio",
                "0.5",
            ]
        )
        statistics = build_targeted_subset(args)
        self.assertEqual(statistics["target_classes"], [{"class_id": 2, "name": "bad"}])
        self.assertEqual(statistics["target_tile_count"], 2)
        self.assertEqual(statistics["replay_tile_count"], 1)
        self.assertFalse(statistics["labels_rewritten"])
        selected_paths = [
            Path(line)
            for line in (output / "train.txt").read_text(encoding="utf-8").splitlines()
        ]
        mixed = next(path for path in selected_paths if "pageA__id1__x0000" in path.name)
        mixed_label = Path(str(mixed).replace("images", "labels")).with_suffix(".txt")
        self.assertIn("0 0.5 0.5 0.1 0.1", mixed_label.read_text(encoding="utf-8"))

    def test_targeted_default_uses_documented_weak_class_threshold(self):
        args = parse_args(
            [
                "targeted",
                "--dataset-root",
                str(self.dataset),
                "--output-dir",
                str(self.root / "unused"),
                "--target-class",
                "bad",
            ]
        )
        self.assertEqual(args.maximum_map, 0.50)


if __name__ == "__main__":
    unittest.main()
