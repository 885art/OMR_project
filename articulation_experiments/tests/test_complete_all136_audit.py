import json
import tempfile
import unittest
from pathlib import Path

from articulation_experiments.dataset.audit_complete_all136 import (
    build_audit,
    parse_args,
)


class CompleteAll136AuditTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        (self.dataset / "statistics.json").write_text(
            json.dumps(
                {
                    "configuration": {
                        "tile_size": 1024,
                        "overlap": 256,
                    },
                    "splits": {
                        "train": {
                            "source_image_count": 2,
                            "tile_count": 11,
                            "source_target_instance_count": 10,
                            "tile_instance_count": 18,
                        },
                        "val": {
                            "source_image_count": 1,
                            "tile_count": 5,
                            "source_target_instance_count": 4,
                            "tile_instance_count": 7,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        (self.dataset / "validation_report.json").write_text(
            '{"passed": true}\n', encoding="utf-8"
        )
        self.write_chunk(
            "train",
            0,
            source_image_count=2,
            positive_tile_count=10,
            negative_tile_count=1,
            available_negative_tile_count=2,
            source_target_instance_count=10,
            tile_instance_count=18,
            duplicate_annotation_id_count=6,
            duplicate_extra_assignment_count=8,
            clipped_bbox_count=3,
        )
        self.write_chunk(
            "val",
            0,
            source_image_count=1,
            positive_tile_count=4,
            negative_tile_count=1,
            available_negative_tile_count=3,
            source_target_instance_count=4,
            tile_instance_count=7,
            duplicate_annotation_id_count=2,
            duplicate_extra_assignment_count=3,
            clipped_bbox_count=1,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def write_chunk(self, split, shard_id, **values):
        chunk = self.dataset / "chunks" / f"{split}_{shard_id:03d}"
        chunk.mkdir(parents=True)
        values["tile_count"] = (
            values["positive_tile_count"] + values["negative_tile_count"]
        )
        values["candidate_tile_count"] = (
            values["positive_tile_count"]
            + values["available_negative_tile_count"]
        )
        (chunk / "statistics.json").write_text(
            json.dumps({"splits": {split: values}}), encoding="utf-8"
        )

    def arguments(self, *extra):
        return parse_args(["--dataset-root", str(self.dataset), *extra])

    def test_audit_aggregates_chunk_statistics_and_benchmark(self):
        audit = build_audit(
            self.arguments("--batch-size", "2", "--seconds-per-batch", "1")
        )
        self.assertEqual(audit["totals"]["tile_count"], 16)
        self.assertEqual(audit["totals"]["positive_tile_count"], 14)
        self.assertEqual(audit["totals"]["negative_tile_count"], 2)
        self.assertAlmostEqual(
            audit["splits"]["train"]["tile_instances_per_source_annotation"],
            1.8,
        )
        self.assertEqual(audit["benchmark"]["training_batches_per_epoch"], 6)
        self.assertEqual(audit["benchmark"]["validation_batches"], 2)

    def test_shift_geometry_exposes_large_edge_overlap(self):
        geometry = build_audit(self.arguments())["geometry_example"]
        self.assertEqual(geometry["x_starts"], [0, 768, 936])
        self.assertEqual(geometry["y_starts"], [0, 768, 1536, 1748])
        self.assertEqual(
            geometry["x_adjacent_overlaps"][-1]["overlap_pixels"], 856
        )
        self.assertEqual(
            geometry["y_adjacent_overlaps"][-1]["overlap_pixels"], 812
        )

    def test_master_mismatch_fails_closed(self):
        master = json.loads(
            (self.dataset / "statistics.json").read_text(encoding="utf-8")
        )
        master["splits"]["train"]["tile_count"] = 999
        (self.dataset / "statistics.json").write_text(
            json.dumps(master), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "Master/chunk mismatch"):
            build_audit(self.arguments())

    def test_optional_duplicate_scan_matches_float32_yolo_rows(self):
        label_dir = (
            self.dataset / "chunks" / "train_000" / "labels" / "train"
        )
        label_dir.mkdir(parents=True)
        (label_dir / "duplicate.txt").write_text(
            "0 0.500000001 0.5 0.1 0.1\n"
            "0 0.500000002 0.5 0.1 0.1\n",
            encoding="utf-8",
        )
        audit = build_audit(self.arguments("--scan-label-duplicates"))
        scanned = audit["exact_yolo_label_duplicate_scan"]["train"]
        self.assertEqual(scanned["label_files_scanned"], 1)
        self.assertEqual(scanned["label_files_with_exact_duplicate_rows"], 1)
        self.assertEqual(scanned["exact_duplicate_rows"], 1)


if __name__ == "__main__":
    unittest.main()
