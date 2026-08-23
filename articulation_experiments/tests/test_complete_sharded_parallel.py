import json
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from articulation_experiments.dataset.convert_deepscores_complete_sharded import (
    main,
    parse_args,
)


FAKE_CONVERTER = r"""
import argparse
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--splits", required=True)
parser.add_argument("--train-json")
parser.add_argument("--val-json")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--overwrite", action="store_true")
args, _ = parser.parse_known_args()

split = args.splits
source = Path(args.train_json or args.val_json)
config = json.loads(source.read_text(encoding="utf-8"))
chunk = Path(args.output_dir)
chunk.mkdir(parents=True, exist_ok=True)
coordination = source.parent / "fake_coordination"
coordination.mkdir(exist_ok=True)
started = coordination / f"{source.stem}.started"
started.write_text("started\n", encoding="utf-8")

log_path = source.parent / "fake_converter.log"
with log_path.open("a", encoding="utf-8") as log:
    log.write(f"START {split} {config['shard_id']} resume={args.resume}\n")

barrier = int(config.get("barrier", 1))
deadline = time.monotonic() + 5.0
while len(list(coordination.glob("*.started"))) < barrier:
    if time.monotonic() >= deadline:
        print("fake concurrency barrier timed out", file=sys.stderr)
        raise SystemExit(9)
    time.sleep(0.02)

if config.get("fail"):
    raise SystemExit(7)
if config.get("fail_once"):
    sentinel = source.parent / f"{source.stem}.failed_once"
    if not sentinel.exists():
        sentinel.write_text("failed\n", encoding="utf-8")
        raise SystemExit(8)

time.sleep(float(config.get("delay", 0)))
image_dir = chunk / "images" / split
label_dir = chunk / "labels" / split
image_dir.mkdir(parents=True, exist_ok=True)
label_dir.mkdir(parents=True, exist_ok=True)
stem = f"{split}_{config['shard_id']:03d}"
(image_dir / f"{stem}.png").write_bytes(b"fake-png")
(label_dir / f"{stem}.txt").write_text(
    "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
)
(chunk / "dataset.yaml").write_text("names: {}\n", encoding="utf-8")
statistics = {
    "splits": {
        split: {
            "tile_count": 1,
            "tile_instance_count": 1,
            "source_image_count": 1,
            "source_target_instance_count": 1,
            "unassigned_annotation_count": 0,
            "dropped_bbox_count": 0,
            "target_annotations_missing_from_image_ann_ids": 0,
            "source_filenames": [f"source_{split}_{config['shard_id']:03d}.png"],
        }
    }
}
(chunk / "statistics.json").write_text(
    json.dumps(statistics), encoding="utf-8"
)
with log_path.open("a", encoding="utf-8") as log:
    log.write(f"END {split} {config['shard_id']}\n")
"""


class CompleteShardedParallelTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.complete = self.root / "complete"
        self.output = self.root / "output"
        (self.complete / "images").mkdir(parents=True)
        self.mapping = self.root / "mapping.json"
        self.mapping.write_text(
            json.dumps({"yolo_names": [f"class_{index}" for index in range(136)]}),
            encoding="utf-8",
        )
        self.converter = self.root / "fake_converter.py"
        self.converter.write_text(
            textwrap.dedent(FAKE_CONVERTER),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def add_shard(self, split, shard_id, **configuration):
        suffix = "train" if split == "train" else "test"
        path = self.complete / f"deepscores-complete-{shard_id}_{suffix}.json"
        path.write_text(
            json.dumps({"shard_id": shard_id, **configuration}),
            encoding="utf-8",
        )
        return path

    def arguments(self, *extra):
        return [
            "--complete-root",
            str(self.complete),
            "--output-dir",
            str(self.output),
            "--class-mapping",
            str(self.mapping),
            "--converter",
            str(self.converter),
            *extra,
        ]

    def converter_log(self):
        path = self.complete / "fake_converter.log"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    def test_workers_one_is_default_and_sequential(self):
        self.add_shard("train", 0)
        self.add_shard("train", 1)
        self.add_shard("val", 0)
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(self.arguments()), 0)
        self.assertIn("Complete shard conversion workers: 1", output.getvalue())
        self.assertEqual(
            self.converter_log(),
            [
                "START train 0 resume=False",
                "END train 0",
                "START train 1 resume=False",
                "END train 1",
                "START val 0 resume=False",
                "END val 0",
            ],
        )

    def test_parallel_shards_and_deterministic_aggregation(self):
        self.add_shard("train", 0, barrier=2, delay=0.2)
        self.add_shard("train", 1, barrier=2)
        self.add_shard("val", 0)
        self.assertEqual(main(self.arguments("--workers", "2")), 0)
        statistics = json.loads(
            (self.output / "statistics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [(item["split"], item["shard_id"]) for item in statistics["shards"]],
            [("train", 0), ("train", 1), ("val", 0)],
        )
        train_index = (self.output / "train.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertTrue(train_index[0].endswith("train_000.png"))
        self.assertTrue(train_index[1].endswith("train_001.png"))

    def test_completed_chunks_are_reused_with_parallel_workers(self):
        self.add_shard("train", 0)
        self.add_shard("val", 0)
        self.assertEqual(main(self.arguments("--workers", "2")), 0)
        first_log = self.converter_log()
        self.assertEqual(
            main(self.arguments("--workers", "2", "--resume")),
            0,
        )
        self.assertEqual(self.converter_log(), first_log)

    def test_parallel_resume_still_rejects_fingerprint_mismatch(self):
        self.add_shard("train", 0)
        self.add_shard("val", 0)
        self.assertEqual(main(self.arguments("--workers", "2")), 0)
        first_log = self.converter_log()
        mapping = json.loads(self.mapping.read_text(encoding="utf-8"))
        mapping["test_revision"] = 2
        self.mapping.write_text(json.dumps(mapping), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "conversion_fingerprint"):
            main(self.arguments("--workers", "2", "--resume"))
        self.assertEqual(self.converter_log(), first_log)

    def test_incomplete_chunk_resumes(self):
        self.add_shard("train", 0, fail_once=True)
        self.add_shard("val", 0)
        with self.assertRaisesRegex(RuntimeError, "split=train, shard_id=0"):
            main(self.arguments("--workers", "1"))
        self.assertTrue(
            (self.output / "chunks" / "train_000" / "chunk_conversion.json").is_file()
        )
        self.assertFalse(
            (self.output / "chunks" / "train_000" / "chunk_complete.json").exists()
        )

        self.assertEqual(main(self.arguments("--workers", "2", "--resume")), 0)
        self.assertIn("START train 0 resume=True", self.converter_log())

    def test_parallel_failure_does_not_publish_master_dataset(self):
        self.add_shard("train", 0, delay=0.1)
        self.add_shard("train", 1, fail=True)
        self.add_shard("val", 0)
        self.output.mkdir(parents=True)
        (self.output / "validation_report.json").write_text(
            '{"passed": true}\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(RuntimeError, "split=train, shard_id=1"):
            main(self.arguments("--workers", "2", "--resume"))
        self.assertFalse((self.output / "validation_report.json").exists())
        self.assertFalse((self.output / "dataset.yaml").exists())
        self.assertFalse(
            (self.output / "chunks" / "train_000" / "chunk_complete.json").exists()
        )

    def test_smoke_limits_still_work_with_workers(self):
        self.add_shard("train", 0)
        self.add_shard("train", 1)
        self.add_shard("val", 0)
        self.add_shard("val", 1)
        self.assertEqual(
            main(
                self.arguments(
                    "--workers",
                    "2",
                    "--max-shards-per-split",
                    "1",
                    "--max-images-per-shard",
                    "10",
                )
            ),
            0,
        )
        chunks = sorted(path.name for path in (self.output / "chunks").iterdir())
        self.assertEqual(chunks, ["train_000", "val_000"])

    def test_workers_must_be_positive(self):
        with self.assertRaises(SystemExit):
            parse_args(self.arguments("--workers", "0"))


if __name__ == "__main__":
    unittest.main()
