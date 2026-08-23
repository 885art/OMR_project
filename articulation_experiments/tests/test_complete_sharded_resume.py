import argparse
import json
import tempfile
import unittest
from pathlib import Path

from articulation_experiments.dataset.convert_deepscores_complete_sharded import (
    build_chunk_conversion_record,
    build_conversion_fingerprint,
    determine_chunk_action,
)


def converter_args(**overrides):
    values = {
        "tile_size": 1024,
        "overlap": 256,
        "minimum_intersection_ratio": 0.6,
        "negative_ratio": 0.05,
        "png_compress_level": 1,
        "max_shards_per_split": 1,
        "max_images_per_shard": 10,
        "workers": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class CompleteShardedResumeTest(unittest.TestCase):
    def test_fingerprint_changes_with_mapping_converter_and_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = root / "mapping.json"
            converter = root / "converter.py"
            mapping.write_text('{"version": 1}', encoding="utf-8")
            converter.write_text("print('v1')\n", encoding="utf-8")

            original = build_conversion_fingerprint(
                converter_args(), mapping, converter
            )
            changed_parameter = build_conversion_fingerprint(
                converter_args(overlap=128), mapping, converter
            )
            self.assertNotEqual(original, changed_parameter)

            parallel_workers = build_conversion_fingerprint(
                converter_args(workers=8), mapping, converter
            )
            self.assertEqual(original, parallel_workers)

            mapping.write_text('{"version": 2}', encoding="utf-8")
            changed_mapping = build_conversion_fingerprint(
                converter_args(), mapping, converter
            )
            self.assertNotEqual(original, changed_mapping)

            mapping.write_text('{"version": 1}', encoding="utf-8")
            converter.write_text("print('v2')\n", encoding="utf-8")
            changed_converter = build_conversion_fingerprint(
                converter_args(), mapping, converter
            )
            self.assertNotEqual(original, changed_converter)

    def test_resume_requires_matching_progress_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            chunk = Path(temporary) / "chunk"
            chunk.mkdir()
            source = {"path": "source.json", "size": 10, "mtime_ns": 20}
            conversion = {"schema_version": 1, "mapping_sha256": "abc"}
            expected = build_chunk_conversion_record(
                source, conversion, "train", 0
            )

            self.assertEqual(
                determine_chunk_action(
                    chunk,
                    expected,
                    resume=True,
                    overwrite_chunks=False,
                ),
                "convert",
            )

            (chunk / "images").mkdir()
            with self.assertRaisesRegex(RuntimeError, "no conversion fingerprint"):
                determine_chunk_action(
                    chunk,
                    expected,
                    resume=True,
                    overwrite_chunks=False,
                )

            (chunk / "chunk_conversion.json").write_text(
                json.dumps(expected), encoding="utf-8"
            )
            self.assertEqual(
                determine_chunk_action(
                    chunk,
                    expected,
                    resume=True,
                    overwrite_chunks=False,
                ),
                "resume",
            )

            changed = build_chunk_conversion_record(
                source,
                {"schema_version": 1, "mapping_sha256": "changed"},
                "train",
                0,
            )
            with self.assertRaisesRegex(RuntimeError, "conversion_fingerprint"):
                determine_chunk_action(
                    chunk,
                    changed,
                    resume=True,
                    overwrite_chunks=False,
                )
            self.assertEqual(
                determine_chunk_action(
                    chunk,
                    changed,
                    resume=True,
                    overwrite_chunks=True,
                ),
                "overwrite",
            )

    def test_completed_and_recoverable_chunks_are_distinguished(self):
        with tempfile.TemporaryDirectory() as temporary:
            chunk = Path(temporary) / "chunk"
            chunk.mkdir()
            expected = build_chunk_conversion_record(
                {"path": "source.json", "size": 10, "mtime_ns": 20},
                {"schema_version": 1, "mapping_sha256": "abc"},
                "val",
                3,
            )
            (chunk / "chunk_conversion.json").write_text(
                json.dumps(expected), encoding="utf-8"
            )
            (chunk / "dataset.yaml").write_text("names: {}\n", encoding="utf-8")
            (chunk / "statistics.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                determine_chunk_action(
                    chunk,
                    expected,
                    resume=True,
                    overwrite_chunks=False,
                ),
                "recover",
            )

            (chunk / "chunk_complete.json").write_text(
                json.dumps(expected), encoding="utf-8"
            )
            self.assertEqual(
                determine_chunk_action(
                    chunk,
                    expected,
                    resume=True,
                    overwrite_chunks=False,
                ),
                "reuse",
            )


if __name__ == "__main__":
    unittest.main()
