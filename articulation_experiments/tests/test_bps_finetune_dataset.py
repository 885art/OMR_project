import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERTER = REPO_ROOT / "articulation_experiments/dataset/convert_bps_yolo_finetune.py"
VALIDATOR = REPO_ROOT / "articulation_experiments/dataset/validate_finetune_yolo_dataset.py"


class BpsFinetuneDatasetTest(unittest.TestCase):
    def make_source(self, root: Path, curve: bool = False) -> tuple[Path, Path]:
        source = root / "source"
        (source / "images").mkdir(parents=True)
        (source / "labels").mkdir()
        (source / "classes.txt").write_text(
            "articStaccatoAbove\nslur\ntie\n", encoding="utf-8"
        )
        works = ["Beethoven_Op001", "Beethoven_Op002", "Beethoven_Op003"]
        for index, work in enumerate(works):
            stem = f"{work}-01-01"
            Image.new("RGB", (640, 640), "white").save(source / "images" / f"{stem}.png")
            if curve:
                rows = ["1 0.5 0.5 0.5 0.08"]
                if index == 0:
                    rows.append("2 0.5 1.25 0.4 0.05")
            else:
                rows = ["0 0.5 0.5 0.04 0.04"]
            (source / "labels" / f"{stem}.txt").write_text(
                "\n".join(rows) + "\n", encoding="utf-8"
            )
        manifest = root / "split.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": "test_split",
                    "splits": {
                        "train": {"expected_page_count": 1, "works": [works[0]]},
                        "val": {"expected_page_count": 1, "works": [works[1]]},
                        "test": {"expected_page_count": 1, "works": [works[2]]},
                    },
                }
            ),
            encoding="utf-8",
        )
        return source, manifest

    def run_converter(self, source: Path, manifest: Path, output: Path, task: str) -> None:
        subprocess.run(
            [
                sys.executable,
                str(CONVERTER),
                "--source-root",
                str(source),
                "--output-dir",
                str(output),
                "--task",
                task,
                "--split-manifest",
                str(manifest),
                "--tile-size",
                "512",
                "--overlap",
                "128",
                "--negative-ratio",
                "0",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_symbol_conversion_is_work_split_and_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, manifest = self.make_source(root)
            output = root / "symbols"
            self.run_converter(source, manifest, output, "symbols")
            subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR),
                    "--dataset-root",
                    str(output),
                    "--expected-classes",
                    "50",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            for split in ("train", "val", "test"):
                self.assertTrue(any((output / "images" / split).iterdir()))
                label = next((output / "labels" / split).glob("*.txt"))
                self.assertTrue(label.read_text(encoding="utf-8").startswith("2 "))

    def test_curve_conversion_drops_fully_outside_box_without_editing_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, manifest = self.make_source(root, curve=True)
            original = (source / "labels" / "Beethoven_Op001-01-01.txt").read_text(
                encoding="utf-8"
            )
            output = root / "curves"
            self.run_converter(source, manifest, output, "curves")
            stats = json.loads((output / "statistics.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["global_counters"]["dropped_outside_or_zero_area"], 1)
            self.assertEqual(
                (source / "labels" / "Beethoven_Op001-01-01.txt").read_text(
                    encoding="utf-8"
                ),
                original,
            )
            subprocess.run(
                [sys.executable, str(VALIDATOR), "--dataset-root", str(output), "--expected-classes", "1"],
                check=True,
                capture_output=True,
                text=True,
            )


if __name__ == "__main__":
    unittest.main()
