import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from server.import_bpsd_piano_images import (
    discover_pages,
    import_groups,
    load_template,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class BpsdImporterTest(unittest.TestCase):
    def make_image(self, path: Path, color: int) -> None:
        Image.new("L", (32, 24), color=color).save(path, format="JPEG")

    def test_rejects_noncontiguous_numeric_pages(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            self.make_image(source / "Beethoven_Op001-01-010.jpeg", 10)
            self.make_image(source / "Beethoven_Op001-01-02.jpeg", 20)
            self.make_image(source / "Beethoven_Op001-01-01.jpeg", 30)
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                discover_pages(source)

    def test_imports_png_structure_config_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "pdf_data"
            source.mkdir()
            output.mkdir()
            self.make_image(source / "Beethoven_Op001-01-01.jpeg", 10)
            self.make_image(source / "Beethoven_Op001-01-02.jpeg", 20)
            groups = discover_pages(source)
            template = load_template(REPO_ROOT / "pianoConfigTemplate.json")

            dry_run = import_groups(groups, output, template, (3, 4), False, False)
            self.assertEqual(dry_run["piece_count"], 1)
            self.assertFalse((output / "Beethoven_Op001-01").exists())

            manifest = import_groups(groups, output, template, (3, 4), True, False)
            piece = "Beethoven_Op001-01"
            config_path = output / piece / f"{piece}.json"
            image_path = output / piece / "imgs" / f"{piece}_1" / f"{piece}_1.png"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["numPage"], 2)
            self.assertEqual(config["score_mode"], "piano")
            self.assertEqual(config["tsChange"][0]["time_signature"], [3, 4])
            self.assertEqual(config["_bpsd_import"]["time_signature_status"], "needs_manual_review")
            self.assertTrue(image_path.is_file())
            with Image.open(image_path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (32, 24))
            self.assertEqual(manifest["written_page_count"], 2)
            self.assertTrue((output / "bpsd_import_manifest.json").is_file())
            self.assertTrue((output / "piecesToRun.bpsd.json").is_file())

            config["tsChange"][0]["time_signature"] = [2, 4]
            config_path.write_text(json.dumps(config), encoding="utf-8")
            repeated = import_groups(groups, output, template, (4, 4), True, False)
            preserved = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(repeated["existing_page_count"], 2)
            self.assertEqual(preserved["tsChange"][0]["time_signature"], [2, 4])


if __name__ == "__main__":
    unittest.main()
