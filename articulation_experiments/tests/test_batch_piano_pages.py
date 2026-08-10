import tempfile
import unittest
from pathlib import Path

from server.batch_integrated_piano_pages import collect_inputs, write_gallery


class BatchPianoPagesTest(unittest.TestCase):
    def test_collects_supported_images_in_stable_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "b.jpeg").write_bytes(b"x")
            (root / "A.png").write_bytes(b"x")
            (root / "ignore.txt").write_text("x", encoding="utf-8")
            pages = collect_inputs(root, "*", 1)
            self.assertEqual([page.name for page in pages], ["A.png"])

    def test_writes_gallery_with_json_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            source = output / "page.jpeg"
            source.write_bytes(b"x")
            index = write_gallery(
                output,
                [
                    {
                        "image_id": "page",
                        "source": str(source),
                        "staff_count": 10,
                        "symbol_candidate_count": 20,
                        "curve_candidate_count": 5,
                    }
                ],
            )
            text = index.read_text(encoding="utf-8")
            self.assertIn("symbols/page.articulations.json", text)
            self.assertIn("curves/page.slurs_ties.json", text)


if __name__ == "__main__":
    unittest.main()
