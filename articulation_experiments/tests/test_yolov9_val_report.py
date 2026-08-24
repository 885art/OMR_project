import unittest

from articulation_experiments.train.yolov9_val_report import report_document


class YoloV9ValReportTest(unittest.TestCase):
    def test_report_contains_machine_readable_per_class_ap(self):
        report = report_document(
            ["good", "bad"],
            [0.8, 0.7, 0.75, 0.6, 0.1, 0.2, 0.3],
            [0.9, 0.2],
            [1.0, 2.0, 3.0],
            {"data": "subset.yaml"},
        )
        self.assertEqual(report["aggregate"]["map50_95"], 0.6)
        self.assertEqual(
            report["per_class"][1],
            {"class_id": 1, "name": "bad", "map50_95": 0.2},
        )

    def test_class_count_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            report_document(["a", "b"], [0] * 7, [0.1], [0] * 3, {})


if __name__ == "__main__":
    unittest.main()
