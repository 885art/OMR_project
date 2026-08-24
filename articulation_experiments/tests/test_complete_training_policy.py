import unittest
from pathlib import Path

import yaml


class CompleteTrainingPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[2]

    def test_complete_hyp_only_shortens_warmup(self):
        config_dir = self.repo_root / "articulation_experiments" / "configs"
        base = yaml.safe_load(
            (config_dir / "yolov9_score_hyp.yaml").read_text(encoding="utf-8")
        )
        complete = yaml.safe_load(
            (config_dir / "yolov9_score_complete_finetune_hyp.yaml").read_text(
                encoding="utf-8"
            )
        )
        changed = {
            key for key in base.keys() | complete.keys() if base.get(key) != complete.get(key)
        }
        self.assertEqual(changed, {"warmup_epochs"})
        self.assertEqual(base["warmup_epochs"], 2.0)
        self.assertEqual(complete["warmup_epochs"], 0.1)

    def test_server_defaults_are_short_full_training_and_no_fake_batch_resume(self):
        script = (self.repo_root / "server" / "train_yolov9_all136.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('EPOCHS="${EPOCHS:-2}"', script)
        self.assertIn('PATIENCE="${PATIENCE:-0}"', script)
        self.assertIn('SAVE_PERIOD="${SAVE_PERIOD:-1}"', script)
        self.assertIn('DISABLE_PLOTS="${DISABLE_PLOTS:-1}"', script)
        self.assertIn("--noplots", script)
        self.assertIn("Exact mid-epoch resume is NOT supported", script)
        self.assertNotIn("checkpoint-every", script.casefold())

    def test_pilot_and_targeted_modes_do_not_change_formal_default(self):
        script = (self.repo_root / "server" / "train_yolov9_all136.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('pilot_1epoch) EPOCHS=1; PATIENCE=0;', script)
        self.assertIn('targeted) EPOCHS="${TARGETED_EPOCHS:-1}";', script)
        self.assertIn('EPOCHS="${EPOCHS:-2}"', script)
        self.assertIn("TRAIN_DATASET_ROOT", script)

    def test_index_preparation_uses_separate_complete_variables(self):
        script = (
            self.repo_root / "server" / "prepare_complete_all136_indexes.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("VALIDATION_SUBSET_DATASET", script)
        self.assertIn("TARGETED_DATASET", script)
        self.assertIn("BASELINE_REPORT", script)
        self.assertNotIn("COMPLETE_CONVERSION_WORKERS", script)

    def test_windows_complete_launcher_matches_short_finetune_policy(self):
        script = (
            self.repo_root / "server" / "train_yolov9_complete_all136_3090.bat"
        ).read_text(encoding="utf-8")
        self.assertIn('set "EPOCHS=2"', script)
        self.assertIn('set "PATIENCE=0"', script)
        self.assertIn('if /I "%~1"=="pilot1"', script)
        self.assertIn("yolov9_e_dense_all136_30ep_b4_3090", script)
        self.assertNotIn("yolov9_e_deepscores_dense_all136_30ep_b4_3090", script)
        self.assertIn("yolov9_score_complete_finetune_hyp.yaml", script)
        self.assertIn("--noplots", script)


if __name__ == "__main__":
    unittest.main()
