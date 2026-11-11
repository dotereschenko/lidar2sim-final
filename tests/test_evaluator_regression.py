import tempfile
import unittest
from pathlib import Path


class EvaluatorRegressionTests(unittest.TestCase):
    def test_gt_and_all_negative_controls_in_both_modes(self):
        from tools.check_evaluator import run_regression
        with tempfile.TemporaryDirectory() as tmp:
            report=run_regression(Path(tmp)/'result')
            self.assertTrue(report['passed'],report['checks'])
            self.assertTrue(all(report['checks'].values()))
            self.assertGreaterEqual(len(report['checks']),16)
            with self.assertRaises(FileExistsError):
                run_regression(Path(tmp)/'result')
