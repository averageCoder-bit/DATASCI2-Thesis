"""Local synthetic contract tests for new import-safe modeling utilities.

Run from repository root:
  python -m unittest discover -s tests -p test_verified_source.py -v
Does NOT test restored checkpoint inference; use Airflow for that.
"""
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from src.modeling.verified.calibration import (
    fit_gmm_candidates, negative_log_density, select_validation_threshold,
)
from src.modeling.verified.evaluation import align_proxy_labels, evaluate_heldout
from src.modeling.verified.reconstruction import reconstruction_mae
from src.modeling.verified.config import GMM_COMPONENTS


class VerifiedSourceTests(unittest.TestCase):
    def test_threshold_parity_with_exhaustive_original_for_60_cases(self):
        for seed in range(60):
            rng = np.random.default_rng(seed)
            scores = np.round(rng.normal(size=36), 2)  # many ties
            labels = np.array([0, 1] * 18, dtype=np.int8)
            rng.shuffle(labels)
            chosen = select_validation_threshold(scores, labels)
            best = None
            for candidate in np.unique(scores):
                ba = balanced_accuracy_score(labels, (scores >= candidate).astype(int))
                if best is None or ba > best[1]:
                    best = (float(candidate), float(ba))
            self.assertEqual(chosen["threshold"], best[0])
            self.assertAlmostEqual(chosen["balanced_accuracy"], best[1], places=12)

    def test_gmm_bic_selection_and_score_direction(self):
        rng = np.random.default_rng(4)
        data = np.abs(np.r_[rng.normal(.05,.01,60),rng.normal(.3,.03,60)])
        model, rows = fit_gmm_candidates(data)
        self.assertEqual([r["n_components"] for r in rows], list(GMM_COMPONENTS))
        self.assertEqual(model.n_components, min(rows,key=lambda r:r["bic"])["n_components"])
        actual = negative_log_density(model, np.array([.04, .2, .4]))
        self.assertTrue(np.allclose(actual, -model.score_samples(np.array([.04,.2,.4]).reshape(-1,1))))

    def test_label_alignment_out_of_order(self):
        table = pd.DataFrame({"sequence_id":[2,0,1],"proxy_anomaly":[1,0,1]})
        actual = align_proxy_labels(table, [0,1,2])
        np.testing.assert_array_equal(actual, [0,1,1])
        with self.assertRaises(ValueError):
            align_proxy_labels(table.iloc[:2], [0,1,2])
        with self.assertRaises(ValueError):
            align_proxy_labels(pd.DataFrame({"sequence_id":[0,0],"proxy_anomaly":[1,0]}),[0,1])

    def test_reconstruction_mae_axes_and_dtype(self):
        x = np.zeros((2,60,7), dtype=np.float32)
        rec = x.copy()
        rec[0] = 0.5
        np.testing.assert_array_equal(reconstruction_mae(x,rec),np.array([.5,0],dtype=np.float32))
        self.assertEqual(reconstruction_mae(x,rec).dtype, np.float32)

    def test_test_metrics_and_threshold_inclusive(self):
        scores = np.array([-.1,.8,0.0,.5])
        y = np.array([0,1,1,0])
        result = evaluate_heldout(scores,y,0.0)
        self.assertEqual((result["TP"],result["TN"],result["FP"],result["FN"]),(2,1,1,0))
        self.assertEqual(result["predicted_positive"],3)

    def test_validate_nan_and_mismatch(self):
        with self.assertRaises(ValueError):
            select_validation_threshold([1.,np.nan],[1,0])
        with self.assertRaises(ValueError):
            select_validation_threshold([1.,2.],[1,1])
        with self.assertRaises(ValueError):
            negative_log_density(None, [-.1,.1])


if __name__ == "__main__":
    unittest.main()
