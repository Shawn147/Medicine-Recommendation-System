"""Compact weight files restore the values inference expects."""
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.ml.compact_weights import (
    dump_compact_estimator, load_compact_state, restore_estimator_precision, save_compact_state,
)


class CompactWeightTests(unittest.TestCase):
    def test_matrices_round_trip_close_to_the_original(self):
        state = {
            'weight': torch.linspace(-1, 1, 64).reshape(4, 16),
            'bias': torch.tensor([0.25, -0.5]),
        }
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'model.safetensors'
            save_compact_state(state, path)
            restored = load_compact_state(path)
        self.assertLess((restored['weight'] - state['weight']).abs().max().item(), 0.08)
        self.assertLess((restored['bias'] - state['bias']).abs().max().item(), 1e-3)

    def test_float16_coefficients_predict_like_the_original(self):
        features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        labels = np.array(['a', 'a', 'b', 'b'])
        model = LogisticRegression(random_state=0).fit(features, labels)
        expected = model.predict(features)
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'model.joblib'
            dump_compact_estimator(model, path)
            import joblib
            loaded = restore_estimator_precision(joblib.load(path))
        self.assertEqual(loaded.coef_.dtype, np.float64)
        self.assertEqual(list(loaded.predict(features)), list(expected))
