"""Startup checks that keep a new machine from opening a broken symptom model."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.ensure_runtime import scipy_requirement, symptom_model_ready


class RuntimeSetupTests(unittest.TestCase):
    def test_scipy_pin_follows_python_version(self):
        self.assertEqual(scipy_requirement((3, 9)), 'scipy==1.13.1')
        self.assertEqual(scipy_requirement((3, 10, 11)), 'scipy==1.13.1')
        self.assertEqual(scipy_requirement((3, 11, 0)), 'scipy==1.17.1')
        self.assertEqual(scipy_requirement((3, 12, 16)), 'scipy==1.17.1')

    def test_missing_weights_need_a_build(self):
        with TemporaryDirectory() as folder:
            self.assertFalse(symptom_model_ready(Path(folder), 'abc'))

    def test_empty_weight_file_needs_a_build(self):
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / 'transformer').mkdir()
            (directory / 'transformer' / 'model.safetensors').write_bytes(b'')
            (directory / 'baseline.joblib').write_bytes(b'model')
            (directory / 'metrics.json').write_text(json.dumps({'data_fingerprint': 'abc'}))
            self.assertFalse(symptom_model_ready(directory, 'abc'))

    def test_matching_fingerprint_is_ready(self):
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / 'transformer').mkdir()
            (directory / 'transformer' / 'model.safetensors').write_bytes(b'weights')
            (directory / 'baseline.joblib').write_bytes(b'model')
            (directory / 'metrics.json').write_text(json.dumps({'data_fingerprint': 'abc'}))
            self.assertTrue(symptom_model_ready(directory, 'abc'))
            self.assertFalse(symptom_model_ready(directory, 'other'))
