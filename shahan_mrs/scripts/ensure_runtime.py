"""Prepare a fresh machine before the app opens.

SciPy can be left half-installed after an interrupted pip run. The symptom
model weights are local files and are absent from a copy that omitted them.
Both failures surface later as "unable to review your symptoms."
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Same split as requirements.txt. SciPy 1.17+ has no Python 3.9 or 3.10 wheels.
SCIPY_REQUIREMENT = {
    False: 'scipy==1.13.1',
    True: 'scipy==1.17.1',
}
# The saved evaluation was produced with two epochs.
TRAIN_EPOCHS = 2


def scipy_requirement(version):
    return SCIPY_REQUIREMENT[version >= (3, 11)]


def symptom_model_ready(directory, current_fingerprint):
    weights = directory / 'transformer' / 'model.safetensors'
    baseline = directory / 'baseline.joblib'
    metrics_path = directory / 'metrics.json'
    if not _nonempty(weights) or not _nonempty(baseline) or not metrics_path.is_file():
        return False
    try:
        metrics = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return metrics.get('data_fingerprint') == current_fingerprint


def _nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def _scipy_imports():
    return subprocess.call(
        [sys.executable, '-c', 'from scipy._lib import _pep440'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ) == 0


def repair_scipy():
    if _scipy_imports():
        return
    requirement = scipy_requirement(sys.version_info)
    print('SciPy is incomplete. Reinstalling ' + requirement + '.', flush=True)
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install',
        '--force-reinstall', '--no-deps', '--only-binary=:all:', requirement,
    ])
    if not _scipy_imports():
        raise SystemExit('SciPy still cannot be imported. Symptom review needs a complete SciPy wheel.')


def build_symptom_model():
    print('Saved symptom model is missing or does not match the data.', flush=True)
    print('Downloading the public MiniLM encoder and training the local model.', flush=True)
    print('The first run needs a network connection and takes several minutes.', flush=True)
    if not (ROOT / 'artifacts/pretrained/minilm/model.safetensors').is_file():
        subprocess.check_call([sys.executable, str(ROOT / 'scripts/download_model.py')])
    subprocess.check_call([
        sys.executable, str(ROOT / 'scripts/train_chatbot.py'), '--epochs', str(TRAIN_EPOCHS),
    ])


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    repair_scipy()
    from src.data.loading import ARTIFACT_DIR, fingerprint
    if symptom_model_ready(ARTIFACT_DIR, fingerprint()):
        return
    build_symptom_model()
    if not symptom_model_ready(ARTIFACT_DIR, fingerprint()):
        raise SystemExit('The symptom model is still not ready. Symptom review cannot start.')


if __name__ == '__main__':
    main()
