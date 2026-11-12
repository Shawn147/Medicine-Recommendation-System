"""Store model weights in a smaller form and restore them for inference."""
import json

import joblib
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


GROUP = 16


def save_compact_state(state, path):
    """Pack large matrices as 4-bit groups and keep small tensors in float16."""
    packed, kinds = {}, {}
    for name, value in state.items():
        value = value.detach().cpu().contiguous()
        if value.is_floating_point() and value.ndim >= 2 and value.shape[-1] % GROUP == 0:
            value = value.float()
            grouped = value.reshape(*value.shape[:-1], -1, GROUP)
            scale = grouped.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7
            quantized = (grouped / scale).round().clamp(-8, 7).to(torch.int8)
            low = quantized[..., 0::2] & 0x0F
            high = quantized[..., 1::2] & 0x0F
            packed[name] = (low | (high << 4)).to(torch.uint8).contiguous()
            packed[name + '__scale'] = scale.squeeze(-1).half().contiguous()
            kinds[name] = f'g4:{GROUP}'
        elif value.is_floating_point():
            packed[name] = value.half()
            kinds[name] = 'f16'
        else:
            packed[name] = value
            kinds[name] = 'raw'
    save_file(packed, str(path), metadata={'quant': json.dumps(kinds)})


def load_compact_state(path):
    with safe_open(str(path), framework='pt') as handle:
        metadata = handle.metadata() or {}
    tensors = load_file(str(path))
    kinds = json.loads(metadata['quant']) if metadata.get('quant') else None
    if kinds is None:
        return {name: value.float() if value.is_floating_point() else value for name, value in tensors.items()}
    restored = {}
    for name, kind in kinds.items():
        value = tensors[name]
        if kind == 'row':
            scale = tensors[name + '__scale'].float().reshape(value.shape[0], *([1] * (value.ndim - 1)))
            restored[name] = value.float() * scale
        elif kind.startswith('g4:'):
            low = (value & 0x0F).to(torch.int8)
            high = ((value >> 4) & 0x0F).to(torch.int8)
            low = torch.where(low >= 8, low - 16, low)
            high = torch.where(high >= 8, high - 16, high)
            quantized = torch.stack((low, high), dim=-1).flatten(-2)
            scale = tensors[name + '__scale'].float().unsqueeze(-1)
            grouped = quantized.float() * scale
            restored[name] = grouped.reshape(*grouped.shape[:-2], -1)
        elif kind == 'f16':
            restored[name] = value.float()
        else:
            restored[name] = value
    return restored


def _classifier(estimator):
    return estimator.named_steps['clf'] if hasattr(estimator, 'named_steps') else estimator


def dump_compact_estimator(estimator, path):
    """Write classifier coefficients as int8 without changing the live model."""
    classifier = _classifier(estimator)
    coefficient, intercept = classifier.coef_, classifier.intercept_
    scale = np.maximum(np.max(np.abs(coefficient), axis=1, keepdims=True), 1e-8) / 127
    classifier.coef_ = np.clip(np.round(coefficient / scale), -127, 127).astype(np.int8)
    classifier.coef_scale_ = scale.astype(np.float32)
    classifier.intercept_ = intercept.astype(np.float16)
    try:
        joblib.dump(estimator, path, compress=3)
    finally:
        classifier.coef_ = coefficient
        classifier.intercept_ = intercept
        if hasattr(classifier, 'coef_scale_'):
            del classifier.coef_scale_


def restore_estimator_precision(estimator):
    classifier = _classifier(estimator)
    if getattr(classifier, 'coef_', None) is None:
        return estimator
    if classifier.coef_.dtype == np.int8 and hasattr(classifier, 'coef_scale_'):
        scale = np.asarray(classifier.coef_scale_, dtype=np.float64)
        classifier.coef_ = np.ascontiguousarray(classifier.coef_, dtype=np.float64) * scale
        del classifier.coef_scale_
    elif classifier.coef_.dtype != np.float64:
        classifier.coef_ = np.ascontiguousarray(classifier.coef_, dtype=np.float64)
    if getattr(classifier, 'intercept_', None) is not None and classifier.intercept_.dtype != np.float64:
        classifier.intercept_ = np.ascontiguousarray(classifier.intercept_, dtype=np.float64)
    return estimator
