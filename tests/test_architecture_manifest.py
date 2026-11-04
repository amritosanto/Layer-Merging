import json

import pytest
import torch

from src.qwen3_adapter import inspect_model
from src.utils import atomic_json


def test_quantized_architecture_manifest_roundtrip(tiny, tmp_path):
    model = tiny[0]
    # The real Transformers NF4 loader adds this private field. It survives
    # dequantized_copy, even after quantization_config has been removed.
    model.config._pre_quantization_dtype = torch.float16
    model.config.dtype_metadata = {'nested': [torch.float32, torch.bfloat16]}
    manifest = inspect_model(model, strict=False)
    path = tmp_path / 'architecture.json'
    atomic_json(manifest, path)
    saved = json.loads(path.read_text(encoding='utf-8'))
    assert saved == manifest
    assert saved['config']['_pre_quantization_dtype'] == 'float16'
    assert saved['config']['dtype_metadata']['nested'] == ['float32', 'bfloat16']
    assert model.config._pre_quantization_dtype is torch.float16
    assert model.config.dtype_metadata['nested'][0] is torch.float32


def test_unquantized_architecture_preserves_config(tiny):
    model = tiny[0]
    manifest = inspect_model(model, strict=False)
    assert manifest['config'] == json.loads(json.dumps(model.config.to_dict()))
    json.dumps(manifest, allow_nan=False)


def test_architecture_does_not_hide_unknown_metadata(tiny):
    model = tiny[0]
    model.config.unsupported_metadata = object()
    with pytest.raises(TypeError, match='not JSON serializable'):
        inspect_model(model, strict=False)
