import torch
import pytest
from src.weight_patching import pack,unpack

def test_all_qwen_families(tiny):
    model,_,_,_,families=tiny
    params=dict(model.model.layers[0].named_parameters())
    for f in families:
        x=params[f.name]; assert torch.equal(unpack(pack(x,f.patch),f.patch,f.shape),x)
    assert params['self_attn.q_proj.weight'].shape!=(16,16)

def test_invalid_patch():
    with pytest.raises(ValueError):pack(torch.zeros(7,9),(4,4))
