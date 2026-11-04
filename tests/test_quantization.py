import pytest
import torch
from src.quantization import nf4_config

def test_nf4_configuration():
    c=nf4_config(); assert c.load_in_4bit and c.bnb_4bit_quant_type=='nf4'
    assert c.bnb_4bit_use_double_quant and c.bnb_4bit_compute_dtype==torch.float16

@pytest.mark.skipif(not torch.cuda.is_available(),reason='Real NF4 requires a CUDA device; also checked by Kaggle preflight')
def test_actual_nf4_roundtrip():
    import bitsandbytes as bnb
    matrix=torch.randn(128,128,device='cuda',dtype=torch.float16)
    quant,state=bnb.functional.quantize_4bit(matrix,quant_type='nf4',compress_statistics=True)
    dense=bnb.functional.dequantize_4bit(quant,state)
    assert dense.shape==matrix.shape and dense.dtype==matrix.dtype
    assert quant.numel()<matrix.numel()
    assert not torch.equal(dense,matrix)
