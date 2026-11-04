import torch
import copy
from transformers import AutoModelForCausalLM
from src.model_export import variant,compress_in_place

def test_dequantized_source_exports_without_nf4_metadata(tiny,tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from src.quantization import dequantized_copy
    from src.utils import module_hash

    source,merger,_,ids,_=tiny
    source=copy.deepcopy(source).half()
    source.config.quantization_config={'load_in_4bit':True}
    source.config._pre_quantization_dtype=torch.float16
    # CPU stand-in for NF4 storage: exercise the real copy/export path without
    # requiring CUDA. Real NF4 kernels are tested separately on GPU.
    class PackedParameter(torch.nn.Parameter):
        pass
    projection=source.model.layers[0].self_attn.q_proj
    projection.weight=PackedParameter(projection.weight.detach(),requires_grad=False)
    projection.weight.quant_state=object()
    backend=SimpleNamespace(nn=SimpleNamespace(Params4bit=PackedParameter),
                            functional=SimpleNamespace(dequantize_4bit=lambda data,state:data))
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules,'bitsandbytes',backend)
        dense,manifest=dequantized_copy(source,'cpu')
    assert manifest['nf4_matrices']==1
    assert module_hash(dense)==module_hash(source)
    # The original quantized model must keep its runtime metadata.
    assert source.config._pre_quantization_dtype is torch.float16
    assert source.config.quantization_config=={'load_in_4bit':True}
    with variant(dense,merger,(0,),method='average'):
        dense.save_pretrained(tmp_path,safe_serialization=True)
        expected_hash=module_hash(dense)
        with torch.no_grad(): expected=dense(ids).logits
    assert not hasattr(dense.config,'quantization_config')
    assert not hasattr(dense.config,'_pre_quantization_dtype')
    loaded=AutoModelForCausalLM.from_pretrained(tmp_path,torch_dtype=torch.float16,attn_implementation='sdpa')
    assert module_hash(loaded)==expected_hash
    # Reload reconstructs non-persistent RoPE buffers; allow FP16 rounding,
    # while the parameter hash above must still match exactly.
    with torch.no_grad(): torch.testing.assert_close(loaded(ids).logits,expected,rtol=1e-2,atol=5e-4)
    assert loaded.generate(ids[:1],max_new_tokens=2,do_sample=False).shape[1]>ids.shape[1]

def test_export_reload_and_cache_generation(tiny,tmp_path):
    model,merger,_,ids,_=tiny
    with variant(model,merger,(0,3,6,9)):
        model.save_pretrained(tmp_path,safe_serialization=True)
        expected=model(ids).logits
    loaded=AutoModelForCausalLM.from_pretrained(tmp_path)
    assert len(loaded.model.layers)==8
    torch.testing.assert_close(loaded(ids).logits,expected)
    generated=loaded.generate(ids[:1],max_new_tokens=3,do_sample=False)
    assert generated.shape[1]>ids.shape[1]
    assert [b.self_attn.layer_idx for b in loaded.model.layers]==list(range(8))

def test_reconstructed_controls_match_variants(tiny):
    model,merger,_,ids,_=tiny; plan=(0,3,6)
    generated={p:merger(dict(model.model.layers[p].named_parameters()),dict(model.model.layers[p+1].named_parameters()),p) for p in plan}
    for method in ('learned','average','drop_first','drop_second'):
        control=compress_in_place(copy.deepcopy(model),plan,method,generated)
        with variant(model,merger,plan,method):expected=model(ids).logits
        torch.testing.assert_close(control(ids).logits,expected,rtol=0,atol=0)
