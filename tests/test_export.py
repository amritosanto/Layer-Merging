import torch
import copy
from transformers import AutoModelForCausalLM
from src.model_export import variant,compress_in_place

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
