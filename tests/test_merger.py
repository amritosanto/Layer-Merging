import torch
from src.model_export import variant
from src.qwen3_adapter import teacher_states,inspect_model
from src.merger import correction_penalty

def test_zero_anchor_and_all_levels(tiny):
    model,merger,cfg,ids,_=tiny
    inspect_model(model,strict=False)
    assert len(merger.blocks)==6
    with torch.no_grad():
        a,b=[dict(x.named_parameters()) for x in model.model.layers[:2]]
        generated=merger(a,b,0)
        for key,value in b.items():torch.testing.assert_close(generated[key],value,rtol=0,atol=0)
        assert correction_penalty(generated,a,b)==0
        for k in range(1,6):
            plan=tuple(range(0,k*2,2))
            with variant(model,merger,plan):
                assert len(model.model.layers)==12-k
                x=model(ids).logits
            with variant(model,merger,plan,'drop_first'):y=model(ids).logits
            torch.testing.assert_close(x,y,rtol=0,atol=0)
        assert len(model.model.layers)==12

def test_manual_rope_gqa_forward(tiny):
    model,_,_,ids,_=tiny
    with torch.no_grad():
        direct=model.model(ids,use_cache=False).last_hidden_state
        manual=model.model.norm(teacher_states(model,ids)[-1])
    torch.testing.assert_close(direct,manual,rtol=1e-5,atol=1e-6)
