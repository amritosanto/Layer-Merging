import copy
import torch
from src.objectives import training_loss,output_objectives
from src.utils import module_hash

def test_gradient_flow_and_frozen_models(tiny):
    teacher,merger,cfg,ids,_=tiny; source=copy.deepcopy(teacher); before=module_hash(teacher)
    loss,_=training_loss(teacher,source,merger,ids,(0,3,6,9),cfg,True)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in merger.dec.parameters())
    assert all(torch.isfinite(p.grad).all() for p in merger.parameters() if p.grad is not None)
    assert all(p.grad is None for m in (teacher,source) for p in m.parameters())
    assert module_hash(teacher)==before and module_hash(source)==before
    optimizer=torch.optim.AdamW(merger.parameters(),lr=1e-4); optimizer.step(); optimizer.zero_grad()
    loss,_=training_loss(teacher,source,merger,ids,(1,4),cfg,False); loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in merger.enc.parameters())

def test_chunked_kl_and_gradient_equivalence(tiny):
    teacher,_,cfg,ids,_=tiny
    h=teacher.model(ids).last_hidden_state.detach(); cfg.logit_chunk_tokens=3
    identity=output_objectives(h,h,ids,teacher.lm_head,teacher.lm_head,cfg,False)
    assert abs(float(identity[0]))<1e-6
    student=(h+torch.randn_like(h)*.1).requires_grad_()
    chunk=output_objectives(student,h,ids,teacher.lm_head,teacher.lm_head,cfg,True)
    cg=torch.autograd.grad(chunk[0],student)[0]
    cfg.logit_chunk_tokens=10000
    full=output_objectives(student,h,ids,teacher.lm_head,teacher.lm_head,cfg,False)
    fg=torch.autograd.grad(full[0],student)[0]
    torch.testing.assert_close(chunk,full,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(cg,fg,rtol=1e-4,atol=1e-6)
