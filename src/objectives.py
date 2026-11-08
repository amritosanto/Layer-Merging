import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .qwen3_adapter import teacher_states,layer_context,run_layer
from .merger import correction_penalty
from .pair_selection import valid_plan

def output_objectives(student,teacher,ids,student_head,teacher_head,cfg,memory_checkpoint=True):
    sh=student[:,:-1].reshape(-1,student.shape[-1]); th=teacher[:,:-1].reshape(-1,teacher.shape[-1])
    labels=ids[:,1:].reshape(-1); terms=[]; temperature=cfg.temperature
    def chunk(x,y,label):
        with torch.no_grad():
            tl=teacher_head(y).float(); tp=F.log_softmax(tl/temperature,dim=-1)
            tc=F.cross_entropy(tl,label,reduction='sum')
        sl=student_head(x).float(); sp=F.log_softmax(sl/temperature,dim=-1)
        kd=F.kl_div(sp,tp,log_target=True,reduction='sum')*temperature**2
        sc=F.cross_entropy(sl,label,reduction='sum')
        return torch.stack((kd,sc,tc))
    for start in range(0,len(labels),cfg.logit_chunk_tokens):
        args=(sh[start:start+cfg.logit_chunk_tokens],th[start:start+cfg.logit_chunk_tokens],labels[start:start+cfg.logit_chunk_tokens])
        terms.append(checkpoint(chunk,*args,use_reentrant=False) if memory_checkpoint else chunk(*args))
    return torch.stack(terms).sum(0)/len(labels)

def local_hidden_distribution_kl(student,teacher,temperature):
    """KL across hidden feature coordinates, averaged over tokens.

    Hidden vectors are not vocabulary distributions: this intentionally creates
    *feature* distributions via per-token RMS normalization and softmax.
    Normalization removes raw activation scale; separate hidden MSE protects it.
    All softmax/KL operations run in FP32. Teacher is always detached.
    """
    s=student.float()
    t=teacher.detach().float()
    s=s*torch.rsqrt(s.square().mean(dim=-1,keepdim=True)+1e-6)
    t=t*torch.rsqrt(t.square().mean(dim=-1,keepdim=True)+1e-6)
    log_s=F.log_softmax(s/temperature,dim=-1)
    with torch.no_grad():
        log_t=F.log_softmax(t/temperature,dim=-1)
    return F.kl_div(log_s,log_t,log_target=True,reduction='none').sum(dim=-1).mean()*temperature**2


def training_loss(teacher,source,merger,ids,pairs,cfg,memory_checkpoint=True):
    pairs=valid_plan(pairs,len(source.model.layers)); targets=teacher_states(teacher,ids)
    sources=[dict(layer.named_parameters()) for layer in source.model.layers]
    generated={}
    for p in pairs:
        def generate(p=p):return merger(sources[p],sources[p+1],p)
        generated[p]=checkpoint(generate,use_reentrant=False) if memory_checkpoint else generate()
    penalties=[correction_penalty(generated[p],sources[p],sources[p+1]) for p in pairs]
    h=source.model.embed_tokens(ids); context=layer_context(source,h); hidden=[]; local_kls=[]; i=0
    while i<len(sources):
        merged=i in generated; block=source.model.layers[i+1 if merged else i]
        params=generated.get(i)
        def run(x,block=block,params=params):return run_layer(block,x,context,params)
        h=checkpoint(run,h,use_reentrant=False) if memory_checkpoint and (merged or h.requires_grad) else run(h)
        if merged:
            target=targets[i+2]
            hidden.append((h.float()-target.float()).square().mean()/target.float().square().mean().clamp_min(1e-8))
            if cfg.local_kl_weight:
                local_kls.append(local_hidden_distribution_kl(h,target,cfg.local_kl_temperature))
        i+=2 if merged else 1
    student_final=source.model.norm(h)
    with torch.no_grad():teacher_final=teacher.model.norm(targets[-1])
    kd,sc,tc=output_objectives(student_final,teacher_final,ids,source.lm_head,teacher.lm_head,cfg,memory_checkpoint)
    hidden=torch.stack(hidden).mean(); correction=torch.stack(penalties).mean()
    local_kl=torch.stack(local_kls).mean() if local_kls else hidden.new_zeros(())
    loss=(hidden+cfg.local_kl_weight*local_kl+cfg.kd_weight*kd+
          cfg.correction_weight*correction+cfg.ce_weight*sc)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite four-term merger loss')
    return loss,dict(loss=loss.detach(),hidden_loss=hidden.detach(),
                     local_hidden_kl=local_kl.detach(),distillation_loss=kd.detach(),
                     correction_loss=correction.detach(),student_ce=sc.detach(),teacher_ce=tc.detach(),ce_gap=(sc-tc).detach())
