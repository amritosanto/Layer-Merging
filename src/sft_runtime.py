"""SFT batch profiling, diagnostic monitoring and completed-checkpoint recovery."""
import json
import os
import subprocess
import time
from pathlib import Path
import torch
from .utils import atomic_json,digest


def finite_gradients(model):
    flags=[torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    if not flags or not bool(torch.stack(flags).all()):raise FloatingPointError('Nonfinite or missing actual SFT gradients; refusing optimizer update')


def choose_batch(rows,effective):
    valid=[r for r in rows if r['status']=='ok' and effective%r['microbatch']==0]
    if not valid:raise RuntimeError('No stable SFT microbatch; preserve checkpoints and change BOTH variants in a fresh configuration')
    best=max(valid,key=lambda r:(r['tokens_per_second'],-r['peak_bytes'],-r['microbatch']))
    return dict(best,gradient_accumulation=effective//best['microbatch'],effective_batch_size=effective)


def profile_candidates(candidates,probe,effective):
    rows=[]
    for batch in candidates:
        if effective%batch:continue
        try:rows.append(dict(probe(batch),microbatch=batch,status='ok'))
        except torch.cuda.OutOfMemoryError:rows.append(dict(microbatch=batch,status='oom'))
        except FloatingPointError as e:rows.append(dict(microbatch=batch,status='nonfinite',reason=str(e)))
    return dict(selected=choose_batch(rows,effective),candidates=rows)


def autotune(cfg,model,data,pad_token,folder,identity):
    from .sft_training import response_collator
    path=folder/'preflight.json'
    effective=cfg.sft_effective_batch_size or cfg.sft_batch_size*cfg.sft_gradient_accumulation
    if path.exists():
        cached=json.loads(path.read_text())
        if cached.get('identity')!=identity:raise RuntimeError('Stale SFT batch profile')
        return cached['selected']
    longest=sorted(data['examples']+data.get('validation',[]),key=lambda r:len(r['input_ids']),reverse=True)
    def probe(batch):
        samples=longest[:batch]
        if len(samples)!=batch:raise FloatingPointError('Insufficient distinct profiling samples')
        reserve=[];inputs=None;loss=None
        try:
            model.train();model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            reserve=[torch.zeros(p.numel()*2,device=p.device,dtype=torch.float32) for p in model.parameters() if p.requires_grad]
            inputs={k:v.to('cuda:0') for k,v in response_collator(samples,pad_token).items()}
            # One warmup then one measured forward/backward; optimizer weights are never changed.
            durations=[]
            for trial in range(2):
                model.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.float16):loss=model(**inputs).loss
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite actual profiling loss')
                loss.backward();finite_gradients(model);torch.cuda.synchronize()
                durations.append(time.perf_counter()-start);loss=None
            peak=torch.cuda.max_memory_allocated()
            if peak>cfg.peak_memory_fraction*torch.cuda.get_device_properties(0).total_memory:raise torch.cuda.OutOfMemoryError('Insufficient optimizer/activation headroom')
            return dict(tokens_per_second=sum(len(r['input_ids']) for r in samples)/max(durations[-1],1e-9),peak_bytes=peak,
                        seconds=durations[-1],max_tokens=max(len(r['input_ids']) for r in samples))
        finally:
            loss=None;inputs=None;reserve.clear();model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    result=profile_candidates([1,2,4,8] if cfg.sft_autotune_batch else [cfg.sft_batch_size],probe,effective)
    atomic_json(dict(result,identity=identity),path)
    print('SFT batch profile:',result,flush=True)
    return result['selected']


def gpu_sample():
    sample=dict(allocated_bytes=torch.cuda.memory_allocated(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                utilization_percent=None,power_watts=None)
    physical=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
    try:
        result=subprocess.run(['nvidia-smi','-i',physical,'--query-gpu=utilization.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=3)
        util,power=result.stdout.strip().split(',')
        import math
        if not all(math.isfinite(float(v)) for v in (util,power)):raise ValueError('Nonfinite device telemetry')
        sample.update(utilization_percent=float(util),power_watts=float(power))
    except (OSError,ValueError,subprocess.SubprocessError) as error:sample['monitor_error']=str(error)
    return sample


def completed_state(checkpoint,cfg,data,audit):
    """Called only after checkpoint hashes and the run identity have been verified."""
    if not checkpoint:return None
    path=Path(checkpoint);state=json.loads((path/'trainer_state.json').read_text())
    if state.get('epoch',0)<cfg.sft_epochs:return None
    effective=cfg.sft_effective_batch_size or cfg.sft_batch_size*cfg.sft_gradient_accumulation
    import math
    expected_steps=math.ceil(len(data['examples'])/effective)*cfg.sft_epochs
    if state['epoch']!=cfg.sft_epochs or state['global_step']!=expected_steps:raise RuntimeError('Completed SFT checkpoint epoch/step mismatch')
    exposure=json.loads((path/'exposure.json').read_text())
    expected={l:cfg.sft_epochs*sum(r['language']==l for r in data['examples']) for l in cfg.training_languages}
    if exposure['examples']!=expected or exposure['tokens']!={l:cfg.sft_epochs*audit['tokens_per_language'][l] for l in cfg.training_languages}:
        raise RuntimeError('Completed checkpoint exposure mismatch')
    return state,exposure


def restore_adapter(model,path):
    from peft.utils.save_and_load import load_peft_weights,set_peft_model_state_dict
    result=set_peft_model_state_dict(model,load_peft_weights(path,device='cpu'))
    if result.unexpected_keys or any('lora_' in k for k in result.missing_keys):raise RuntimeError('Completed adapter checkpoint mismatch')


def child_environments(environ,single_gpu=False):
    return [dict(environ,CUDA_VISIBLE_DEVICES=str(0 if single_gpu else i)) for i in range(2)]
