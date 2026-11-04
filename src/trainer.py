import gc
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from huggingface_hub import HfApi
from config import VERSION
from .utils import atomic_json,atomic_torch,digest,module_hash,cpu_state,rng_state,restore_rng,seed_all,file_digest
from .quantization import load_training_models,load_model,nf4_config
from .qwen3_adapter import inspect_model,families_from_block,teacher_states
from .weight_patching import make_families,pack,unpack
from .merger import PairWeightMerger
from .objectives import training_loss,output_objectives
from .distributed_training import DualWorkers
from .pair_selection import task_for_batch,checkpoint_plans,compatible_selection
from .wiki40b_loader import prepare_data,balanced_epoch
from .model_export import variant,export_model

def prepare_run(cfg):
    cfg.validate(); out=Path(cfg.output_dir)
    for folder in ('checkpoints','logs','manifests','rankings','model_exports','benchmark_results','charts'):
        (out/folder).mkdir(parents=True,exist_ok=True)
    lock_path=out/'manifests/run.json'; settings=json.loads(json.dumps(asdict(cfg)))
    from .utils import file_digest
    root=Path(__file__).resolve().parents[1]
    implementation=digest({str(p.relative_to(root)):file_digest(p) for p in sorted(list((root/'src').glob('*.py'))+list((root/'benchmarks').glob('*.py'))+[root/'config.py',root/'run.py',root/'src/language_coverage.json'])})
    # Benchmark protocol is locked separately, so full evaluation can reuse pilot-trained weights.
    for key in ('evaluation_mode','pilot_questions','mmmlu_few_shot','include_few_shot','mgsm_few_shot',
                'eval_context','eval_batch_size','max_new_tokens','bootstrap_samples','benchmark_languages','mgsm_max_new_tokens'):
        settings.pop(key)
    if lock_path.exists():
        lock=json.loads(lock_path.read_text())
        if lock['config']!=settings or lock['version']!=VERSION or lock.get('implementation')!=implementation:raise RuntimeError('Incompatible run configuration; choose a fresh output_dir')
        return lock
    api=HfApi()
    revisions={name:api.dataset_info(name).sha for name in ('openai/MMMLU',cfg.include_dataset,'juletxara/mgsm')}
    revisions[cfg.fineweb_dataset]=api.dataset_info(cfg.fineweb_dataset,revision=cfg.fineweb_revision).sha
    revisions[cfg.wiki_dataset]='tfds-wiki40b-1.3.0' if cfg.wiki_backend=='tfds' else api.dataset_info(cfg.wiki_dataset).sha
    lock=dict(version=VERSION,implementation=implementation,config=settings,model_revision=api.model_info(cfg.model_id,revision=cfg.model_revision).sha,
              dataset_revisions=revisions)
    atomic_json(lock,lock_path); return lock

def hardware(cfg):
    needed=1 if cfg.single_gpu_debug else 2
    if torch.cuda.device_count()<needed:raise RuntimeError(f'Need {needed} CUDA GPU(s); enable Kaggle T4 x2. No silent fallback.')
    rows=[]
    for i in range(needed):
        p=torch.cuda.get_device_properties(i)
        rows.append(dict(device=i,name=p.name,total_vram=p.total_memory,compute_capability=[p.major,p.minor]))
    result=dict(gpus=rows,torch=torch.__version__,cuda=torch.version.cuda,quantization=nf4_config().to_dict(),
                dependencies={m:__import__(m).__version__ for m in ('transformers','datasets','accelerate','bitsandbytes')})
    print(json.dumps(result,indent=2)); return result

@torch.no_grad()
def score_tokens(model,ids,cfg):
    device=next(model.parameters()).device; total=0.; count=0
    # Disjoint target windows, one overlapping context token at each boundary.
    for start in range(0,len(ids)-1,cfg.train_seq_len-1):
        batch=torch.tensor(ids[start:start+cfg.train_seq_len],device=device)[None]
        if batch.shape[1]<2:continue
        with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'):
            h=model.model(batch,use_cache=False).last_hidden_state[0]
            for i in range(0,len(h)-1,cfg.logit_chunk_tokens):
                end=min(i+cfg.logit_chunk_tokens,len(h)-1)
                logits=model.lm_head(h[i:end]).float()
                total+=float(F.cross_entropy(logits,batch[0,i+1:end+1],reduction='sum')); count+=end-i
    if count!=len(ids)-1:raise AssertionError('Diagnostic token accounting')
    if not math.isfinite(total):raise FloatingPointError('Nonfinite diagnostic CE')
    return total/count

class AnchoredExperiment:
    def __init__(self,cfg):
        self.cfg=cfg; self.out=Path(cfg.output_dir); self.lock=prepare_run(cfg); seed_all(cfg.seed)
        atomic_json(hardware(cfg),self.out/'manifests/hardware.json')
        self.tokenizer=AutoTokenizer.from_pretrained(cfg.model_id,revision=self.lock['model_revision'])
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        self.teacher,self.source,self.quant_manifest=load_training_models(cfg,self.lock['model_revision'],'cuda:0')
        architecture=inspect_model(self.source); inspect_model(self.teacher)
        self.families=make_families(families_from_block(self.source.model.layers[0]),cfg.matrix_patch,cfg.vector_patch)
        self.merger=PairWeightMerger(cfg,self.families).to('cuda:0')
        architecture.update(families=[dict(name=f.name,shape=f.shape,patch=f.patch,tokens=f.count,kind=f.kind) for f in self.families],
                            total_weight_tokens=2*sum(f.count for f in self.families),merger_parameters=sum(p.numel() for p in self.merger.parameters()))
        atomic_json(architecture,self.out/'manifests/architecture.json')
        qpath=self.out/'manifests/quantization.json'
        if qpath.exists() and json.loads(qpath.read_text())!=self.quant_manifest:raise RuntimeError('Source quantization changed; refusing resume')
        atomic_json(self.quant_manifest,qpath)
        self.data,self.data_manifest=prepare_data(cfg,self.tokenizer,self.lock['dataset_revisions'])
        self.signature=digest(dict(run=self.lock,data=self.data_manifest,quantization=self.quant_manifest))
        self.dual=DualWorkers(self.teacher,self.source,self.merger,cfg)
        self.optimizer=torch.optim.AdamW(self.merger.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
        # Allocate Adam state BEFORE batch tuning so memory measurements include optimizer overhead.
        for p in self.merger.parameters():
            self.optimizer.state[p].update(step=torch.tensor(0.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
        self.scaler=torch.amp.GradScaler('cuda')
        self.state=dict(epoch=0,cursor=0,step=0,seconds=0.,best=None,batch_size=None,last_validation=0.)
        self.logs=[]; self.validation=[]

    def restore(self):
        path=self.out/'checkpoints/latest.pt'
        if not path.exists():path=self.out/'checkpoints/merger_final.pt'
        if not path.exists():return False
        saved=torch.load(path,map_location='cpu',weights_only=False)  # Only locally generated trusted checkpoints.
        if saved['signature']!=self.signature:raise RuntimeError('Checkpoint/config/data/source mismatch (GPT-2 checkpoints are incompatible)')
        self.merger.load_state_dict(saved['merger']); self.optimizer.load_state_dict(saved['optimizer'])
        self.scaler.load_state_dict(saved['scaler']); self.state=saved['state']; self.logs=saved['logs']; self.validation=saved['validation']
        restore_rng(saved['rng']); self.dual.sync(); return True

    def save(self):
        atomic_torch(dict(signature=self.signature,version=VERSION,merger=cpu_state(self.merger),optimizer=self.optimizer.state_dict(),
            scaler=self.scaler.state_dict(),scheduler=None,state=self.state,rng=rng_state(),config=asdict(self.cfg),
            data_manifest=self.data_manifest,quantization=self.quant_manifest,model_revision=self.lock['model_revision'],
            sampler=dict(epoch=self.state['epoch'],cursor=self.state['cursor'],seed=self.cfg.seed),logs=self.logs,validation=self.validation),
            self.out/'checkpoints/latest.pt')
        pd.DataFrame(self.logs).to_csv(self.out/'logs/training.csv',index=False)
        pd.DataFrame(self.validation).to_csv(self.out/'logs/validation.csv',index=False)

    def snapshot(self,name):
        atomic_torch(dict(signature=self.signature,merger=cpu_state(self.merger),state=self.state.copy()),self.out/f'checkpoints/{name}.pt')

    def integrity(self):
        for worker in self.dual.workers:
            assert module_hash(worker.teacher)==self.quant_manifest['teacher_hash']
            assert module_hash(worker.source)==self.quant_manifest['dense_hash']
            assert all(p.grad is None for model in (worker.teacher,worker.source) for p in model.parameters())

    def preflight(self):
        cfg=self.cfg; ids=torch.stack([v for _,v in balanced_epoch(self.data,cfg.train_seq_len,cfg.seed,0)[:2]])[:,:16].to('cuda:0')
        before=module_hash(self.merger)
        for f in self.families:
            x=dict(self.source.model.layers[0].named_parameters())[f.name]
            assert torch.equal(unpack(pack(x,f.patch),f.patch,f.shape),x)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            direct=self.teacher.model(ids,use_cache=False).last_hidden_state
            manual=self.teacher.model.norm(teacher_states(self.teacher,ids)[-1])
            torch.testing.assert_close(direct,manual,rtol=2e-3,atol=2e-3)
            for p in ((0,13,26) if cfg.mode=='smoke' else range(27)):
                a,b=[dict(x.named_parameters()) for x in self.source.model.layers[p:p+2]]
                for key,value in self.merger(a,b,p).items():torch.testing.assert_close(value,b[key].float(),rtol=0,atol=0)
            for k in sorted({1,min(7,cfg.max_merged_pairs),cfg.max_merged_pairs}):
                plan=tuple(range(0,2*k,2))
                with variant(self.source,self.merger,plan):
                    assert len(self.source.model.layers)==28-k
                    a=self.source.model(ids,use_cache=False).last_hidden_state
                with variant(self.source,self.merger,plan,'drop_first'):b=self.source.model(ids,use_cache=False).last_hidden_state
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            identity=output_objectives(direct,direct,ids,self.teacher.lm_head,self.teacher.lm_head,cfg,False)[0]
            assert abs(float(identity))<1e-5
        self.dual.sync(); self.dual.clear()
        loss,_,work=self.dual.loss(ids,(0,4,8,12,16)); loss.backward()
        for worker in self.dual.workers:
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in worker.merger.dec.parameters())
        primary=list(self.merger.dec.parameters())[0]
        expected=primary.grad.clone()
        for worker in self.dual.workers[1:]:expected+=list(worker.merger.dec.parameters())[0].grad.to('cuda:0')
        self.dual.reduce(); torch.testing.assert_close(primary.grad,expected)
        initial=cpu_state(self.merger)
        assert all(torch.isfinite(p.grad).all() for p in self.merger.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(self.merger.parameters(),cfg.clip_grad)
        self.optimizer.step()
        assert module_hash(self.merger)!=before, 'Representative optimizer step made no update'
        self.merger.load_state_dict(initial)
        # Preflight is only run before restore/updates. Reset the intentionally probed Adam state.
        for state in self.optimizer.state.values():
            state['step'].zero_(); state['exp_avg'].zero_(); state['exp_avg_sq'].zero_()
        self.dual.sync(); del initial
        self.dual.clear(); self.integrity(); assert module_hash(self.merger)==before
        atomic_json(dict(passed=True,patches=True,zero_correction=True,teacher_manual_match=True,
                         gradient_flow=True,dual_gradient_sum=True,per_gpu_work=work,identity_kl=True),
                    self.out/'manifests/gpu_preflight.json')

    def tune_batch(self):
        if self.state['batch_size']:return
        windows=balanced_epoch(self.data,self.cfg.train_seq_len,self.cfg.seed,0); rows=[]
        minimum=len(self.dual.workers)
        for size in sorted(set([minimum]+[x for x in (2,4,8,16) if minimum<=x<=self.cfg.max_global_batch_size])):
            if size>len(windows):continue
            try:
                batch=torch.stack([v for _,v in windows[:size]])
                for d in range(len(self.dual.workers)):torch.cuda.reset_peak_memory_stats(d)
                self.dual.sync()
                for trial in range(2):
                    self.dual.clear(); started=time.monotonic()
                    loss,_,work=self.dual.loss(batch,tuple(range(0,2*self.cfg.max_merged_pairs,2))); loss.backward(); self.dual.reduce(); self.dual.synchronize()
                    duration=time.monotonic()-started
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite batch probe')
                    del loss
                fractions=[torch.cuda.max_memory_allocated(d)/torch.cuda.get_device_properties(d).total_memory for d in range(len(work))]
                rows.append(dict(batch=size,status='ok',seconds=duration,tokens_per_second=size*self.cfg.train_seq_len/duration,
                                 peak_fraction=max(fractions),per_gpu_work=str(work)))
            except torch.cuda.OutOfMemoryError:
                rows.append(dict(batch=size,status='oom')); break
            finally:
                self.dual.clear(); gc.collect()
                for d in range(len(self.dual.workers)):
                    with torch.cuda.device(d):torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(self.out/'logs/batch_tuning.csv',index=False)
        safe=[r for r in rows if r['status']=='ok' and r['peak_fraction']<=self.cfg.peak_memory_fraction]
        if not safe:raise RuntimeError('No safe maximum-merge batch. Reduce sequence length in a new run; inspect GPU allocations.')
        self.state['batch_size']=max(safe,key=lambda r:(r['tokens_per_second'],r['batch']))['batch']

    def score(self,pairs=(),method='learned',partition='validation',model=None):
        with torch.autocast('cuda',dtype=torch.float16),variant(model or self.source,self.merger,pairs,method) as m:
            per_lang={lang:score_tokens(m,data[partition],self.cfg) for lang,data in self.data.items()}
        return float(np.mean(list(per_lang.values()))),per_lang

    def validate(self):
        self.merger.eval()
        teacher=self.score(model=self.teacher)[0]
        # One deterministic representative plan, not a sweep over all levels.
        plan=tuple(range(0,2*min(3,self.cfg.max_merged_pairs),2))
        ce=self.score(plan)[0]
        row=dict(step=self.state['step'],epoch=self.state['epoch'],teacher_ce=teacher,
                 validation_ce=ce,selection_ce_gap=ce-teacher)
        self.validation.append(row); self.state['best']=ce
        self.merger.train(); print('Validation:',row,flush=True)

    def train(self):
        resumed=self.restore()
        if (self.out/'manifests/training_complete.json').exists():return
        started=time.monotonic(); base=self.state['seconds']; safe_checkpoint=True
        try:
            if not (self.out/'manifests/gpu_preflight.json').exists():self.preflight()
            self.tune_batch()
            if not self.validation:self.validate()
            self.state['last_validation']=base+time.monotonic()-started
            self.save()  # Persist a boundary before any optimizer mutation.
            while self.state['epoch']<self.cfg.epochs:
                windows=balanced_epoch(self.data,self.cfg.train_seq_len,self.cfg.seed,self.state['epoch'])
                while self.state['cursor']<len(windows):
                    elapsed=base+time.monotonic()-started
                    if (self.cfg.training_hours and elapsed>=self.cfg.training_hours*3600) or (self.cfg.max_optimizer_steps and self.state['step']>=self.cfg.max_optimizer_steps):break
                    step_start=time.monotonic(); cursor=self.state['cursor']; bs=self.state['batch_size']
                    stop=min(cursor+bs*self.cfg.gradient_accumulation,len(windows))
                    for j in range(cursor+1,stop):
                        if len(windows[j][1])!=len(windows[cursor][1]):
                            stop=j; break
                    plan=task_for_batch(self.cfg.seed,self.state['epoch'],self.state['step'],min_pairs=self.cfg.min_merged_pairs,max_pairs=self.cfg.max_merged_pairs)
                    self.dual.sync(); self.dual.clear(); totals={}; work_total=[0]*len(self.dual.workers)
                    for d in range(len(work_total)):torch.cuda.reset_peak_memory_stats(d)
                    for begin in range(cursor,stop,bs):
                        batch=torch.stack([v for _,v in windows[begin:min(begin+bs,stop)]])
                        loss,logs,work=self.dual.loss(batch,plan); weight=len(batch)/(stop-cursor)
                        if not torch.isfinite(loss):raise FloatingPointError('NaN/Inf training loss')
                        self.scaler.scale(loss*weight).backward()
                        for k,v in logs.items():totals[k]=totals.get(k,0.)+float(v)*weight
                        work_total=[a+b for a,b in zip(work_total,work)]
                    self.dual.reduce(); self.scaler.unscale_(self.optimizer)
                    norm=torch.nn.utils.clip_grad_norm_(self.merger.parameters(),self.cfg.clip_grad)
                    safe_checkpoint=False
                    old_scale=self.scaler.get_scale(); self.scaler.step(self.optimizer); self.scaler.update()
                    if self.scaler.get_scale()<old_scale:
                        if self.scaler.get_scale()<1:raise FloatingPointError('Repeated AMP overflow; inspect source/losses')
                        safe_checkpoint=True
                        continue  # Retry same cursor; no skipped examples.
                    self.dual.synchronize(); duration=max(time.monotonic()-step_start,1e-9)
                    self.state.update(step=self.state['step']+1,cursor=stop,seconds=base+time.monotonic()-started)
                    row=dict(**totals,grad_norm=float(norm),step=self.state['step'],epoch=self.state['epoch'],
                             pairs=str(plan),global_batch=stop-cursor,seconds=duration,tokens_per_second=sum(len(v) for _,v in windows[cursor:stop])/duration,
                             token_counts=json.dumps({l:sum(len(v) for language,v in windows[cursor:stop] if language==l) for l in self.data}),
                             languages=json.dumps([l for l,_ in windows[cursor:stop]]))
                    for d,n in enumerate(work_total):row.update({f'gpu{d}_windows':n,f'gpu{d}_peak_bytes':torch.cuda.max_memory_allocated(d)})
                    self.logs.append(row); safe_checkpoint=True
                    print(f"Step {self.state['step']} loss={totals['loss']:.5f} pairs={plan}",flush=True)
                    if self.state['seconds']-self.state['last_validation']>=self.cfg.validation_interval_minutes*60:
                        self.validate(); self.state['last_validation']=base+time.monotonic()-started
                    if self.state['step']%self.cfg.save_every_steps==0:self.save()
                if self.state['cursor']<len(windows):break
                self.state['epoch']+=1; self.state['cursor']=0; self.validate(); self.save()
            if self.state['epoch']<self.cfg.epochs:
                raise RuntimeError(f"Training incomplete: {self.state['epoch']}/{self.cfg.epochs} epochs; checkpoint saved for resume. Increase/disable the time or step cap.")
            if self.validation[-1]['step']!=self.state['step']:self.validate()
            self.state['seconds']=base+time.monotonic()-started
            self.integrity()
            self.save()
            (self.out/'checkpoints/latest.pt').replace(self.out/'checkpoints/merger_final.pt')
            atomic_json(dict(state=self.state,stop_reason='epochs',actual_training_tokens_per_language={l:sum(json.loads(r['token_counts'])[l] for r in self.logs) for l in self.data},metrics=self.logs,validation=self.validation),self.out/'logs/training_metrics.json')
            atomic_json(dict(state=self.state,signature=self.signature,stop_reason=('steps' if self.cfg.max_optimizer_steps and self.state['step']>=self.cfg.max_optimizer_steps else 'epochs' if self.state['epoch']==self.cfg.epochs else 'time_budget'),
                             checkpoint_sha256=file_digest(self.out/'checkpoints/merger_final.pt'),integrity_passed=True),self.out/'manifests/training_complete.json')
        finally:
            self.state['seconds']=base+time.monotonic()-started; self.dual.clear()
            if safe_checkpoint and not (self.out/'manifests/training_complete.json').exists():self.save()

    def rank_and_export(self):
        from .ce_search import search_and_export
        return search_and_export(self)
