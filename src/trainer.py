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
from .utils import atomic_json,atomic_torch,digest,module_hash,cpu_state,rng_state,restore_rng,seed_all
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
    # Benchmark protocol is locked separately, so full evaluation can reuse pilot-trained weights.
    for key in ('evaluation_mode','pilot_questions','mmmlu_few_shot','include_few_shot','mgsm_few_shot',
                'eval_context','eval_batch_size','max_new_tokens','bootstrap_samples'):
        settings.pop(key)
    if lock_path.exists():
        lock=json.loads(lock_path.read_text())
        if lock['config']!=settings or lock['version']!=VERSION:raise RuntimeError('Incompatible run configuration; choose a fresh output_dir')
        return lock
    api=HfApi()
    revisions={name:api.dataset_info(name).sha for name in ('openai/MMMLU',cfg.include_dataset,'juletxara/mgsm')}
    revisions[cfg.wiki_dataset]='tfds-wiki40b-1.3.0' if cfg.wiki_backend=='tfds' else api.dataset_info(cfg.wiki_dataset).sha
    lock=dict(version=VERSION,config=settings,model_revision=api.model_info(cfg.model_id,revision=cfg.model_revision).sha,
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
        self.data,self.data_manifest=prepare_data(cfg,self.tokenizer,self.lock['dataset_revisions'][cfg.wiki_dataset])
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
            for p in range(27):
                a,b=[dict(x.named_parameters()) for x in self.source.model.layers[p:p+2]]
                for key,value in self.merger(a,b,p).items():torch.testing.assert_close(value,b[key].float(),rtol=0,atol=0)
            for k in range(1,6):
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
        with torch.autocast('cuda',dtype=torch.float16):
            export_model(self.source,self.merger,self.tokenizer,(0,),self.out/'model_exports/preflight',verify=True)
        atomic_json(dict(passed=True,patches=True,zero_correction=True,teacher_manual_match=True,gradient_flow=True,
                         dual_gradient_sum=True,per_gpu_work=work,joint_levels=[1,2,3,4,5],identity_kl=True,export_reload=True),
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
                    loss,_,work=self.dual.loss(batch,(0,4,8,12,16)); loss.backward(); self.dual.reduce(); self.dual.synchronize()
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
        if not safe:raise RuntimeError('No safe five-merge batch. Reduce sequence length in a new run; inspect GPU allocations.')
        self.state['batch_size']=max(safe,key=lambda r:(r['tokens_per_second'],r['batch']))['batch']

    def score(self,pairs=(),method='learned',partition='validation',model=None):
        with torch.autocast('cuda',dtype=torch.float16),variant(model or self.source,self.merger,pairs,method) as m:
            per_lang={lang:score_tokens(m,data[partition],self.cfg) for lang,data in self.data.items()}
        return float(np.mean(list(per_lang.values()))),per_lang

    def validate(self):
        self.merger.eval(); path=self.out/'rankings/validation_drop.json'
        baselines=json.loads(path.read_text()) if path.exists() else {}
        levels={k:[] for k in range(1,6)}
        for plan in checkpoint_plans():
            key=','.join(map(str,plan))
            if key not in baselines:
                baselines[key]=self.score(plan,'drop_first')[0]; atomic_json(baselines,path)
            score,_=self.score(plan); levels[len(plan)].append(score-baselines[key])
        row=dict(step=self.state['step'],epoch=self.state['epoch'],**{f'level_{k}_ce_gap':float(np.mean(v)) for k,v in levels.items()})
        row['selection_ce_gap']=float(np.mean([np.mean(v) for v in levels.values()])); self.validation.append(row)
        if self.state['best'] is None or row['selection_ce_gap']<self.state['best']:
            self.state['best']=row['selection_ce_gap']; self.snapshot('best')
        self.merger.train(); print('Validation:',row,flush=True)

    def train(self):
        resumed=self.restore()
        if (self.out/'manifests/training_complete.json').exists():return
        started=time.monotonic(); base=self.state['seconds']
        try:
            if not resumed:self.snapshot('initial'); self.preflight()
            self.tune_batch()
            if not self.validation:self.validate()
            self.state['last_validation']=base+time.monotonic()-started
            while self.state['epoch']<self.cfg.epochs:
                windows=balanced_epoch(self.data,self.cfg.train_seq_len,self.cfg.seed,self.state['epoch'])
                while self.state['cursor']<len(windows):
                    elapsed=base+time.monotonic()-started
                    if elapsed>=self.cfg.training_hours*3600:break
                    step_start=time.monotonic(); cursor=self.state['cursor']; bs=self.state['batch_size']
                    stop=min(cursor+bs*self.cfg.gradient_accumulation,len(windows))
                    plan=task_for_batch(self.cfg.seed,self.state['epoch'],cursor//(bs*self.cfg.gradient_accumulation))
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
                    old_scale=self.scaler.get_scale(); self.scaler.step(self.optimizer); self.scaler.update()
                    if self.scaler.get_scale()<old_scale:
                        if self.scaler.get_scale()<1:raise FloatingPointError('Repeated AMP overflow; inspect source/losses')
                        continue  # Retry same cursor; no skipped examples.
                    self.dual.synchronize(); duration=time.monotonic()-step_start
                    self.state.update(step=self.state['step']+1,cursor=stop,seconds=base+time.monotonic()-started)
                    row=dict(**totals,grad_norm=float(norm),step=self.state['step'],epoch=self.state['epoch'],
                             pairs=str(plan),global_batch=stop-cursor,seconds=duration,tokens_per_second=(stop-cursor)*self.cfg.train_seq_len/duration,
                             languages=json.dumps([l for l,_ in windows[cursor:stop]]))
                    for d,n in enumerate(work_total):row.update({f'gpu{d}_windows':n,f'gpu{d}_peak_bytes':torch.cuda.max_memory_allocated(d)})
                    self.logs.append(row); print(f"Step {self.state['step']} loss={totals['loss']:.5f} pairs={plan}",flush=True)
                    if self.state['seconds']-self.state['last_validation']>=self.cfg.validation_interval_minutes*60:
                        self.validate(); self.state['last_validation']=base+time.monotonic()-started
                    if self.state['step']%self.cfg.save_every_steps==0:self.save()
                if self.state['cursor']<len(windows):break
                self.state['epoch']+=1; self.state['cursor']=0; self.validate(); self.save()
            if self.validation[-1]['step']!=self.state['step']:self.validate()
            self.state['seconds']=base+time.monotonic()-started
            self.snapshot('final'); self.integrity()
            atomic_json(dict(state=self.state,stop_reason='epochs' if self.state['epoch']==self.cfg.epochs else 'time_budget',
                             integrity_passed=True),self.out/'manifests/training_complete.json')
        finally:
            self.state['seconds']=base+time.monotonic()-started; self.dual.clear(); self.save()

    def rank_and_export(self):
        if not (self.out/'manifests/training_complete.json').exists():raise RuntimeError('Finish the training stage before ranking/export')
        saved=torch.load(self.out/'checkpoints/best.pt',map_location='cpu',weights_only=False)
        if saved['signature']!=self.signature:raise RuntimeError('Best checkpoint mismatch')
        self.merger.load_state_dict(saved['merger']); self.merger.eval()
        if saved['state']['step']==0:
            print('Best checkpoint is the zero-correction initialization. Do not report learned improvement.',flush=True)
        cache_path=self.out/'rankings/cache.json'; cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
        checkpoint_hash=module_hash(self.merger); diagnostic_rows={}
        def score(plan,method):
            key=digest([self.signature,checkpoint_hash,list(plan),method])
            if key not in cache:cache[key]=self.score(plan,method,'ranking'); atomic_json(cache,cache_path)
            for language,ce in cache[key][1].items():
                diagnostic_rows[(str(plan),method,language)]=dict(pairs=str(plan),method=method,language=language,ce=ce,ppl=math.exp(ce))
            return cache[key][0]
        baseline=score((),'learned'); rows=[]
        for p in range(27):
            learned=score((p,),'learned'); drop=score((p,),'drop_first')
            accepted=learned<=drop-self.cfg.accept_margin_ce
            rows.append(dict(pair=p,learned_ce=learned,learned_ppl=math.exp(learned),drop_first_ce=drop,drop_first_ppl=math.exp(drop),accepted=accepted,
                             raw_cost=100*math.expm1(learned-baseline),guarded_cost=100*math.expm1((learned if accepted else drop)-baseline)))
        pd.DataFrame(rows).to_csv(self.out/'rankings/single_pairs.csv',index=False)
        plans={}; exports=[]; controls=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            generated={}
            for p in range(27):
                a,b=[dict(x.named_parameters()) for x in self.source.model.layers[p:p+2]]
                generated[p]={k:v.detach().half().cpu() for k,v in self.merger(a,b,p).items()}
            atomic_torch(dict(checkpoint_hash=checkpoint_hash,pairs=generated),self.out/'checkpoints/generated_pairs.pt')
            del generated
        # Raw plans are primary. Reference-style guarded additive plans are a separately labelled control.
        for selection in ('raw','guarded'):
            costs=[r['raw_cost' if selection=='raw' else 'guarded_cost'] for r in rows]
            for k in range(1,6):
                pairs=compatible_selection(costs,k); learned=score(pairs,'learned'); drop=score(pairs,'drop_first')
                accepted=learned<=drop-self.cfg.accept_margin_ce
                key=f'{selection}_{k}'; plans[key]=dict(pairs=list(pairs),learned_ce=learned,drop_first_ce=drop,accepted=accepted,
                                                      additive_cost=sum(costs[p] for p in pairs))
                methods=('learned','drop_first','drop_second','average') if selection=='raw' else ('learned' if accepted else 'drop_first',)
                for method in methods:
                    label=f'{method}_{k}' if selection=='raw' else f'guarded_{k}'
                    if selection!='raw' or method!='learned':
                        controls.append(dict(name=label+'_fp16',path=self.cfg.model_id,revision=self.lock['model_revision'],nf4=False,
                                             layers=28-k,control=True,method=method,pairs=list(pairs)))
                        continue
                    destination=self.out/'model_exports'/label
                    with torch.autocast('cuda',dtype=torch.float16):
                        info=export_model(self.source,self.merger,self.tokenizer,pairs,destination,method,
                                          dict(selection=selection,accepted=accepted,selected_step=saved['state']['step']))
                    exports.append(dict(name=label,path=str(destination),**info))
                atomic_json(plans,self.out/'rankings/plans.json')
                atomic_json(exports,self.out/'manifests/exports.json')
                atomic_json(controls,self.out/'manifests/controls.json')
        quant=load_model(self.cfg.model_id,self.lock['model_revision'],'cuda:0',True)
        qce={lang:score_tokens(quant,d['ranking'],self.cfg) for lang,d in self.data.items()}; del quant
        tce={lang:score_tokens(self.teacher,d['ranking'],self.cfg) for lang,d in self.data.items()}
        for method,values in [('original_fp16',tce),('original_nf4',qce)]:
            for language,ce in values.items():diagnostic_rows[('',method,language)]=dict(pairs='',method=method,language=language,ce=ce,ppl=math.exp(ce))
        pd.DataFrame(diagnostic_rows.values()).to_csv(self.out/'logs/wiki_ranking_diagnostics.csv',index=False)
        atomic_json(dict(nf4_ce=qce,teacher_ce=tce,dequantized_source_ce=baseline,checkpoint_hash=checkpoint_hash),self.out/'rankings/baselines.json')
        self.integrity()
