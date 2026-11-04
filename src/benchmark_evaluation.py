import gc
import hashlib
import json
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from datasets import get_dataset_config_names,load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi
from benchmarks import mmmlu,include,mgsm
from .utils import atomic_json,digest
from .quantization import load_model,dequantized_copy,MODEL_LOAD_LOCK

def content_id(row):return hashlib.sha256(row['question'].strip().encode()).hexdigest()

def stratified_sample(rows,n,seed):
    rng=np.random.default_rng(seed); groups={}
    for row in rows:groups.setdefault(row.get('subject','unknown'),[]).append(row)
    for group in groups.values():rng.shuffle(group)
    result=[]
    while len(result)<min(n,len(rows)):
        for key in rng.permutation(sorted(groups)):
            if groups[key]:result.append(groups[key].pop())
            if len(result)==min(n,len(rows)):break
    return result

def mc_prompt(support,target):
    def question(row):
        return row['question']+'\n'+'\n'.join(f'{chr(65+i)}. {c}' for i,c in enumerate(row['choices']))+'\nAnswer:'
    return '\n\n'.join([question(r)+' '+chr(65+r['label']) for r in support]+[question(target)])

def prepare_benchmarks(cfg,lock,tokenizer):
    path=Path(cfg.output_dir)/f'manifests/benchmark_prompts_{cfg.evaluation_mode}.json'
    protocol=dict(language_subset=cfg.benchmark_languages,experiment_mode=cfg.mode,mode=cfg.evaluation_mode,pilot=cfg.pilot_questions,seed=cfg.seed,
                  shots=[cfg.mmmlu_few_shot,cfg.include_few_shot,cfg.mgsm_few_shot],context=cfg.eval_context,
                  max_new_tokens=cfg.max_new_tokens,model_revision=lock['model_revision'],revisions=lock['dataset_revisions'],
                  version='fewshot-label-likelihood-v1',overflow='skip identical IDs for every variant; never truncate',
                  mc_support='fixed stratified disjoint test support when no validation split; non-official protocol',
                  mgsm_alignment='same official row indices across languages; verify answer_number sequence')
    protocol=json.loads(json.dumps(protocol))
    if path.exists():
        result=json.loads(path.read_text(encoding='utf-8'))
        if result['protocol']!=protocol:raise RuntimeError('Benchmark protocol changed; refuse stale predictions')
        if result['signature']!=digest(dict(protocol=protocol,jobs=result['jobs'])):raise RuntimeError('Prompt cache hash mismatch')
        return result
    jobs=[]; mgsm_answers=None; shared_indices=None
    specifications=[('MMMLU','openai/MMMLU',mmmlu.LOCALES,cfg.mmmlu_few_shot,mmmlu.adapt),
                    ('INCLUDE',cfg.include_dataset,None,cfg.include_few_shot,include.adapt),
                    ('MGSM','juletxara/mgsm',mgsm.LANGUAGES,cfg.mgsm_few_shot,None)]
    for benchmark,dataset,configs,shots,adapter in specifications:
        revision=lock['dataset_revisions'][dataset]
        if configs is None:configs=[c for c in get_dataset_config_names(dataset,revision=revision) if c not in ('default','all')]
        if benchmark=='INCLUDE':
            files=HfApi().list_repo_files(dataset,repo_type='dataset',revision=revision)
            published={p.split('/')[0] for p in files if '/test-' in p and p.endswith('.parquet')}
            missing=set(configs)-published
            if missing-{'Dutch-Flemish'}:raise RuntimeError(f'Unexpected missing INCLUDE files: {missing}')
            configs=[c for c in configs if c in published]
        if (cfg.mode=='smoke' and cfg.evaluation_mode=='pilot') or cfg.benchmark_languages:
            requested=(cfg.benchmark_languages or {}).get(benchmark,[])
            chosen=[c for c in requested if c in configs]
            if cfg.mode=='smoke' and cfg.evaluation_mode=='pilot':
                chosen+=sorted(set(configs)-set(chosen))[:max(0,3-len(chosen))]
                configs=chosen[:3]
            elif requested:
                if len(chosen)!=len(requested):raise ValueError(f'Unavailable {benchmark} languages')
                configs=chosen
        for language in configs:
            ds=load_dataset(dataset,language,revision=revision)
            if benchmark=='MGSM':
                support=list(ds['train'])[:shots]
                if len(support)!=shots:raise ValueError('MGSM few-shot support missing')
                tests=list(ds['test'])
                if len(tests)!=250:raise ValueError('Expected 250 official MGSM test problems')
                answers=[r['answer_number'] for r in tests]
                if mgsm_answers is None:
                    mgsm_answers=answers
                    shared_indices=sorted(np.random.default_rng(cfg.seed).choice(250,min(cfg.pilot_questions,250),replace=False).tolist())
                elif answers!=mgsm_answers:raise RuntimeError('MGSM problem alignment changed; inspect dataset before proceeding')
                selected=range(250) if cfg.evaluation_mode=='full' else shared_indices
                support_hashes={content_id(r) for r in support}
                examples=[]
                for i in selected:
                    row=tests[i]
                    if content_id(row) in support_hashes:raise AssertionError('MGSM demonstration/test leakage')
                    examples.append(dict(id=str(i),prompt=mgsm.prompt(support,row['question'],language),
                                         target=mgsm.normalized_number(row['answer_number']),choices=None,
                                         subject='arithmetic',domain='math',regional='unknown',support_ids=[f'train:{j}' for j in range(shots)]))
            else:
                tests=[adapter(r,i) for i,r in enumerate(ds['test'])]
                # Deduplicate by target text before support selection.
                tests=list({content_id(r):r for r in reversed(tests)}.values())
                if 'validation' in ds:
                    pool=[adapter(r,i) for i,r in enumerate(ds['validation'])]; support_split='validation'
                else:pool=tests; support_split='test-support'
                support=stratified_sample(pool,shots,cfg.seed)
                if len(support)!=shots:raise ValueError(f'{benchmark}/{language}: insufficient demonstrations')
                support_hashes={content_id(r) for r in support}; eligible=[r for r in tests if content_id(r) not in support_hashes]
                selected=eligible if cfg.evaluation_mode=='full' else stratified_sample(eligible,cfg.pilot_questions,cfg.seed)
                examples=[dict(id=r['id'],prompt=mc_prompt(support,r),target=r['label'],choices=r['choices'],
                               subject=r['subject'],domain=r['domain'],regional=r['regional'],
                               support_ids=[f'{support_split}:{s["id"]}' for s in support]) for r in selected]
            for example in examples:
                example['input_ids']=tokenizer.encode(example['prompt'],add_special_tokens=False)
                example['prompt_hash']=digest(example['prompt'])
                reserve=cfg.max_new_tokens if benchmark=='MGSM' else 8
                example['skip_reason']='context_overflow' if len(example['input_ids'])+reserve>cfg.eval_context else None
                if benchmark=='MGSM' and cfg.evaluation_mode=='full' and example['skip_reason']:
                    raise ValueError(f'Full MGSM must score all 250 examples: {language}/{example["id"]} exceeds '
                                     'eval_context. Increase it within the model limit before freezing the protocol.')
            jobs.append(dict(benchmark=benchmark,language=language,examples=examples,
                             fingerprint={k:v._fingerprint for k,v in ds.items()}))
            print('Prepared',benchmark,language,len(examples),flush=True)
    result=dict(protocol=protocol,jobs=jobs,signature=digest(dict(protocol=protocol,jobs=jobs)))
    atomic_json(result,path); return result

@torch.inference_mode()
def score_choices(model,tokenizer,examples,device,batch_size=4):
    """Batched causal likelihood of explicit candidate-label token sequences, including multi-token labels."""
    results=[]
    for example in examples:
        prefix=example['input_ids']; scores=[]
        candidates=[tokenizer.encode(' '+chr(65+i),add_special_tokens=False) for i in range(len(example['choices']))]
        for start in range(0,len(candidates),batch_size):
            labels=candidates[start:start+batch_size]; lengths=[len(prefix)+len(c) for c in labels]; width=max(lengths)
            ids=torch.full((len(labels),width),tokenizer.pad_token_id,device=device,dtype=torch.long); mask=torch.zeros_like(ids)
            for i,c in enumerate(labels):ids[i,:lengths[i]]=torch.tensor(prefix+c,device=device); mask[i,:lengths[i]]=1
            h=model.model(ids,attention_mask=mask,use_cache=False).last_hidden_state
            for i,c in enumerate(labels):
                logits=model.lm_head(h[i,len(prefix)-1:len(prefix)+len(c)-1]).float()
                lp=logits.log_softmax(-1)
                scores.append(float(lp.gather(1,torch.tensor(c,device=device)[:,None]).sum()))
        results.append(dict(prediction=int(np.argmax(scores)),scores=scores,invalid=False,output_tokens=0))
    return results

@torch.inference_mode()
def generate_answers(model,tokenizer,examples,device,cfg,language):
    width=max(len(e['input_ids']) for e in examples)
    ids=torch.full((len(examples),width),tokenizer.pad_token_id,device=device,dtype=torch.long); mask=torch.zeros_like(ids)
    for i,e in enumerate(examples):
        n=len(e['input_ids']); ids[i,-n:]=torch.tensor(e['input_ids'],device=device); mask[i,-n:]=1
    generated=model.generate(ids,attention_mask=mask,do_sample=False,max_new_tokens=cfg.max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id,eos_token_id=model.generation_config.eos_token_id)
    results=[]
    for row in generated[:,width:]:
        text=tokenizer.decode(row,skip_special_tokens=True); answer=mgsm.extract_answer(text,language)
        results.append(dict(prediction=answer,text=text,invalid=answer is None,
                            output_tokens=int((row!=tokenizer.pad_token_id).sum())))
    return results

def variants(cfg,lock):
    out=Path(cfg.output_dir)
    final=json.loads((out/'manifests/final_export.json').read_text())
    from .utils import file_digest
    if any(not (out/rel).is_file() or file_digest(out/rel)!=sha for rel,sha in final['files'].items()):
        raise RuntimeError('Final export integrity check failed')
    return [dict(name='original_fp16',path=cfg.model_id,revision=lock['model_revision'],nf4=False,layers=28),
            dict(name='best_learned_nf4',path=str(out/'model_exports/best_learned_nf4'),revision=None,nf4=True,layers=final['remaining_layers'])]

def read_predictions(path):
    if not path.exists():return []
    # A process interruption can leave one incomplete final line; retain only valid records.
    rows=[]; lines=path.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i!=len(lines)-1:raise
    return rows

def evaluate_variant(cfg,lock,manifest,variant_spec,smoke=False):
    out=Path(cfg.output_dir); name=variant_spec['name']; needed=1 if cfg.single_gpu_debug else 2
    if torch.cuda.device_count()<needed:raise RuntimeError('Benchmark evaluation requires two GPUs (or explicit debug mode)')
    jobs=manifest['jobs']
    if smoke:
        jobs=[dict(next(j for j in jobs if j['benchmark']==b)) for b in ('MMMLU','INCLUDE','MGSM')]
        jobs=[dict(j,examples=[e for e in j['examples'] if not e['skip_reason']][:3]) for j in jobs]
    def work(rank):
        device=f'cuda:{rank}'; torch.cuda.set_device(rank)
        tok=AutoTokenizer.from_pretrained(cfg.model_id,revision=lock['model_revision'])
        if tok.pad_token_id is None:tok.pad_token=tok.eos_token
        if variant_spec.get('control'):
            from .model_export import compress_in_place
            quant=load_model(variant_spec['path'],variant_spec['revision'],device,True)
            model,_=dequantized_copy(quant,device); del quant
            pairs=None
            if variant_spec['method']=='learned':
                pairs=torch.load(out/'checkpoints/generated_pairs.pt',map_location='cpu',weights_only=True)['pairs']
            model=compress_in_place(model,variant_spec['pairs'],variant_spec['method'],pairs); del pairs
        else:model=load_model(variant_spec['path'],variant_spec['revision'],device,variant_spec['nf4'])
        if cfg.eval_context>model.config.max_position_embeddings:raise ValueError('Evaluation context exceeds model capacity')
        # Hash actual tensors plus quantization metadata to invalidate changed checkpoint results.
        from .utils import module_hash
        weight_hash=module_hash(model)
        identity=digest([manifest['signature'],variant_spec,weight_hash,'smoke' if smoke else 'evaluation'])
        torch.cuda.reset_peak_memory_stats(rank); start=time.monotonic(); input_tokens=output_tokens=0
        root=out/('preflight_predictions' if smoke else 'benchmark_results')/cfg.evaluation_mode/name; root.mkdir(parents=True,exist_ok=True)
        for job in jobs[rank::needed]:
            path=root/f"{job['benchmark']}_{job['language']}.jsonl"; old=read_predictions(path)
            if any(r['identity']!=identity for r in old):raise RuntimeError(f'Stale benchmark predictions: {path}')
            # Rewrite valid prefix atomically before appending; drops an interrupted partial line.
            tmp=path.with_suffix('.tmp'); tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in old),encoding='utf-8'); tmp.replace(path)
            done={r['id'] for r in old}; pending=[e for e in job['examples'] if e['id'] not in done]
            for offset in range(0,len(pending),cfg.eval_batch_size):
                batch=pending[offset:offset+cfg.eval_batch_size]; active=[e for e in batch if not e['skip_reason']]
                with torch.autocast('cuda',dtype=torch.float16):
                    predicted=(generate_answers(model,tok,active,device,cfg,job['language']) if job['benchmark']=='MGSM'
                               else score_choices(model,tok,active,device,cfg.eval_batch_size)) if active else []
                by_id={e['id']:p for e,p in zip(active,predicted)}
                with path.open('a',encoding='utf-8') as stream:
                    for e in batch:
                        prediction=by_id.get(e['id'],dict(prediction=None,invalid=True,output_tokens=0))
                        row=dict(identity=identity,variant=name,benchmark=job['benchmark'],language=job['language'],id=e['id'],
                                 target=e['target'],correct=prediction['prediction']==e['target'] if not e['skip_reason'] else None,
                                 prompt_hash=e['prompt_hash'],support_ids=e['support_ids'],skip_reason=e['skip_reason'],
                                 subject=e['subject'],domain=e['domain'],regional=e['regional'],**prediction)
                        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()
                        if not e['skip_reason']:input_tokens+=len(e['input_ids']); output_tokens+=prediction['output_tokens']
            print(name,job['benchmark'],job['language'],'complete',flush=True)
        torch.cuda.synchronize(rank); duration=time.monotonic()-start
        stats=dict(variant=name,gpu=rank,benchmark=','.join(sorted({j['benchmark'] for j in jobs})),seconds=duration,new_input_tokens=input_tokens,new_output_tokens=output_tokens,
                   tokens_per_second=(input_tokens+output_tokens)/max(duration,1e-9),peak_bytes=torch.cuda.max_memory_allocated(rank),
                   layers=variant_spec['layers'],quantization='NF4' if variant_spec['nf4'] else 'FP16',weight_hash=weight_hash)
        # Parameters use original dense shapes, never 4-bit packed storage element counts.
        from transformers import AutoConfig,AutoModelForCausalLM
        dense_config=AutoConfig.from_pretrained(variant_spec['path'],revision=variant_spec['revision'])
        dense_config.num_hidden_layers=variant_spec['layers']
        with MODEL_LOAD_LOCK, torch.device('meta'):
            dense=AutoModelForCausalLM.from_config(dense_config)
        stats['parameters']=sum(p.numel() for p in dense.parameters()); del dense
        final=json.loads((out/'manifests/final_export.json').read_text())
        if rank==0:
            stats['stored_bytes']=(final['baseline']['reference_storage']['bytes'] if name=='original_fp16'
                                   else final['serialized_bytes'])
        stat_path=root/f'performance_gpu{rank}_{"-".join(sorted({j["benchmark"] for j in jobs}))}.json'
        if stat_path.exists():
            old=json.loads(stat_path.read_text())
            if old['weight_hash']!=weight_hash:raise RuntimeError('Stale performance measurements')
            stats['seconds']+=old['seconds']; stats['new_input_tokens']+=old['new_input_tokens']; stats['new_output_tokens']+=old['new_output_tokens']
            stats['peak_bytes']=max(stats['peak_bytes'],old['peak_bytes'])
            stats['tokens_per_second']=(stats['new_input_tokens']+stats['new_output_tokens'])/stats['seconds']
        atomic_json(stats,stat_path); del model; gc.collect(); torch.cuda.empty_cache()
        return stats
    with ThreadPoolExecutor(max_workers=needed) as pool:return list(pool.map(work,range(needed)))
