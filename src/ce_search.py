"""Budget-limited CE selection; benchmark results never enter this module."""
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from .utils import atomic_json,digest,module_hash
from .pair_selection import candidate_plans,local_swaps,select_measured
from .model_export import variant,export_best_nf4


def storage_estimate(source,pairs):
    """NF4 packed weights + double-quant scales + unchanged FP16 parameters."""
    removed={p for p in pairs}; total=0.; parameters=0
    for name,p in source.named_parameters():
        if name.startswith('model.layers.') and int(name.split('.')[2]) in removed:continue
        parameters+=p.numel()
        if p.ndim==2 and name.startswith('model.layers.'):
            total+=math.ceil(p.numel()/2)+math.ceil(p.numel()/64)+4*math.ceil(p.numel()/16384)+1024
        else:total+=p.numel()*2
    return parameters,int(total+65536)  # Approximate config/header overhead; final export is measured.


def reference_storage(cfg,lock):
    info=HfApi().model_info(cfg.model_id,revision=lock['model_revision'],files_metadata=True)
    names=('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt')
    files={f.rfilename:f.size for f in info.siblings
           if f.rfilename.endswith('.safetensors') or f.rfilename in names or f.rfilename.endswith('.safetensors.index.json')}
    if not files or any(v is None for v in files.values()) or not any(n.endswith('.safetensors') for n in files):
        raise RuntimeError('Cannot verify original serialized checkpoint size')
    return dict(bytes=sum(files.values()),files=files,revision=lock['model_revision'],
                basis='Published original 16-bit checkpoint and tokenizer/config files; FP16 has the same tensor byte width')


def search_and_export(experiment):
    e=experiment; cfg=e.cfg; out=e.out
    completion=out/'manifests/training_complete.json'
    if not completion.exists():raise RuntimeError('Training must complete before CE search')
    saved=torch.load(out/'checkpoints/merger_final.pt',map_location='cpu',weights_only=False)
    if saved['signature']!=e.signature or json.loads(completion.read_text())['signature']!=e.signature:
        raise RuntimeError('Final checkpoint/run signature mismatch')
    e.merger.load_state_dict(saved['merger']); e.merger.eval(); e.dual.sync(); e.dual.clear()
    checkpoint_hash=module_hash(e.merger)
    key=digest([e.signature,checkpoint_hash,e.data_manifest['token_digest'],'ce-search-v2'])
    final_path=out/'manifests/final_export.json'
    if final_path.exists():
        final=json.loads(final_path.read_text())
        from .utils import file_digest
        if final['cache_key']!=key:raise RuntimeError('Final export checkpoint/data mismatch')
        if any(not (out/rel).is_file() or file_digest(out/rel)!=sha for rel,sha in final['files'].items()):raise RuntimeError('Final export missing or modified')
        atomic_json(final,out/'rankings/selected_plan.json')
        return final
    path=out/'rankings/ce_cache.json'
    cache=json.loads(path.read_text()) if path.exists() else dict(key=key,plans={})
    if cache['key']!=key:raise RuntimeError('CE cache belongs to different checkpoint/data/protocol')
    def flush():atomic_json(cache,path)
    from .trainer import score_tokens
    if 'baseline' not in cache:
        reference=reference_storage(cfg,e.lock)
        cache['baseline']=dict(teacher_ranking=e.score(partition='ranking',model=e.teacher)[0],
                               teacher_validation=e.score(model=e.teacher)[0],
                               source_ranking=e.score(partition='ranking')[0],reference_storage=reference,
                               unmerged_nf4_estimated_reduction=1-storage_estimate(e.source,())[1]/reference['bytes'])
        flush()
    base=cache['baseline']; original_bytes=base['reference_storage']['bytes']
    def measure(plans):
        missing=[tuple(p) for p in plans if ','.join(map(str,p)) not in cache['plans']]
        # Reuse existing replicas, one sequential queue per GPU (no extra model loads).
        def worker(rank):
            w=e.dual.workers[rank]; values=[]
            with torch.cuda.device(w.device),torch.no_grad():
                for plan in missing[rank::len(e.dual.workers)]:
                    with torch.autocast('cuda',dtype=torch.float16),variant(w.source,w.merger,plan) as model:
                        ce=float(np.mean([score_tokens(model,d['ranking'],cfg) for d in e.data.values()]))
                    if not math.isfinite(ce):raise FloatingPointError('Nonfinite candidate CE')
                    values.append((plan,ce))
            return values
        if missing:
            with ThreadPoolExecutor(max_workers=len(e.dual.workers)) as pool:
                for values in pool.map(worker,range(len(e.dual.workers))):
                    for plan,ce in values:cache['plans'][','.join(map(str,plan))]=ce
                    flush()
        rows=[]
        for plan in plans:
            ce=cache['plans'][','.join(map(str,plan))]; count,size=storage_estimate(e.source,plan)
            rows.append(dict(pairs=list(plan),merged_pairs=len(plan),remaining_layers=28-len(plan),ranking_ce=ce,
                             ce_increase_original=ce-base['teacher_ranking'],ce_increase_source=ce-base['source_ranking'],
                             estimated_parameters=count,estimated_nf4_bytes=size,reduction=1-size/original_bytes))
        return rows
    singles=measure([(i,) for i in range(27)])
    pd.DataFrame([dict(pair=i,original_layers=f'{i},{i+1}',learned_merge_ce=r['ranking_ce'],
                       ce_increase_original=r['ce_increase_original'],estimated_merge_cost=r['ce_increase_original'])
                  for i,r in enumerate(singles)]).to_csv(out/'rankings/single_pair_ce.csv',index=False)
    costs=[r['ce_increase_original'] for r in singles]
    # Reserve a small part of the TOTAL whole-plan budget for local refinements.
    reserve=min(cfg.max_refinement_rounds,max(0,cfg.max_candidate_plans//3))
    plans=candidate_plans(costs,cfg.min_merged_pairs,cfg.max_merged_pairs,cfg.max_candidate_plans-reserve)
    rows=measure(plans)
    for _ in range(cfg.max_refinement_rounds):
        if len(rows)>=cfg.max_candidate_plans:break
        best=select_measured(rows,cfg.min_storage_reduction,cfg.preferred_max_reduction)
        seen={tuple(r['pairs']) for r in rows}
        neighbors=[p for p in local_swaps(best['pairs']) if p not in seen]
        if not neighbors:break
        plan=min(neighbors,key=lambda p:(sum(costs[i] for i in p),p))
        rows+=measure([plan])
    pd.DataFrame(rows).to_csv(out/'rankings/candidate_plans.csv',index=False)
    selected=select_measured(rows,cfg.min_storage_reduction,cfg.preferred_max_reduction)
    selected=dict(selected,cache_key=key,checkpoint_hash=checkpoint_hash,baseline=base,
                  training_updates=saved['state']['step'],searched_plans=len(rows),max_candidate_plans=cfg.max_candidate_plans,
                  selection='Best measured ranking CE in preferred storage range, otherwise >=40%, otherwise best CE with unmet target; approximate budget-limited search')
    selected['validation_ce']=e.score(tuple(selected['pairs']))[0]
    selected['validation_ce_increase']=selected['validation_ce']-base['teacher_validation']
    selected_path=out/'rankings/selected_plan.json'
    export_path=out/'manifests/final_export.json'
    if export_path.exists():
        previous=json.loads(export_path.read_text())
        if previous['cache_key']!=key or previous['pairs']!=selected['pairs']:raise RuntimeError('Stale final export')
        # Verify the actual saved artifact before reusing it.
        from .utils import file_digest
        if any(not (out/rel).is_file() or file_digest(out/rel)!=sha for rel,sha in previous['files'].items()):
            raise RuntimeError('Final export missing or modified')
        return previous
    e.integrity()
    final,info=export_best_nf4(e.source,e.merger,e.tokenizer,tuple(selected['pairs']),out/'model_exports/best_learned_nf4')
    selected.update(info)
    selected['final_nf4_validation_ce']=float(np.mean([score_tokens(final,d['validation'],cfg) for d in e.data.values()]))
    selected['final_nf4_ce_increase']=selected['final_nf4_validation_ce']-base['teacher_validation']
    selected['actual_reduction']=1-info['serialized_bytes']/original_bytes
    selected['storage_target_met']=selected['actual_reduction']>=cfg.min_storage_reduction
    selected['preferred_range_met']=cfg.min_storage_reduction<=selected['actual_reduction']<=cfg.preferred_max_reduction
    del final
    from .utils import file_digest
    selected['files']={str(p.relative_to(out)):file_digest(p) for p in (out/'model_exports/best_learned_nf4').rglob('*') if p.is_file()}
    atomic_json(selected,selected_path); atomic_json(selected,export_path)
    return selected
