"""Budget-limited CE selection; benchmark results never enter this module."""
import json
import copy
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from .utils import atomic_json,digest,module_hash
from .pair_selection import candidate_plans,local_swaps,select_measured,select_feasible,pareto_frontier
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


def search_and_export(experiment,export=True):
    e=experiment; cfg=e.cfg; out=e.out
    completion=out/'manifests/training_complete.json'
    if not completion.exists():raise RuntimeError('Training must complete before CE search')
    saved=torch.load(out/'checkpoints/merger_final.pt',map_location='cpu',weights_only=False)
    if saved['state']['epoch']!=cfg.epochs:raise RuntimeError('Training has not completed every epoch')
    if saved['signature']!=e.signature or json.loads(completion.read_text())['signature']!=e.signature:
        raise RuntimeError('Final checkpoint/run signature mismatch')
    e.merger.load_state_dict(saved['merger']); e.merger.eval(); e.dual.sync(); e.dual.clear()
    checkpoint_hash=module_hash(e.merger)
    key=digest([e.signature,checkpoint_hash,e.data_manifest['token_digest'],'ce-search-ce20-v4'])
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
    from .quantization import load_model,verify_nf4,quantize_dense_nf4
    if 'baseline' not in cache:
        reference=reference_storage(cfg,e.lock)
        baseline=load_model(str(out/'reference_models/original_nf4'),None,'cuda:0',True)
        verify_nf4(baseline)
        def baseline_ce(partition):
            return float(np.mean([score_tokens(baseline,d[partition],cfg) for d in e.data.values()]))
        cache['baseline']=dict(nf4_ranking=baseline_ce('ranking'),nf4_validation=baseline_ce('validation'),
                               nf4_serialized_bytes=e.quant_manifest['nf4_serialized_bytes'],
                               source_ranking=e.score(partition='ranking')[0],reference_storage=reference)
        del baseline; torch.cuda.empty_cache(); flush()
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
                             ce_increase_original=ce-base['nf4_ranking'],ce_increase_source=ce-base['source_ranking'],
                             estimated_parameters=count,estimated_nf4_bytes=size,reduction=1-size/original_bytes))
        return rows
    singles=measure([(i,) for i in range(27)])
    pd.DataFrame([dict(pair=i,original_layers=f'{i},{i+1}',learned_merge_ce=r['ranking_ce'],
                       ce_increase_original=r['ce_increase_original'],estimated_merge_cost=r['ce_increase_original'])
                  for i,r in enumerate(singles)]).to_csv(out/'rankings/single_pair_ce.csv',index=False)
    costs=[r['ce_increase_original'] for r in singles]
    # Reserve a small part of the TOTAL whole-plan budget for local refinements.
    reserve=cfg.max_refinement_rounds if cfg.max_relative_ce_increase is not None else min(cfg.max_refinement_rounds,max(0,cfg.max_candidate_plans//3))
    plans=candidate_plans(costs,cfg.min_merged_pairs,cfg.max_merged_pairs,cfg.max_candidate_plans-reserve)
    rows=measure(plans)
    for _ in range(cfg.max_refinement_rounds):
        if len(rows)>=cfg.max_candidate_plans:break
        best=select_measured(rows,cfg.min_storage_reduction,cfg.ce_comparable_tolerance)
        seen={tuple(r['pairs']) for r in rows}
        neighbors=[p for p in local_swaps(best['pairs']) if p not in seen]
        if not neighbors:break
        plan=min(neighbors,key=lambda p:(sum(costs[i] for i in p),p))
        rows+=measure([plan])
    for row in rows:row['measurement']='dequantized FP16 surrogate CE; not NF4 feasibility'
    pd.DataFrame(rows).to_csv(out/'rankings/candidate_plans.csv',index=False)
    # NF4 finalists are scored in memory, with no candidate checkpoint exports.
    shortlist=[]
    if cfg.max_relative_ce_increase is not None:
        for k in range(cfg.min_merged_pairs,cfg.max_merged_pairs+1):
            shortlist.append(min((r for r in rows if len(r['pairs'])==k),key=lambda r:(r['ranking_ce'],tuple(r['pairs']))))
    remaining=[r for r in rows if r not in shortlist]
    for _ in range(min(cfg.max_final_nf4_candidates-len(shortlist),len(remaining))):
        candidate=select_measured(remaining,cfg.min_storage_reduction,cfg.ce_comparable_tolerance)
        shortlist.append(candidate); remaining.remove(candidate)
    cache.setdefault('nf4_plans',{})
    finalists=[]
    for candidate in shortlist:
        plan=tuple(candidate['pairs']); plan_key=','.join(map(str,plan))
        if plan_key not in cache['nf4_plans']:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16),variant(e.source,e.merger,plan):
                final=quantize_dense_nf4(copy.deepcopy(e.source))
            verify_nf4(final)
            ce=float(np.mean([score_tokens(final,d['ranking'],cfg) for d in e.data.values()]))
            if not math.isfinite(ce):raise FloatingPointError('Nonfinite NF4 finalist CE')
            cache['nf4_plans'][plan_key]=dict(ce=ce,weight_hash=module_hash(final))
            del final; torch.cuda.empty_cache(); flush()
        result=cache['nf4_plans'][plan_key]
        finalists.append(dict(candidate,surrogate_ranking_ce=candidate['ranking_ce'],ranking_ce=result['ce'],
                              nf4_weight_hash=result['weight_hash'],ce_increase_original=result['ce']-base['nf4_ranking']))
    threshold=cfg.max_relative_ce_increase
    selected=(select_feasible(finalists,base['nf4_ranking'],threshold,cfg.min_storage_reduction) if threshold is not None
              else select_measured(finalists,cfg.min_storage_reduction,cfg.ce_comparable_tolerance))
    reference=dict(pairs=[],merged_pairs=0,remaining_layers=28,ranking_ce=base['nf4_ranking'],
                   estimated_parameters=sum(p.numel() for p in e.source.parameters()),
                   estimated_nf4_bytes=base['nf4_serialized_bytes'],serialized_bytes=base['nf4_serialized_bytes'],
                   reduction=1-base['nf4_serialized_bytes']/original_bytes)
    frontier=pareto_frontier([reference]+finalists)
    for row in [reference]+finalists:
        gap=row['ranking_ce']-base['nf4_ranking']
        row.update(original_nf4_ce=base['nf4_ranking'],ce_increase_original=gap,
                   relative_ce_increase=gap/base['nf4_ranking'],relative_ce_increase_percent=100*gap/base['nf4_ranking'],
                   perplexity=math.exp(min(row['ranking_ce'],700)),perplexity_ratio=math.exp(min(gap,700)),
                   reduction_vs_original_nf4=1-row['estimated_nf4_bytes']/base['nf4_serialized_bytes'],
                   ce_target_met=threshold is None or row['ranking_ce']<=base['nf4_ranking']*(1+threshold),
                   storage_target_met=row['reduction']>=cfg.min_storage_reduction,
                   on_pareto_frontier=any(row is p for p in frontier),measurement='true NF4 CE',
                   storage_basis='measured serialized' if not row['pairs'] else 'estimated serialized; selected export verified separately')
        row['feasible']=bool(row['pairs']) and row['ce_target_met'] and row['storage_target_met']
    pd.DataFrame([reference]+finalists).to_csv(out/'rankings/nf4_finalists.csv',index=False)
    pd.DataFrame(frontier).to_csv(out/'rankings/pareto_frontier.csv',index=False)
    from .reporting import plot_pareto
    plot_pareto([reference]+finalists,selected,threshold,out/'charts/ce_compression_pareto.png')
    selected=dict(selected,cache_key=key,checkpoint_hash=checkpoint_hash,baseline=base,
                  training_updates=saved['state']['step'],completed_epochs=saved['state']['epoch'],
                  searched_plans=len(rows),nf4_finalists=len(finalists),max_candidate_plans=cfg.max_candidate_plans,
                  ce_target_met=selected['ce_target_met'],feasible=selected['feasible'],max_relative_ce_increase=threshold,
                  minimum_measured_relative_ce_increase=min(r['relative_ce_increase'] for r in finalists),
                  selection=('Most merged pairs with measured NF4 CE within the relative threshold and estimated storage target; lower CE then pair indices break ties. If infeasible, lowest-CE learned diagnostic. All configured merge depths measured; budget-limited, not a global optimum.' if threshold is not None else 'Lowest NF4 CE with compression near-tie preference.'))
    selected_path=out/'rankings/selected_plan.json'
    atomic_json(selected,selected_path)
    if not export:return selected
    e.integrity()
    final,info=export_best_nf4(e.source,e.merger,e.tokenizer,tuple(selected['pairs']),out/'model_exports/best_learned_nf4')
    if info['weight_hash']!=selected['nf4_weight_hash']:raise RuntimeError('Export differs from scored NF4 finalist')
    selected.update(info)
    selected['final_nf4_validation_ce']=float(np.mean([score_tokens(final,d['validation'],cfg) for d in e.data.values()]))
    selected['final_nf4_ce_increase']=selected['final_nf4_validation_ce']-base['nf4_validation']
    selected['actual_reduction']=1-info['serialized_bytes']/original_bytes
    selected['reduction_vs_original_nf4']=1-info['serialized_bytes']/base['nf4_serialized_bytes']
    selected['storage_target_met']=selected['actual_reduction']>=cfg.min_storage_reduction
    selected['feasible']=selected['ce_target_met'] and selected['storage_target_met']
    selected['target_50_met']=selected['actual_reduction']>=cfg.target_storage_reduction
    del final
    from .utils import file_digest
    selected['files']={str(p.relative_to(out)):file_digest(p) for p in (out/'model_exports/best_learned_nf4').rglob('*') if p.is_file()}
    atomic_json(selected,selected_path); atomic_json(selected,out/'manifests/final_export.json')
    return selected
