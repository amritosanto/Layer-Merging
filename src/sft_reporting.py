"""Post-SFT paired accuracy; pre-SFT CE remains a separate diagnostic."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from .benchmark_evaluation import read_predictions,prompt_path,evaluation_namespace
from .reporting import bootstrap_difference
from .sft_training import verified_specs,NAMES
from .trainer import prepare_run
from .utils import digest


def make_sft_report(cfg):
    out=Path(cfg.output_dir);lock=prepare_run(cfg);specs=verified_specs(cfg,lock)
    manifest=json.loads(prompt_path(cfg).read_text(encoding='utf-8'))
    gate=json.loads((out/'manifests/sft_verified.json').read_text())
    identities={s['name']:digest([manifest['signature'],s,gate['weight_hashes'][s['name']],'evaluation']) for s in specs}
    root=out/'benchmark_results'/evaluation_namespace(cfg);table=[];frequencies=[]
    for job in manifest['jobs']:
        expected={e['id']:e for e in job['examples']};paired=[]
        for name in NAMES:
            rows=read_predictions(root/name/f"{job['benchmark']}_{job['language']}.jsonl")
            if len(rows)!=len(expected) or {r['id'] for r in rows}!=set(expected):raise RuntimeError('Incomplete post-SFT benchmark')
            for r in rows:
                if r['identity']!=identities[name]:raise RuntimeError('Stale post-SFT prediction identity')
                if any(r[k]!=expected[r['id']][k] for k in ('prompt_hash','support_ids','target','skip_reason')):
                    raise RuntimeError('Post-SFT paired prompt/answer mismatch')
            paired.append({r['id']:r for r in rows})
        keys=[k for k,e in expected.items() if not e['skip_reason']]
        if not keys:raise RuntimeError('No scored questions in benchmark language')
        a,b=paired;av=[a[k]['correct'] for k in keys];bv=[b[k]['correct'] for k in keys]
        aa=float(np.mean(av));ba=float(np.mean(bv))
        ci=bootstrap_difference(bv,av,cfg.seed,cfg.bootstrap_samples) if len(keys)>1 else (None,None)
        row=dict(benchmark=job['benchmark'],language=job['language'],questions=len(keys),
                 requested_questions=job.get('requested_questions'),skipped_questions=len(expected)-len(keys),
                 count_reason=job.get('count_reason'),original_fp16_sft_correct=sum(av),merged_nf4_sft_correct=sum(bv),
                 original_fp16_sft_accuracy=aa,merged_nf4_sft_accuracy=ba,accuracy_difference_pp=100*(ba-aa),
                 relative_accuracy_retention=ba/aa if aa else None,paired_ci95_lower_pp=ci[0],paired_ci95_upper_pp=ci[1],
                 accuracy_diagnostic_met=aa-ba<=cfg.accuracy_loss_target)
        for name,records in zip(NAMES,paired):
            row[name+'_incorrect']=len(keys)-sum(records[k]['correct'] for k in keys)
            row[name+'_invalid']=sum(records[k]['invalid'] for k in keys)
            row[name+'_truncated']=sum(records[k].get('truncated',False) for k in keys)
            row[name+'_invalid_rate']=row[name+'_invalid']/len(keys)
            row[name+'_truncation_rate']=row[name+'_truncated']/len(keys)
            if job['benchmark']!='MGSM':
                counts=Counter(records[k]['prediction'] for k in keys)
                margins=[sorted(records[k]['scores'])[-1]-sorted(records[k]['scores'])[-2] for k in keys if len(records[k].get('scores',[]))>=2]
                row[name+'_mean_label_score_margin']=float(np.mean(margins)) if margins else None
                row[name+'_dominant_option_fraction']=max(counts.values())/len(keys)
                row[name+'_option_collapse_flag']=max(counts.values())/len(keys)>=.8
                for option,count in {**{i:0 for i in range(4)},**counts}.items():
                    frequencies.append(dict(benchmark=job['benchmark'],language=job['language'],variant=name,
                                            option=chr(65+option) if isinstance(option,int) else 'invalid',count=count,frequency=count/len(keys)))
        table.append(row)
    frame=pd.DataFrame(table);frame.to_csv(root/'per_language_accuracy.csv',index=False)
    pd.DataFrame(frequencies).to_csv(root/'answer_option_frequencies.csv',index=False)
    aggregate=[]
    for benchmark,group in frame.groupby('benchmark'):
        a=float(group.original_fp16_sft_accuracy.mean());b=float(group.merged_nf4_sft_accuracy.mean())
        aggregate.append(dict(benchmark=benchmark,languages=len(group),questions=int(group.questions.sum()),
            original_fp16_sft_accuracy=a,merged_nf4_sft_accuracy=b,accuracy_difference_pp=100*(b-a),relative_accuracy_retention=b/a if a else None))
    summary=pd.DataFrame(aggregate);summary.to_csv(root/'accuracy_comparison.csv',index=False)
    selected=json.loads((out/'manifests/final_export.json').read_text())
    audit=json.loads((out/'manifests/sft_contamination_audit.json').read_text())
    storage=pd.DataFrame([dict(variant=s['name'],serialized_bytes=s['stored_bytes'],adapter_bytes=s['adapter_bytes'],remaining_layers=s['layers'],removed_layers=28-s['layers'],architecture_parameters=s.get('architecture_parameters'),trainable_adapter_parameters=s.get('trainable_adapter_parameters')) for s in specs])
    storage.to_csv(root/'post_sft_storage.csv',index=False)
    frame[['benchmark','language','questions','paired_ci95_lower_pp','paired_ci95_upper_pp']].to_csv(root/'paired_confidence_intervals.csv',index=False)
    performance=[]
    for spec in specs:
        for p in (root/spec['name']).glob('performance_*.json'):
            performance.append(dict(kind='benchmark',**json.loads(p.read_text())))
        metrics=out/'sft'/spec['name']/'training_metrics.json'
        if metrics.exists():
            steps=json.loads(metrics.read_text())['exposure']['steps'];seconds=sum(r['seconds'] for r in steps);tokens=sum(r['tokens'] for r in steps)
            performance.append(dict(kind='SFT optimizer steps only',variant=spec['name'],seconds=seconds,new_input_tokens=tokens,
                tokens_per_second=tokens/max(seconds,1e-9),optimizer_steps=len(steps),optimizer_updates_per_second=len(steps)/max(seconds,1e-9),mean_step_seconds=seconds/max(len(steps),1),peak_bytes=max((r['peak_bytes'] for r in steps),default=0)))
    pd.DataFrame(performance).to_csv(root/'performance_summary.csv',index=False)
    from .reporting import plt
    fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(frame))
    ax.bar(x-.18,100*frame.original_fp16_sft_accuracy,.36,label='Original FP16 SFT')
    ax.bar(x+.18,100*frame.merged_nf4_sft_accuracy,.36,label='Merged NF4 SFT')
    ax.set_xticks(x,[r.benchmark+'/'+r.language for r in frame.itertuples()],rotation=40,ha='right');ax.set_ylabel('Accuracy (%)');ax.legend();fig.tight_layout()
    (out/'charts').mkdir(exist_ok=True);fig.savefig(out/'charts/post_sft_accuracy.png',dpi=160);plt.close(fig)
    import shutil
    for name in ('accuracy_comparison.csv','per_language_accuracy.csv','answer_option_frequencies.csv','paired_confidence_intervals.csv','performance_summary.csv'):
        shutil.copyfile(root/name,out/'benchmark_results'/name)
    text=['# Eight-block merger: post-SFT FP16 versus NF4',
          'This comparison includes the combined effects of quantization, merging, and SFT. It does not isolate the effect of layer merging.',
          f"Merger training: {selected['completed_epochs']} epochs, {selected['training_updates']} optimizer updates. Eight weight-attention blocks; old six-block checkpoints are incompatible.",
          f"Pre-SFT internal ranking diagnostic only: original FP16 CE={selected['baseline']['reference_ranking']}; merged NF4 CE={selected['ranking_ce']}; relative change={100*selected['relative_ce_increase']:.3f}%; limit={100*cfg.max_relative_ce_increase:.1f}%. Feasible={selected['feasible']}; CE target met={selected['ce_target_met']}. No benchmark accuracy enters selection.",
          f"Separate full validation: original FP16 CE={selected['baseline']['reference_validation']}; merged NF4 CE={selected['final_nf4_validation_ce']}. These are pre-SFT diagnostics, not final benchmark scores.",
          f"Selected pairs={selected['pairs']}; remaining layers={selected['remaining_layers']}. Search is budget-limited, not a global optimum. Pre-SFT savings vs original 16-bit={100*selected['actual_reduction']:.2f}%; vs original NF4={100*selected['reduction_vs_original_nf4']:.2f}%.",
          f"Both SFT models used {cfg.sft_epochs} complete epochs, rank {cfg.sft_lora_rank}, learning rate {cfg.sft_learning_rate}, the same accepted IDs and response-only labels. Accepted examples/language={audit['examples_per_language']}; tokens/language/epoch={audit['tokens_per_language']}.",
          'Contamination exclusions and provenance: manifests/sft_contamination_audit.json. '+audit['limitations'],
          f"SFT composition and shortfalls: {audit.get('actual_allocation')}; Aya shortfalls: {audit.get('aya_shortfalls')}. Arabic and Albanian benchmark languages received no dedicated SFT.",
          f"Architecture-level parameter reduction: {1-specs[1]['architecture_parameters']/specs[0]['architecture_parameters'] if specs[0].get('architecture_parameters') and specs[1].get('architecture_parameters') else 'unavailable'}. Pre-SFT perplexity ratio: {selected.get('perplexity_ratio')}.",
          'Runtime is a planning target, not a measured completion claim or cutoff. Observed timing, throughput and memory are in performance_summary.csv and SFT preflight/training metrics. Per-language validation losses use the independent SFT validation partition; no benchmark-based checkpoint selection.',
          '![Post-SFT language accuracy](charts/post_sft_accuracy.png)',
          '## Language accuracy',frame.to_csv(index=False),'## Macro aggregate accuracy',summary.to_csv(index=False),
          'Accuracy differences and paired confidence intervals are percentage points (merged minus original); retention is a ratio and undefined for zero baseline accuracy. The five-point accuracy diagnostic is separate from the 13% relative CE constraint.',
          '## Final inference storage',storage.to_csv(index=False),
          f"Post-SFT total storage reduction versus FP16 SFT: {100*(1-specs[1]['stored_bytes']/specs[0]['stored_bytes']):.2f}%. NF4 storage includes the base and separate LoRA adapter; FP16 storage is the merged inference model. Training optimizer checkpoints are excluded from inference size.",
          '![Pre-SFT CE frontier](charts/ce_compression_pareto.png)',
          'Pilot confidence intervals quantify only these paired samples. Lexical/provenance filtering cannot prove absence of semantic or translated contamination. No automatic retraining or benchmark-driven plan changes are performed.']
    text.insert(1,f'{cfg.pilot_questions}-question-per-language pilot: preliminary results, not confirmation of a population-level accuracy threshold.')
    runtime_rows=[json.loads(p.read_text()) for p in sorted((out/'logs/stage_runs').glob('*.json'))]
    pd.DataFrame(runtime_rows).to_csv(root/'stage_runtimes.csv',index=False)
    text.append('Recorded command runtimes (concurrent parent/child stages overlap; do not sum them):\n'+pd.DataFrame(runtime_rows).to_csv(index=False))
    text.append(f'Planning target: {cfg.planning_hours} hours; not a cutoff or a measured-completion claim. GPU utilization and power observations are in each SFT training_metrics.json; unused VRAM does not imply idle compute.')
    path=out/'REPORT.md';path.write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    (out/f'REPORT_{evaluation_namespace(cfg)}.md').write_text(path.read_text(encoding='utf-8'),encoding='utf-8');return path
