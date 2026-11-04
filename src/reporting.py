import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .benchmark_evaluation import read_predictions
from benchmarks.include import LANGUAGE_CODES

TITLE='Multilingual Capability Preservation Through Quantization-Aware Learned Transformer Layer Merging in Qwen3'
SECTIONS=['Research objective and hypothesis','Original GPT-2 merger architecture','Adaptation to Qwen3',
          'Quantization methodology','Wiki-40B data and language coverage','Training configuration','Dual-GPU implementation',
          'Layer-pair selection','MMMLU evaluation','INCLUDE evaluation','MGSM evaluation','Multilingual performance comparison',
          'Compression efficiency','Analysis of language-specific failures','Limitations','Reproducibility information',
          'Conclusions and possible future modifications']

def bootstrap_difference(a,b,seed=42,replicates=2000):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    if a.shape!=b.shape or not len(a):raise ValueError('Require paired nonempty outcomes')
    rng=np.random.default_rng(seed); delta=a-b
    means=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(replicates)])
    return tuple(float(v)*100 for v in np.quantile(means,[.025,.975]))

def legacy_report(cfg):
    out=Path(cfg.output_dir); charts=out/'charts'; charts.mkdir(parents=True,exist_ok=True)
    rows=[]; results_dir=out/'benchmark_results'/cfg.evaluation_mode
    results_dir.mkdir(parents=True,exist_ok=True)
    for path in results_dir.glob('*/*.jsonl'):rows.extend(read_predictions(path))
    frames={}; text={s:'Results pending. No experimental measurements available.' for s in SECTIONS}
    text[SECTIONS[0]]='Test whether the supplied second-block-anchored weight-space merger preserves multilingual capabilities after NF4 source quantization. No benchmark scores select weights or layer pairs.'
    text[SECTIONS[1]]='Six bidirectional pre-LN weight-attention blocks; shared shape encoders/decoders; RMS scaling; learned source, layer, family and patch-position embeddings; matched-patch fusion. Zero correction heads reproduce drop-first.'
    text[SECTIONS[2]]='Runtime-derived Qwen3 families include GQA matrices, Q/K RMSNorm, SwiGLU and both block RMSNorm vectors. Linear weights use output-by-input orientation. See manifests/architecture.json and ARCHITECTURE_AUDIT.md.'
    text[SECTIONS[3]]='Frozen original FP16 teacher. NF4 with double quantization and FP16 compute is applied before extracting source matrices. The entire student training representation is dequantized FP16; generated corrections use FP32 arithmetic and differentiable FP16 casts for decoder execution. C and D are separately labelled. Dense controls share the same dequantized source.'
    text[SECTIONS[5]]='Configuration: config.json and manifests/run.json. Defaults: 10 epochs, seven-hour soft training budget, learning rate 1e-4, KL weight 1, correction weight .01, CE weight 0, T=2. Validation gives each compression level equal weight.'
    text[SECTIONS[6]]='Synchronous replica training uses shard-weighted losses and sums scaled gradients before AMP unscale and clipping. Measurements, when present, are in logs/batch_tuning.csv and logs/training.csv. VRAM allocation is not a compute-utilization measurement.'
    text[SECTIONS[7]]='Raw and guarded additive single-pair rankings select separate nonoverlapping plans. Every selected whole plan is measured independently. Raw learned models are always exported. The guard accepts a complete plan only if ranking CE improves over matching drop-first by the configured margin.'
    text[SECTIONS[14]]='Pilot scores are sampled, non-official few-shot evaluations. MMMLU/INCLUDE test-derived support is excluded from scoring. Context-overflow questions are explicitly skipped for all variants; coverage is reported. Checkpoint validation may exceed the soft time budget. This experiment does not establish generalization to other checkpoints. GPU execution is only verified when runtime preflight manifests exist.'
    text[SECTIONS[15]]='Immutable model/dataset revisions, prompt hashes, source hashes, RNG states and software versions are saved in manifests and checkpoints. Resume requires the same configuration and data. Local tests use synthetic small Qwen3 models; they do not establish NF4 or dual-T4 correctness.'
    training_path=out/'manifests/training_complete.json'
    if training_path.exists():
        outcome=json.loads(training_path.read_text()); state=outcome['state']
        text[SECTIONS[5]]+=f"\n\nRecorded run: {state['step']} optimizer updates, {state['epoch']} completed epochs, {state['seconds']/3600:.3f} hours; stop reason: {outcome['stop_reason']}. Best equal-level validation CE gap: {state['best']}."
    export_path=out/'manifests/exports.json'
    if export_path.exists():
        selected=json.loads(export_path.read_text())
        if selected:
            step=selected[0]['selected_step']; text[SECTIONS[7]]+=f'\n\nAll levels use checkpoint update {step}.'
            if step==0:text[SECTIONS[7]]+=' **The zero-correction initialization remained best; this is not evidence of learned improvement.**'
    plt.rcParams.update({'font.size':10,'figure.dpi':140,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout(); fig.savefig(charts/f'{name}.png',bbox_inches='tight'); fig.savefig(charts/f'{name}.pdf',bbox_inches='tight'); plt.close(fig)
    logpath=out/'logs/training.csv'
    if logpath.exists() and logpath.stat().st_size>2:
        log=pd.read_csv(logpath)
        if len(log):
            fig,ax=plt.subplots(figsize=(9,5))
            for metric in ['loss','hidden_loss','distillation_loss','correction_loss']:ax.plot(log.step,log[metric],label=metric)
            vp=out/'logs/validation.csv'
            if vp.exists():
                val=pd.read_csv(vp); ax.plot(val.step,val.selection_ce_gap,label='validation CE gap',marker='o')
            ax.set(xlabel='Optimizer step',ylabel='Loss / CE gap'); ax.legend(); save(fig,'A_training')
            fig,axes=plt.subplots(2,2,figsize=(11,7))
            for d in (0,1):
                if f'gpu{d}_peak_bytes' in log:
                    axes[0,0].plot(log.step,log[f'gpu{d}_peak_bytes']/2**30,label=f'GPU {d}')
                    axes[1,1].plot(log.step,log[f'gpu{d}_windows'],label=f'GPU {d}')
            axes[0,0].set_ylabel('Peak allocated GiB'); axes[0,0].legend()
            axes[0,1].plot(log.step,log.tokens_per_second); axes[0,1].set_ylabel('Input tokens / second')
            axes[1,0].plot(log.step,log.seconds); axes[1,0].set_ylabel('Update seconds')
            axes[1,1].set_ylabel('Windows processed'); axes[1,1].legend()
            for ax in axes.flat:ax.set_xlabel('Optimizer step')
            save(fig,'H_gpu_performance')
    plans_path=out/'rankings/plans.json'
    if plans_path.exists():
        plans=json.loads(plans_path.read_text()); fig,ax=plt.subplots(figsize=(12,4))
        planrows=[]
        for k in range(1,6):
            pairs=plans[f'raw_{k}']['pairs']
            for layer in range(28):
                merged=any(layer in (p,p+1) for p in pairs)
                ax.add_patch(plt.Rectangle((layer,k-.3),.9,.6,color='#6699cc' if merged else '#dddddd'))
            for p in pairs:ax.plot([p+.45,p+1.45],[k+.35,k+.35],color='black',lw=2)
            planrows.append(dict(level=k,pairs=str(pairs),**{a:b for a,b in plans[f'raw_{k}'].items() if a!='pairs'}))
        ax.set(xlim=(0,28),ylim=(.5,5.6),xlabel='Original layer index',ylabel='Number of merged pairs',yticks=range(1,6))
        save(fig,'G_selected_pairs'); pd.DataFrame(planrows).to_csv(out/'rankings/selected_plans.csv',index=False)
    wiki_path=out/'manifests/wiki40b.json'; wiki=json.loads(wiki_path.read_text()) if wiki_path.exists() else None
    if wiki:
        coverage=[dict(language=l,**r['tokens']) for l,r in wiki['languages'].items()]
        pd.DataFrame(coverage).to_csv(out/'logs/language_tokens.csv',index=False)
        text[SECTIONS[4]]=f"Processed {len(coverage)} Wiki-40B configurations. Missing configs: {json.dumps(wiki['missing'])}. Counts and document IDs are in manifests/wiki40b.json; see logs/language_tokens.csv."
    if rows:
        df=pd.DataFrame(rows); df.to_csv(results_dir/'predictions.csv',index=False)
        scored=df[df.skip_reason.isna()].copy()
        if not len(scored):raise RuntimeError('No scored benchmark examples; inspect context limits')
        summary=scored.groupby(['variant','benchmark','language']).agg(accuracy=('correct','mean'),n=('correct','size'),invalid_rate=('invalid','mean')).reset_index()
        total=df.groupby(['variant','benchmark','language']).size().rename('selected_n').reset_index()
        summary=summary.merge(total); summary['coverage']=summary.n/summary.selected_n
        def training_language(row):
            if row.benchmark=='MMMLU':return row.language.split('_')[0].lower()
            if row.benchmark=='INCLUDE':return LANGUAGE_CODES.get(row.language,row.language)
            return row.language
        summary['wiki_language']=summary.apply(training_language,axis=1)
        summary['seen_in_wiki']=summary.wiki_language.map(lambda x:any(l in wiki['languages'] for l in (['zh-cn','zh-tw','zh'] if x=='zh' else [x])) if wiki else False)
        summary.to_csv(results_dir/'per_language.csv',index=False)
        aggregates=[]
        canonical=[]
        for (variant,benchmark),g in summary.groupby(['variant','benchmark']):
            language_values=[]
            for language,lg in g.groupby('wiki_language'):
                value=np.average(lg.accuracy,weights=lg.n); language_values.append(value)
                canonical.append(dict(variant=variant,benchmark=benchmark,language=language,accuracy=value,n=int(lg.n.sum())))
            aggregates.append(dict(variant=variant,benchmark=benchmark,macro_accuracy=np.mean(language_values),
                                   weighted_accuracy=np.average(g.accuracy,weights=g.n),languages=len(language_values),configurations=len(g),n=int(g.n.sum())))
        pd.DataFrame(canonical).to_csv(results_dir/'canonical_language_scores.csv',index=False)
        aggregate=pd.DataFrame(aggregates); aggregate.to_csv(results_dir/'aggregate.csv',index=False)
        comparisons=[]; paired_groups={}
        for (variant,benchmark,language),g in scored.groupby(['variant','benchmark','language']):
            for baseline in ('original_fp16','original_nf4'):
                b=scored[(scored.variant==baseline)&(scored.benchmark==benchmark)&(scored.language==language)]
                if not len(b):continue
                paired=g.merge(b,on=['id','prompt_hash'],suffixes=('_model','_base'))
                if len(paired)!=len(g) or len(paired)!=len(b):raise RuntimeError('Incomplete paired evaluation; finish variants before reporting')
                a=paired.correct_model.astype(float).to_numpy(); base=paired.correct_base.astype(float).to_numpy()
                canonical_language=LANGUAGE_CODES.get(language,language) if benchmark=='INCLUDE' else language
                paired_groups.setdefault((variant,benchmark,baseline),{}).setdefault(canonical_language,[]).append(a-base)
                lo,hi=bootstrap_difference(a,base,cfg.seed,cfg.bootstrap_samples)
                comparisons.append(dict(variant=variant,benchmark=benchmark,language=language,baseline=baseline,
                                        delta_pp=100*(a.mean()-base.mean()),retention=a.mean()/base.mean() if base.mean() else None,
                                        ci_low_pp=lo,ci_high_pp=hi,n=len(a)))
        comparisons=pd.DataFrame(comparisons); comparisons.to_csv(results_dir/'paired_differences.csv',index=False)
        aggregate_intervals=[]
        for (variant,benchmark,baseline),language_groups in paired_groups.items():
            groups=[np.concatenate(arrays) for arrays in language_groups.values()]
            rng=np.random.default_rng(cfg.seed); simulations=[]
            for _ in range(cfg.bootstrap_samples):
                simulations.append(np.mean([g[rng.integers(0,len(g),len(g))].mean() for g in groups]))
            lo,hi=np.quantile(simulations,[.025,.975])*100
            aggregate_intervals.append(dict(variant=variant,benchmark=benchmark,baseline=baseline,languages=len(groups),
                macro_delta_pp=100*np.mean([g.mean() for g in groups]),ci_low_pp=lo,ci_high_pp=hi,
                protocol='paired within language; equal language weights; fixed evaluated languages'))
        pd.DataFrame(aggregate_intervals).to_csv(results_dir/'macro_paired_differences.csv',index=False)
        for category in ('subject','domain','regional'):
            scored.groupby(['variant','benchmark','language',category]).agg(accuracy=('correct','mean'),n=('correct','size')).to_csv(results_dir/f'by_{category}.csv')
        performance=[]
        for path in results_dir.glob('*/performance_gpu*.json'):performance.append(json.loads(path.read_text()))
        perf=pd.DataFrame(performance)
        if len(perf):
            original=int(perf[perf.variant=='original_fp16'].parameters.iloc[0]) if any(perf.variant=='original_fp16') else None
            perf['original_parameters']=original; perf['removed_layers']=28-perf.layers
            perf['parameter_reduction_pct']=100*(1-perf.parameters/original) if original else None
            perf.to_csv(results_dir/'performance.csv',index=False)
        for benchmark in sorted(aggregate.benchmark.unique()):
            a=aggregate[aggregate.benchmark==benchmark]; fig,ax=plt.subplots(figsize=(8,5))
            for suffix in ('fp16','nf4'):
                g=a[a.variant.str.match(r'learned_\d_'+suffix)].copy()
                if len(g):
                    g['layers']=g.variant.map(lambda s:28-int(s.split('_')[1])); g=g.sort_values('layers')
                    ax.plot(g.layers,100*g.macro_accuracy,marker='o',label=f'Learned {suffix}')
                baseline=a[a.variant=='original_'+suffix]
                if len(baseline):ax.axhline(100*baseline.macro_accuracy.iloc[0],ls='--',label='Original '+suffix)
            ax.set(title=benchmark+' — '+cfg.evaluation_mode,xlabel='Remaining decoder layers',ylabel='Macro accuracy (%)'); ax.legend()
            save(fig,'B_F_compression_'+benchmark)
            if len(comparisons):
                c=comparisons[(comparisons.benchmark==benchmark)&(comparisons.baseline=='original_nf4')&comparisons.variant.str.match(r'learned_\d_fp16')]
                if len(c):
                    pivot=c.pivot(index='language',columns='variant',values='delta_pp'); pivot.to_csv(charts/f'C_{benchmark}_data.csv')
                    fig,ax=plt.subplots(figsize=(8,max(5,len(pivot)*.24)))
                    bound=max(float(np.abs(pivot.to_numpy()).max()),1)
                    im=ax.imshow(pivot,aspect='auto',cmap='RdBu',vmin=-bound,vmax=bound)
                    ax.set_yticks(range(len(pivot)),pivot.index); ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=45,ha='right')
                    ax.set_title(benchmark+' accuracy change vs NF4 baseline'); fig.colorbar(im,ax=ax,label='Percentage points'); save(fig,'C_heatmap_'+benchmark)
            language_scores=summary[summary.benchmark==benchmark].pivot(index='language',columns='variant',values='accuracy')
            cols=[c for c in language_scores if c=='original_fp16' or c.startswith('learned_') and c.endswith('fp16')]
            if cols:
                ax=(100*language_scores[cols]).plot(kind='bar',figsize=(max(9,len(language_scores)*.4),5)); ax.set_ylabel('Accuracy (%)'); save(ax.figure,'D_languages_'+benchmark)
            if len(perf) and 'stored_bytes' in perf:
                sizes=perf[perf.gpu==0][['variant','stored_bytes']].drop_duplicates('variant'); points=a.merge(sizes,on='variant')
                fig,ax=plt.subplots(figsize=(9,5)); ax.scatter(points.stored_bytes/2**20,100*points.macro_accuracy)
                for r in points.itertuples():ax.annotate(r.variant,(r.stored_bytes/2**20,100*r.macro_accuracy),fontsize=6)
                ax.set(xlabel='Measured stored size (MiB)',ylabel='Macro accuracy (%)',title=benchmark); save(fig,'E_efficiency_'+benchmark)
        for i,b in zip((8,9,10),('MMMLU','INCLUDE','MGSM')):
            text[SECTIONS[i]]='```\n'+aggregate[aggregate.benchmark==b].to_string(index=False)+'\n```\n\nPer-language scores, coverage and paired 95% intervals are in benchmark_results/.'
        text[SECTIONS[11]]='See per_language.csv, aggregate.csv and paired_differences.csv. Macro-average is primary; weighted aggregate is also reported. Differences are percentage points; retention is a ratio and undefined when baseline accuracy is zero.'
        text[SECTIONS[12]]='Actual serialized sizes, dense logical parameter counts, peak inference VRAM and measured tokens/sec are in performance.csv. Requantization and layer removal are separate operations. Throughput covers newly executed examples only; repeated resume calls are not full-run timing comparisons.'
        if len(comparisons):
            failures=comparisons[comparisons.variant.str.startswith('learned_')].sort_values('delta_pp').head(20)
            text[SECTIONS[13]]='Largest observed decreases (not causal explanations):\n\n```\n'+failures.to_string(index=False)+'\n```'
        text[SECTIONS[16]]='Evaluate effect sizes together with paired uncertainty intervals and individual languages. These outputs do not automatically establish that the architecture succeeds. Architectural changes should follow inspection of completed raw learned results.'
    status='Results are pending; training and benchmark execution have not been established.' if not rows else 'This report contains the available recorded predictions; inspect coverage and completion before drawing conclusions.'
    report=f'# {TITLE}\n\n{status}\n\n'
    for i,s in enumerate(SECTIONS,1):report+=f'## {i}. {s}\n\n{text[s]}\n\n'
    report+='## Figures\n\n'+''.join(f'![{p.stem}](charts/{p.name})\n\n' for p in sorted(charts.glob('*.png')))
    (out/f'REPORT_{cfg.evaluation_mode}.md').write_text(report,encoding='utf-8')
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    return out/'REPORT.md'


def make_report(cfg):
    """Summarize one CE-selected NF4 model and its paired NF4 reference."""
    out=Path(cfg.output_dir)
    selected=json.loads((out/'rankings/selected_plan.json').read_text())
    manifest=json.loads((out/f'manifests/benchmark_prompts_{cfg.evaluation_mode}.json').read_text())
    root=out/'benchmark_results'/cfg.evaluation_mode
    records={}
    for variant_name in ('original_nf4','best_learned_nf4'):
        rows=[]
        for job in manifest['jobs']:
            path=root/variant_name/f"{job['benchmark']}_{job['language']}.jsonl"
            loaded=read_predictions(path)
            expected={e['id']:e for e in job['examples']}
            if len(loaded)!=len(expected) or {r['id'] for r in loaded}!=set(expected):
                raise RuntimeError(f'Incomplete benchmark: {path}')
            for r in loaded:
                e=expected[r['id']]
                if any(r[k]!=e[k] for k in ('prompt_hash','support_ids','target','skip_reason')):
                    raise RuntimeError('Prediction prompt/protocol mismatch')
            rows+=loaded
        records[variant_name]={(r['benchmark'],r['language'],r['id']):r for r in rows}
    reference=records['original_nf4']; learned=records['best_learned_nf4']
    if reference.keys()!=learned.keys():raise RuntimeError('Unpaired benchmark rows')
    comparisons=[]; frequencies=[]
    for benchmark,language in sorted({k[:2] for k in reference}):
        keys=[k for k in reference if k[:2]==(benchmark,language) and not reference[k]['skip_reason']]
        if not keys:raise RuntimeError(f'No scored questions: {benchmark}/{language}')
        a=float(np.mean([reference[k]['correct'] for k in keys])); b=float(np.mean([learned[k]['correct'] for k in keys]))
        comparisons.append(dict(benchmark=benchmark,language=language,questions=len(keys),
                                original_nf4_correct=sum(reference[k]['correct'] for k in keys),learned_nf4_correct=sum(learned[k]['correct'] for k in keys),
                                original_invalid=sum(reference[k]['invalid'] for k in keys),learned_invalid=sum(learned[k]['invalid'] for k in keys),
                                original_truncated=sum(reference[k].get('truncated',False) for k in keys),learned_truncated=sum(learned[k].get('truncated',False) for k in keys),
                                original_nf4_accuracy=a,learned_nf4_accuracy=b,accuracy_loss_pp=100*(a-b),
                                relative_loss_percent=100*(a-b)/a if a else None,
                                observed_target_met=a-b<=cfg.accuracy_loss_target))
        if benchmark!='MGSM':
            from collections import Counter
            for name,rows in records.items():
                counts=Counter(rows[k]['prediction'] for k in keys)
                for option,count in sorted(counts.items(),key=lambda x:str(x[0])):
                    frequencies.append(dict(benchmark=benchmark,language=language,variant=name,
                                            option=chr(65+option) if isinstance(option,int) else 'invalid',
                                            count=count,frequency=count/len(keys)))
    frame=pd.DataFrame(comparisons); frame.to_csv(root/'per_language_accuracy.csv',index=False)
    pd.DataFrame(frequencies).to_csv(root/'answer_option_frequencies.csv',index=False)
    aggregate=[]
    for benchmark,group in frame.groupby('benchmark'):
        a=float(group.original_nf4_accuracy.mean()); b=float(group.learned_nf4_accuracy.mean())
        aggregate.append(dict(benchmark=benchmark,original_nf4_accuracy=a,learned_nf4_accuracy=b,
                              accuracy_loss_pp=100*(a-b),relative_loss_percent=100*(a-b)/a if a else None,
                              observed_target_met=a-b<=cfg.accuracy_loss_target))
    summary=pd.DataFrame(aggregate); summary.to_csv(root/'accuracy_comparison.csv',index=False)
    base=selected['baseline']
    training=json.loads((out/'logs/training_metrics.json').read_text())
    coverage=json.loads((out/'manifests/language_coverage.json').read_text())
    sampled=[r for r in coverage if r['sampled']]
    text=["# Five-epoch multilingual Qwen3 NF4 comparison",
          f"Completed epochs: {selected['completed_epochs']}; optimizer updates: {selected['training_updates']}; training seconds: {training['state']['seconds']:.2f}.",
          f"Actual processed training tokens by language: {training['actual_training_tokens_per_language']}. Loss, teacher/student CE, distillation and correction metrics: logs/training_metrics.json.",
          f"Sampled sources: {[(r['language'],r['source'],r['config']) for r in sampled]}. Availability and exact/proxy/unavailable status for the official 119 entries (plus requested Yoruba) are in manifests/language_coverage.json. Availability is not a claim that all languages were trained.",
          f"Selected original-layer pairs: {[(p,p+1) for p in selected['pairs']]}; remaining layers: {selected['remaining_layers']}.",
          f"Selection: {selected['selection']}. Measured {selected['searched_plans']} surrogate plans and {selected['nf4_finalists']} NF4 finalists; no global-optimum claim.",
          f"Original NF4 ranking CE: {base['nf4_ranking']:.6f}; selected NF4 ranking CE: {selected['ranking_ce']:.6f}; delta: {selected['ce_increase_original']:.6f}.",
          f"Original NF4 validation CE: {base['nf4_validation']:.6f}; selected NF4 validation CE: {selected['final_nf4_validation_ce']:.6f}; delta: {selected['final_nf4_ce_increase']:.6f}.",
          f"Final serialized bytes: {selected['serialized_bytes']}; original NF4 bytes: {base['nf4_serialized_bytes']}; original FP16-width reference bytes: {base['reference_storage']['bytes']}.",
          f"Total saving vs FP16: {100*selected['actual_reduction']:.2f}%; additional saving vs original NF4: {100*selected['reduction_vs_original_nf4']:.2f}%. Quantization savings are not attributed to the merger.",
          f"At least 40% storage target met: {selected['storage_target_met']}; at least 50%: {selected['target_50_met']}. Savings above 50% are acceptable.",
          'FP16 storage reference: '+base['reference_storage']['basis']+'.',
          selected['format']+'. Both benchmark models use the same NF4 double-quantization/FP16-compute policy. FP16 teacher is restricted to training.',
          '## Pilot accuracy (macro over selected languages)',summary.to_csv(index=False),
          f"Observed five-percentage-point target met on every benchmark: {all(r['observed_target_met'] for r in aggregate)}. Negative degradation denotes improvement.",
          'Per-language correct counts, accuracies, absolute/relative degradation, invalid and truncated counts: per_language_accuracy.csv. Answer-option frequencies reuse the same predictions.',
          'Limitations: ten-question language samples are preliminary and cannot establish a five-point threshold scientifically. CE does not guarantee accuracy; no acceptable performance is claimed solely from CE. Large losses are reported without benchmarking more candidates. Coverage proxies do not establish dedicated dialect training.']
    report=out/'REPORT.md'; report.write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    return report
