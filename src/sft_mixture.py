"""Audited Aya + independent verified numeric curricula, with disjoint validation."""
import json
from collections import Counter
from pathlib import Path
from datasets import load_dataset
from .sft_data import LANGUAGES,PROTOCOL,ContaminationIndex,normalized,response_tokens,load_prepared
from .sft_generated import generate,VERSION
from .utils import digest,atomic_json


def allocation(n,instruction=None):
    instruction=round(.4*n) if instruction is None else instruction
    math=round((n-instruction)*35/60)
    return dict(instruction=instruction,math=math,mc=n-instruction-math)


def prepare_mixture(cfg,lock,tokenizer,manifest):
    out=Path(cfg.output_dir);path=out/'manifests/sft_contamination_audit.json'
    if path.exists() and json.loads(path.read_text()).get('complete'):return load_prepared(cfg,lock,manifest)
    if manifest['signature']!=digest(dict(protocol=manifest['protocol'],jobs=manifest['jobs'])):raise RuntimeError('Changed benchmark audit manifest')
    texts=[]
    for job in manifest['jobs']:
        if not job['examples']:raise RuntimeError('Empty benchmark job')
        for e in job['examples']:
            if not e.get('audit_texts') or not e.get('support_ids'):raise RuntimeError('Missing audit question/answer/demonstration material')
            texts+=e['audit_texts']
    index=ContaminationIndex(texts,cfg.sft_near_duplicate_threshold)
    langs=list(cfg.training_languages)
    if set(langs)-LANGUAGES.keys():raise RuntimeError('Unverified language')
    sizes=dict(train=cfg.sft_examples_per_language,validation=cfg.sft_validation_examples_per_language)
    target={split:allocation(n) for split,n in sizes.items()}
    records={split:{l:[] for l in langs} for split in sizes};rejected=[];seen=set();counts={}
    identity=digest([lock,manifest['signature'],PROTOCOL])
    progress=out/'sft/preparation_progress.json'
    if progress.exists():
        state=json.loads(progress.read_text(encoding='utf-8'))
        if state.get('payload_digest')!=digest({k:v for k,v in state.items() if k!='payload_digest'}):raise RuntimeError('Corrupt SFT preparation checkpoint')
        if state['identity']!=identity:raise RuntimeError('Stale SFT preparation checkpoint')
        records=state['records'];rejected=state['rejected'];seen=set(state['seen'])
    def save_progress():
        state=dict(identity=identity,records=records,rejected=rejected,seen=sorted(seen))
        atomic_json(dict(state,payload_digest=digest(state)),progress)
    def add(row,language,split,source,category):
        prompt=str(row['inputs']);response=str(row['targets'])
        rid=digest([source,language,prompt,response]);question_id=digest(normalized(prompt));problem=row.get('problem_id',question_id)
        reason=None
        if rid in seen or question_id in seen or problem in seen:return False
        if category=='instruction':
            import re
            provenance=' '.join(str(row.get(k,'')) for k in ('source','dataset','provenance','annotation_type'))
            if row.get('annotation_type')!='original_annotations':reason='non_original_or_unknown_provenance'
            elif re.search(r'mgsm|gsm8k|mmmlu|\bmmlu\b|include[-_ ]?(lite|44|benchmark)',provenance+' '+prompt,re.I):reason='benchmark_source_marker'
            elif min(len(normalized(prompt)),len(normalized(response)))<12 or normalized(prompt)==normalized(response):reason='quality_or_incomplete'
        reason=reason or index.match(prompt)
        # A/B/C/D labels alone are not evidence of contamination; audit the full MC question/options.
        if category!='mc':reason=reason or index.match(response) or index.match(prompt+' '+response)
        encoded=None if reason else response_tokens(tokenizer,prompt,response,cfg.sft_max_length)
        if not reason and encoded is None:reason='context_or_empty_response'
        if not reason and split=='train' and sum(len(r['input_ids']) for r in records[split][language])+len(encoded['input_ids'])>cfg.sft_tokens_per_language:reason='token_budget'
        if reason:rejected.append(dict(id=rid,language=language,split=split,source=source,reason=reason));return False
        seen.update((rid,question_id,problem))
        records[split][language].append(dict(id=rid,language=language,source=source,category=category,problem_id=problem,
            label=row.get('label'),subject=row.get('subject','instruction'),prompt_hash=digest(prompt),response_hash=digest(response),
            verified_answer=row.get('verified_answer'),operands=row.get('operands'),**encoded))
        return True
    audit=dict(identity=identity,complete=False,benchmark_signature=manifest['signature'],protocol=PROTOCOL,
        dataset=cfg.sft_dataset,revision=lock['dataset_revisions'][cfg.sft_dataset],generated_source=VERSION,
        benchmark_text_digest=digest(index.texts),near_duplicate_threshold=cfg.sft_near_duplicate_threshold,
        limitations='Audit addresses detectable added-SFT overlap only, not historical pretraining or undetectable semantic/translated contamination. Generated academic data covers numeric subjects, not broad factual knowledge.',
        benchmark_languages_without_dedicated_sft=['Arabic','Albanian'],requested_allocation=target)
    atomic_json(audit,path)
    try:
        codes={LANGUAGES[l]:l for l in langs}
        stream=load_dataset(cfg.sft_dataset,'default',split='train',revision=audit['revision'],streaming=True)
        for number,row in enumerate(stream.shuffle(seed=cfg.seed,buffer_size=1000)):
            language=codes.get(row.get('language_code'))
            if language:
                for split in sizes:
                    if sum(r['category']=='instruction' for r in records[split][language])<target[split]['instruction']:
                        add(row,language,split,cfg.sft_dataset+'@'+audit['revision'],'instruction');break
            if number%1000==0:save_progress()
            if all(sum(r['category']=='instruction' for r in records[s][l])>=target[s]['instruction'] for s in sizes for l in langs):break
        for split,n in sizes.items():
            for language in langs:
                actual_aya=sum(r['category']=='instruction' for r in records[split][language])
                quotas=allocation(n,actual_aya)  # Transparently redistribute an Aya shortfall to verified numeric tasks.
                for category in ('math','mc'):
                    accepted=sum(r['category']==category for r in records[split][language])
                    for attempt in range(cfg.sft_max_generated_attempts):
                        if accepted>=quotas[category]:break
                        idx=attempt+(0 if split=='train' else cfg.sft_max_generated_attempts)
                        row=generate(language,category,idx,cfg.seed,label_index=accepted)
                        if add(row,language,split,VERSION,category):accepted+=1
                        if attempt%200==0:save_progress()
                    if accepted<quotas[category]:raise RuntimeError(f'SFT shortfall: {split}/{language}/{category} {accepted}/{quotas[category]}')
                counts[split+'/'+language]=dict(Counter(r['category'] for r in records[split][language]))
                if len(records[split][language])!=n:raise RuntimeError('SFT quota mismatch')
        data=dict(benchmark_signature=manifest['signature'],examples=[r for l in langs for r in records['train'][l]],
                  validation=[r for l in langs for r in records['validation'][l]])
        if {r['problem_id'] for r in data['examples']}&{r['problem_id'] for r in data['validation']}:raise RuntimeError('SFT partition leakage')
        audit.update(complete=True,accepted_digest=digest(data),accepted_ids=[r['id'] for r in data['examples']],
            examples_per_language={l:len(records['train'][l]) for l in langs},validation_examples_per_language={l:len(records['validation'][l]) for l in langs},
            tokens_per_language={l:sum(len(r['input_ids']) for r in records['train'][l]) for l in langs},
            validation_tokens_per_language={l:sum(len(r['input_ids']) for r in records['validation'][l]) for l in langs},
            actual_allocation=counts,aya_shortfalls={s+'/'+l:target[s]['instruction']-counts[s+'/'+l].get('instruction',0) for s in sizes for l in langs},
            mc_label_counts={s+'/'+l:dict(Counter(chr(65+r['label']) for r in records[s][l] if r['category']=='mc')) for s in sizes for l in langs},
            exclusions=rejected,exclusion_counts=dict(Counter(r['reason'] for r in rejected)))
        atomic_json(data,out/'sft/accepted_examples.json');atomic_json(audit,path)
        atomic_json(audit,out/'sft/contamination_audit.json');atomic_json(rejected,out/'sft/rejected_examples.json')
        atomic_json(audit['accepted_ids'],out/'sft/accepted_training_ids.json')
        atomic_json([r['id'] for r in data['validation']],out/'sft/accepted_validation_ids.json')
        atomic_json({k:v for k,v in audit.items() if k not in ('exclusions','accepted_ids')},out/'sft/dataset_manifest.json')
        save_progress();return data,audit
    except BaseException as error:
        save_progress();audit.update(error=repr(error),exclusions=rejected,counts=counts)
        atomic_json(audit,path);atomic_json(audit,out/'sft/contamination_audit.json');raise
