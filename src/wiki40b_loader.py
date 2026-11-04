"""Wiki-40B TRAIN only; article IDs define partitions globally across languages."""
import hashlib
import re
from pathlib import Path
import numpy as np
import torch
from datasets import get_dataset_config_names, load_dataset
from .utils import atomic_json, atomic_torch, digest

def clean_article(text):
    if not isinstance(text,str):return ''
    # Marker lines are structure; the title, section text and Unicode content remain.
    text=re.sub(r'(?m)^_(?:START_ARTICLE|START_SECTION|START_PARAGRAPH)_\s*$', '', text)
    return text.replace('_NEWLINE_','\n').strip()

def document_partition(document_id,seed):
    bucket=int(hashlib.sha256(f'{seed}:{document_id}'.encode()).hexdigest()[:8],16)%100
    return 'validation' if bucket<10 else 'ranking' if bucket<20 else 'train'

def prepare_data(cfg,tokenizer,revision):
    from .multilingual_data import train_configs,coverage,SMOKE_LANGUAGES,MAPPING_PATH
    from .utils import file_digest
    import json
    revisions=revision if isinstance(revision,dict) else {cfg.wiki_dataset:revision,cfg.fineweb_dataset:cfg.fineweb_revision}
    out=Path(cfg.output_dir); path=out/'manifests/training_data.json'; cache=out/'training_tokens.pt'
    coverage_path=out/'manifests/language_coverage.json'
    settings=dict(revisions={d:revisions[d] for d in (cfg.wiki_dataset,cfg.fineweb_dataset)},backend=cfg.wiki_backend,
                  seed=cfg.seed,languages=list(cfg.training_languages),replacements=cfg.language_replacements,
                  budgets=[cfg.train_tokens_per_language,cfg.validation_tokens_per_language,cfg.ranking_tokens_per_language],
                  tokenizer=tokenizer.name_or_path,seq_len=cfg.train_seq_len,buffer=cfg.wiki_shuffle_buffer,
                  max_articles=cfg.max_articles_per_language,eos=tokenizer.eos_token_id,mapping_sha256=file_digest(MAPPING_PATH))
    settings=json.loads(json.dumps(settings))
    if path.exists() and cache.exists():
        manifest=json.loads(path.read_text(encoding='utf-8'))
        if manifest['settings']!=settings:raise RuntimeError('Training cache settings differ; use a fresh output directory')
        data=torch.load(cache,weights_only=True)
        if digest(data)!=manifest['token_digest']:raise RuntimeError('Training cache checksum mismatch')
        atomic_json(manifest['coverage'],coverage_path)
        return data,manifest
    if cfg.wiki_backend=='tfds':
        from .wiki40b_tfds import configs as tfds_configs
        wiki_configs=tfds_configs()
    else:wiki_configs=train_configs(cfg.wiki_dataset,revisions[cfg.wiki_dataset])
    fineweb_configs=train_configs(cfg.fineweb_dataset,revisions[cfg.fineweb_dataset])
    report=coverage(wiki_configs,fineweb_configs,cfg); byname={r['language']:r for r in report}
    languages=list(cfg.training_languages) or (list(SMOKE_LANGUAGES) if cfg.mode=='smoke' else [r['language'] for r in report if r['coverage_status']=='Exact coverage'])
    data={}; records={}; missing={}; seen_content=set(); loaded_sources=set()
    budgets=dict(zip(('train','validation','ranking'),settings['budgets']))
    global_ids={k:set() for k in budgets}
    for requested in languages:
        language=(cfg.language_replacements or {}).get(requested,requested)
        if language not in byname:raise ValueError(f'Unmapped requested language: {language}')
        entry=byname[language]; source,config=entry['source'],entry['config']
        entry['requested_tokens']=budgets.copy()
        if requested!=language:entry['documented_replacement_for']=requested
        atomic_json(report,coverage_path)
        if not source:
            raise RuntimeError(f'{requested}: unavailable. Set an explicit language_replacements entry; no silent substitution.')
        if (source,config) in loaded_sources:raise ValueError('Two requested names map to the same source subset; avoid duplicate proxy sampling')
        loaded_sources.add((source,config))
        tokens={k:[] for k in budgets}; docs={k:[] for k in budgets}; seen_ids=set()
        try:
            if source==cfg.wiki_dataset and cfg.wiki_backend=='tfds':
                from .wiki40b_tfds import rows as tfds_rows
                rows=tfds_rows(config,cfg.seed,cfg.wiki_shuffle_buffer)
            else:
                rows=load_dataset(source,name=config,split='train',streaming=True,revision=revisions[source])
                rows=rows.shuffle(seed=cfg.seed,buffer_size=cfg.wiki_shuffle_buffer)
            for row_index,row in enumerate(rows):
                if row_index>=cfg.max_articles_per_language:break
                text=clean_article(row.get('text')) if source==cfg.wiki_dataset else row.get('text','')
                if not isinstance(text,str) or not text.strip():continue
                text_hash=hashlib.sha256(re.sub(r'\s+',' ',text).strip().encode()).hexdigest()
                if text_hash in seen_content:continue
                identity=row.get('wikidata_id') if source==cfg.wiki_dataset else (row.get('id') or row.get('url'))
                doc_id=source+':'+str(identity or text_hash)
                if doc_id in seen_ids:continue
                seen_ids.add(doc_id)
                partition=document_partition(doc_id,cfg.seed)
                if len(tokens[partition])>=budgets[partition]:continue
                ids=tokenizer.encode(text,add_special_tokens=False)
                if not ids:continue
                seen_content.add(text_hash)
                ids=(ids+[tokenizer.eos_token_id])[:budgets[partition]-len(tokens[partition])]
                start=len(tokens[partition]); tokens[partition].extend(ids)
                docs[partition].append(dict(id=doc_id,sha256=text_hash,start=start,end=len(tokens[partition])))
                global_ids[partition].add(doc_id)
                if all(len(tokens[k])>=v for k,v in budgets.items()):break
            if any(len(tokens[k])<v for k,v in budgets.items()):raise RuntimeError('Insufficient independent documents to meet token budgets')
            if len(tokens['train'])<cfg.train_seq_len:raise RuntimeError('No complete training window')
            data[language]=tokens
            records[language]=dict(source=source,configuration=config,revision=revisions[source],script=entry['script'],
                                   tokens={k:len(v) for k,v in tokens.items()},documents=docs,
                                   training_tail_tokens=len(tokens['train'])%cfg.train_seq_len)
            entry.update(sampled=True,actual_train_tokens=len(tokens['train']),actual_validation_tokens=len(tokens['validation']),actual_ranking_tokens=len(tokens['ranking']))
            print(language,source,config,records[language]['tokens'],flush=True)
        except Exception as error:
            missing[language]=repr(error); entry['sampling_error']=repr(error)
            atomic_json(report,coverage_path)
            atomic_json(dict(settings=settings,languages=records,missing=missing),out/'manifests/training_data_progress.json')
            if not cfg.allow_missing_languages:raise RuntimeError(f'{source}/{config} failed; no silent corpus substitution') from error
    if not data:raise RuntimeError('No training language loaded')
    for a,b in [('train','validation'),('train','ranking'),('validation','ranking')]:
        if global_ids[a]&global_ids[b]:raise AssertionError('Document-ID leakage')
    manifest=dict(settings=settings,coverage=report,languages=records,missing=missing,token_digest=digest(data),
                  document_partition='source/document identity hash 80/10/10; global normalized-content deduplication')
    atomic_torch(data,cache); atomic_json(manifest,path); atomic_json(report,coverage_path)
    return data,manifest


def balanced_epoch(data,seq_len,seed,epoch):
    """Round robin randomized language queues, no duplicated windows or padded updates."""
    rng=np.random.default_rng(seed+epoch); queues={}
    for lang,parts in sorted(data.items()):
        windows=torch.tensor(parts['train'][:len(parts['train'])//seq_len*seq_len]).reshape(-1,seq_len)
        queues[lang]=[windows[i] for i in rng.permutation(len(windows))]
    result=[]
    while any(queues.values()):
        for lang in rng.permutation(sorted(queues)):
            if queues[lang]:result.append((lang,queues[lang].pop()))
    return result
