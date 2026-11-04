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
    out=Path(cfg.output_dir); path=out/'manifests/wiki40b.json'; cache=out/'wiki40b.pt'
    settings=dict(dataset=cfg.wiki_dataset,backend=cfg.wiki_backend,revision=revision,seed=cfg.seed,languages=list(cfg.wiki_languages),
                  budgets=[cfg.train_tokens_per_language,cfg.validation_tokens_per_language,cfg.ranking_tokens_per_language],
                  tokenizer=tokenizer.name_or_path,seq_len=cfg.train_seq_len,buffer=cfg.wiki_shuffle_buffer,
                  max_articles=cfg.max_articles_per_language,eos=tokenizer.eos_token_id)
    if path.exists() and cache.exists():
        import json
        manifest=json.loads(path.read_text(encoding='utf-8'))
        if manifest['settings']!=settings:raise RuntimeError('Wiki cache settings differ; use a fresh output directory')
        data=torch.load(cache,weights_only=True)
        if digest(data)!=manifest['token_digest']:raise RuntimeError('Wiki cache checksum mismatch')
        return data,manifest
    if cfg.wiki_backend=='tfds':
        from .wiki40b_tfds import configs as tfds_configs
        configs=tfds_configs()
    else:configs=get_dataset_config_names(cfg.wiki_dataset,revision=revision)
    languages=list(cfg.wiki_languages) or [x for x in configs if x not in ('all','default')]
    if not languages:raise RuntimeError('No Wiki-40B language configs discovered')
    data={}; records={}; missing={}; content_owner={}
    budgets=dict(zip(('train','validation','ranking'),settings['budgets']))
    global_ids={k:set() for k in budgets}
    for language in languages:
        tokens={k:[] for k in budgets}; docs={k:[] for k in budgets}; seen=set(); seen_ids=set()
        try:
            if cfg.wiki_backend=='tfds':
                from .wiki40b_tfds import rows as tfds_rows
                rows=tfds_rows(language,cfg.seed,cfg.wiki_shuffle_buffer)
            else:
                rows=load_dataset(cfg.wiki_dataset,language,split='train',streaming=True,revision=revision)
                rows=rows.shuffle(seed=cfg.seed,buffer_size=cfg.wiki_shuffle_buffer)
            for row_index,row in enumerate(rows):
                if row_index>=cfg.max_articles_per_language:break
                text=clean_article(row.get('text')); text_hash=hashlib.sha256(text.encode()).hexdigest()
                if not text or text_hash in seen:continue
                seen.add(text_hash)
                doc_id=str(row.get('wikidata_id') or text_hash)
                if doc_id in seen_ids:continue
                seen_ids.add(doc_id)
                partition=document_partition(doc_id,cfg.seed)
                # Skip exact content duplicates that have conflicting document IDs/partitions.
                if text_hash in content_owner and content_owner[text_hash]!=partition:continue
                if len(tokens[partition])>=budgets[partition]:continue
                ids=tokenizer.encode(text,add_special_tokens=False)
                if not ids:continue
                content_owner[text_hash]=partition
                remaining=budgets[partition]-len(tokens[partition])
                ids=(ids+[tokenizer.eos_token_id])[:remaining]
                start=len(tokens[partition]); tokens[partition].extend(ids)
                docs[partition].append(dict(id=doc_id,sha256=text_hash,start=start,end=len(tokens[partition])))
                global_ids[partition].add(doc_id)
                if all(len(tokens[k])>=v for k,v in budgets.items()):break
            if any(len(tokens[k])<2 for k in budgets):raise RuntimeError('Insufficient independent documents for three partitions')
            if len(tokens['train'])<cfg.train_seq_len:raise RuntimeError('No complete training window')
            data[language]=tokens
            records[language]=dict(tokens={k:len(v) for k,v in tokens.items()},documents=docs,
                                   training_tail_tokens=len(tokens['train'])%cfg.train_seq_len)
            print(language,records[language]['tokens'],flush=True)
        except Exception as error:
            missing[language]=repr(error)
            atomic_json(dict(settings=settings,languages=records,missing=missing),out/'manifests/wiki40b_progress.json')
            if not cfg.allow_missing_languages:
                raise RuntimeError(f'Wiki-40B {language} failed. Check downloads/config; no corpus substitution. '
                                   'Use wiki_backend="tfds" with the official TFDS release in a fresh run, or '
                                   'set allow_missing_languages explicitly to record and omit unavailable configs.') from error
    if not data:raise RuntimeError('No Wiki-40B language was loaded')
    for a,b in [('train','validation'),('train','ranking'),('validation','ranking')]:
        if global_ids[a]&global_ids[b]:raise AssertionError('Document-ID leakage')
    manifest=dict(settings=settings,available_languages=configs,languages=records,missing=missing,
                  token_digest=digest(data),document_partition='global wikidata_id hash: 80/10/10; train split only')
    atomic_torch(data,cache); atomic_json(manifest,path)
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
