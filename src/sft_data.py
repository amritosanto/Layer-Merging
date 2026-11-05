"""Paired response-only Aya samples; benchmark material is audit-only."""
import json
import re
import unicodedata
from collections import Counter,defaultdict
from pathlib import Path
from datasets import load_dataset
from .utils import atomic_json,digest

LANGUAGES={'English':'eng','Hindi':'hin','Japanese':'jpn','Bengali':'ben','Swahili':'swa','Yoruba':'yor','Telugu':'tel'}
PROTOCOL='aya-original-audit-response-only-v1'


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',str(text)).casefold() if c.isalnum())


def shingles(text):
    return {text[i:i+5] for i in range(max(0,len(text)-4))}


class ContaminationIndex:
    def __init__(self,texts,threshold):
        self.texts=sorted({normalized(t) for t in texts if normalized(t)})
        if not self.texts:raise RuntimeError('Empty benchmark contamination index')
        self.exact={digest(t) for t in self.texts};self.threshold=threshold
        self.grams=[shingles(t) for t in self.texts];self.index=defaultdict(set)
        for i,grams in enumerate(self.grams):
            for gram in grams:self.index[gram].add(i)

    def match(self,text):
        value=normalized(text)
        if digest(value) in self.exact:return 'exact_normalized_hash'
        grams=shingles(value);candidates=Counter(i for g in grams for i in self.index.get(g,()))
        for i,common in candidates.items():
            other=self.texts[i]
            if min(len(value),len(other))>=20 and (value in other or other in value):return 'substring_overlap'
            if common/max(1,len(grams|self.grams[i]))>=self.threshold:return 'near_duplicate_or_ambiguous'
        return None


def response_tokens(tokenizer,prompt,response,max_length):
    prefix=tokenizer.encode('Instruction:\n'+prompt+'\n\nResponse:\n',add_special_tokens=False)
    answer=tokenizer.encode(response,add_special_tokens=False)+[tokenizer.eos_token_id]
    if not prefix or len(answer)<2 or len(prefix)+len(answer)>max_length:return None
    return dict(input_ids=prefix+answer,labels=[-100]*len(prefix)+answer)


def load_prepared(cfg,lock,manifest):
    out=Path(cfg.output_dir)
    if manifest['signature']!=digest(dict(protocol=manifest['protocol'],jobs=manifest['jobs'])):
        raise RuntimeError('Frozen benchmark manifest hash mismatch')
    audit=json.loads((out/'manifests/sft_contamination_audit.json').read_text(encoding='utf-8'))
    data=json.loads((out/'sft/accepted_examples.json').read_text(encoding='utf-8'))
    expected=digest([lock,manifest['signature'],PROTOCOL])
    if not audit.get('complete') or audit['identity']!=expected or audit['accepted_digest']!=digest(data):
        raise RuntimeError('SFT audit incomplete, changed or incompatible')
    if data['benchmark_signature']!=manifest['signature']:raise RuntimeError('SFT benchmark freeze changed')
    return data,audit


def prepare_sft(cfg,lock,tokenizer,manifest):
    out=Path(cfg.output_dir);path=out/'manifests/sft_contamination_audit.json'
    if path.exists() and json.loads(path.read_text()).get('complete'):
        return load_prepared(cfg,lock,manifest)
    if not cfg.post_sft:raise ValueError('Requires post-SFT configuration')
    texts=[]
    for job in manifest['jobs']:
        if not job['examples']:raise RuntimeError('Cannot audit an empty benchmark job')
        for example in job['examples']:
            if not example.get('audit_texts') or not example.get('support_ids'):
                raise RuntimeError('Benchmark question/answer/support audit material missing')
            texts.extend(example['audit_texts'])
    index=ContaminationIndex(texts,cfg.sft_near_duplicate_threshold)
    languages=list(cfg.training_languages)
    if any(l not in LANGUAGES for l in languages):raise ValueError('Unverified Aya language mapping')
    counts={l:0 for l in languages};tokens={l:0 for l in languages};accepted=[];excluded=[];seen=set()
    identity=digest([lock,manifest['signature'],PROTOCOL])
    audit=dict(identity=identity,complete=False,benchmark_signature=manifest['signature'],
               benchmark_text_digest=digest(index.texts),benchmark_text_count=len(index.texts),
               dataset=cfg.sft_dataset,revision=lock['dataset_revisions'][cfg.sft_dataset],split='train',
               protocol=PROTOCOL,near_duplicate_threshold=cfg.sft_near_duplicate_threshold,
               provenance_policy='Only original_annotations; reject re-annotations, unknown provenance, benchmark source markers and ambiguous text matches.',
               limitations='Lexical/provenance audit does not prove absence of translated or semantic contamination.')
    atomic_json(audit,path)
    stream=load_dataset(cfg.sft_dataset,'default',split='train',revision=audit['revision'],streaming=True)
    by_code={LANGUAGES[l]:l for l in languages}
    try:
        for row in stream.shuffle(seed=cfg.seed,buffer_size=1000):
            language=by_code.get(row.get('language_code'))
            if language is None or counts[language]>=cfg.sft_examples_per_language:continue
            if not {'inputs','targets','annotation_type'}<=row.keys():raise RuntimeError('Aya schema changed')
            prompt=str(row['inputs']);response=str(row['targets']);rid=digest([audit['revision'],language,prompt,response])
            reason=None
            provenance=' '.join(str(row.get(k,'')) for k in ('source','dataset','provenance','annotation_type'))
            if row['annotation_type']!='original_annotations':reason='non_original_or_unknown_provenance'
            elif re.search(r'mgsm|gsm8k|mmmlu|\bmmlu\b|include[-_ ]?(lite|44|benchmark)',provenance+' '+prompt,re.I):reason='benchmark_source_marker'
            elif not normalized(prompt) or not normalized(response):reason='empty'
            elif rid in seen or digest(normalized(prompt)) in seen:reason='duplicate_sft'
            else:reason=index.match(prompt) or index.match(response) or index.match(prompt+' '+response)
            encoded=None if reason else response_tokens(tokenizer,prompt,response,cfg.sft_max_length)
            if not reason and encoded is None:reason='context_or_empty_response'  # No truncation.
            if not reason and tokens[language]+len(encoded['input_ids'])>cfg.sft_tokens_per_language:reason='token_budget'
            if reason:excluded.append(dict(id=rid,language=language,reason=reason));continue
            seen.update((rid,digest(normalized(prompt))))
            accepted.append(dict(id=rid,language=language,**encoded))
            counts[language]+=1;tokens[language]+=len(encoded['input_ids'])
            if all(n==cfg.sft_examples_per_language for n in counts.values()):break
        if not all(n==cfg.sft_examples_per_language for n in counts.values()):
            raise RuntimeError(f'Insufficient audited Aya examples; no language substitution: {counts}')
        data=dict(benchmark_signature=manifest['signature'],examples=accepted)
        audit.update(complete=True,accepted_ids=[r['id'] for r in accepted],accepted_digest=digest(data),
                     examples_per_language=counts,tokens_per_language=tokens,
                     response_tokens_per_language={l:sum(sum(v!=-100 for v in r['labels']) for r in accepted if r['language']==l) for l in languages},
                     exclusions=excluded,exclusion_counts=dict(Counter(r['reason'] for r in excluded)))
        atomic_json(data,out/'sft/accepted_examples.json');atomic_json(audit,path)
        atomic_json(audit['accepted_ids'],out/'manifests/sft_accepted_ids.json')
        return data,audit
    except BaseException as error:
        audit.update(complete=False,error=repr(error),examples_per_language=counts,exclusions=excluded)
        atomic_json(audit,path);raise
