"""Small read-only network checks; no pretrained weights or benchmark inference."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
import src
from datasets import get_dataset_config_names,load_dataset
from huggingface_hub import HfApi
from transformers import AutoConfig,AutoTokenizer
from benchmarks import mmmlu,include,mgsm
from src.utils import atomic_json
from src.wiki40b_loader import clean_article

api=HfApi(); report={}
model='Qwen/Qwen3-0.6B-Base'; revision=api.model_info(model).sha
config=AutoConfig.from_pretrained(model,revision=revision)
tokenizer=AutoTokenizer.from_pretrained(model,revision=revision)
report['model']=dict(revision=revision,architecture={k:getattr(config,k) for k in ('hidden_size','head_dim','num_hidden_layers','num_attention_heads','num_key_value_heads')},
                     label_tokens={c:tokenizer.encode(' '+c,add_special_tokens=False) for c in 'ABCD'})
for dataset,language,adapter in [('google/wiki40b','en',None),('openai/MMMLU','AR_XY',mmmlu.adapt),
                                  ('CohereLabs/include-lite-44','Albanian',include.adapt)]:
    revision=api.dataset_info(dataset).sha
    configs=get_dataset_config_names(dataset,revision=revision)
    split='train' if adapter is None else 'test'
    row=next(iter(load_dataset(dataset,language,revision=revision,split=split,streaming=True)))
    if adapter is None:
        assert clean_article(row['text'])
    else:adapter(row,0)
    report[dataset]=dict(revision=revision,configs=configs,sample_config=language,columns=list(row),adapter_passed=True)
    print(dataset,'passed',len(configs),'configs',flush=True)
reference=None; revision=api.dataset_info('juletxara/mgsm').sha
for language in mgsm.LANGUAGES:
    ds=load_dataset('juletxara/mgsm',language,revision=revision)
    answers=list(ds['test']['answer_number'])
    assert len(ds['train'])==8 and len(answers)==250
    if reference is None:reference=answers
    else:assert answers==reference,language
    assert len(tokenizer.encode(mgsm.prompt(list(ds['train']),ds['test'][0]['question'],language)))>0
report['juletxara/mgsm']=dict(revision=revision,languages=mgsm.LANGUAGES,aligned_answer_sequences=True,train_examples=8,test_examples=250)
atomic_json(report,ROOT/'validation/live_schemas.json'); print('All live schema checks passed')
