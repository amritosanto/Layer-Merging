import ast
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from config import Config
from src.pair_selection import task_for_batch,valid_plan
from src.sft_generated import generate,solve,TEMPLATES
from src.sft_training import PairedSampler
from src.sft_mixture import prepare_mixture,allocation
from src.sft_data import ContaminationIndex
from src.utils import digest


def test_heavy_configuration_and_depth_exposure():
    c=Config.load('config.smoke_sft13.json')
    assert (c.train_tokens_per_language,c.validation_tokens_per_language,c.ranking_tokens_per_language)==(10000,3000,2048)
    assert (c.sft_examples_per_language,c.sft_validation_examples_per_language,c.sft_max_length)==(900,100,512)
    assert c.sft_enabled and not c.rl_enabled and c.sft_lora_dropout==.05
    plans=[task_for_batch(42,0,i,weights=c.merge_count_weights) for i in range(140)]
    assert Counter(map(len,plans))=={k:10 for k in range(1,15)}
    assert all(valid_plan(p)==p for p in plans)
    assert allocation(900)==dict(instruction=360,math=315,mc=225)
    c.rl_enabled=True
    with pytest.raises(AssertionError):c.validate()


def test_verified_math_and_balanced_mc():
    from fractions import Fraction
    assert [solve(i,3,10,2) for i in range(8)]==[32,28,10,24,26,30,10,Fraction(3,13)]
    for language in TEMPLATES:
        labels=[]
        for i in range(32):
            math=generate(language,'math',i,42)
            assert str(solve(*math['operands']))==math['verified_answer']
            assert math['targets'].endswith('#### '+math['verified_answer'])
            mc=generate(language,'mc',i,42,i)
            assert len(set(mc['choices']))==4 and mc['choices'][mc['label']]==str(solve(*mc['operands']))
            assert mc['targets']==' '+chr(65+mc['label']) and mc['inputs'].endswith('Answer:')
            labels.append(mc['label'])
        assert Counter(labels)=={i:8 for i in range(4)}


def test_sampler_order_independent_of_model_rng():
    a=PairedSampler(list(range(30)),42);b=PairedSampler(list(range(30)),42)
    first=list(a);torch.rand(3000);assert list(b)==first
    a.set_epoch(1);b.set_epoch(1)
    assert list(a)==list(b) and list(a)!=first


def test_heavy_mixture_isolation_and_cache(tmp_path,monkeypatch):
    import src.sft_mixture as m
    cfg=Config.load('config.smoke_sft13.json');cfg.output_dir=str(tmp_path);cfg.training_languages=['English','Japanese']
    cfg.sft_examples_per_language=20;cfg.sft_validation_examples_per_language=8
    class Stream(list):
        def shuffle(self,**kwargs):return self
    rows=Stream(dict(language_code=code,inputs=f'Explain distant galaxies and planets in scenario {i} for {code}',
                     targets=f'A unique detailed discussion about stellar atmospheres {i} in {code}',annotation_type='original_annotations')
                for code in ('eng','jpn') for i in range(15))
    monkeypatch.setattr(m,'load_dataset',lambda *a,**kw:rows)
    jobs=[dict(benchmark='MMMLU',language='AR_XY',examples=[dict(audit_texts=['Forbidden benchmark question about ancient diplomacy','secret answer'],support_ids=['support'])])]
    protocol={};manifest=dict(protocol=protocol,jobs=jobs,signature=digest(dict(protocol=protocol,jobs=jobs)))
    tok=SimpleNamespace(encode=lambda text,**kw:[3,4,5],eos_token_id=2)
    lock=dict(dataset_revisions={cfg.sft_dataset:'pinned'})
    data,audit=prepare_mixture(cfg,lock,tok,manifest)
    assert len(data['examples'])==40 and len(data['validation'])==16
    assert not {r['problem_id'] for r in data['examples']}&{r['problem_id'] for r in data['validation']}
    assert len({r['id'] for r in data['examples']+data['validation']})==56
    assert all(r['labels'][:3]==[-100]*3 for r in data['examples'])
    for key,count in audit['actual_allocation'].items():assert count==allocation(20 if key.startswith('train') else 8)
    for counts in audit['mc_label_counts'].values():assert max(counts.values())-min(counts.values())<=1
    assert prepare_mixture(cfg,lock,tok,manifest)==(data,audit)
    assert (tmp_path/'sft/accepted_validation_ids.json').exists()
    assert audit['benchmark_languages_without_dedicated_sft']==['Arabic','Albanian']


def test_numeric_ambiguity_filter():
    audit=ContaminationIndex(['A worker packed 23 boxes with 17 items and added 41 items.'],.45)
    assert audit.match('箱は23個、中身は17個、追加は41個。')=='ambiguous_numeric_translation'


def test_real_cpu_trainer_exposure_and_validation(tiny,tmp_path):
    from transformers import TrainingArguments
    from peft import get_peft_model,LoraConfig
    from src.sft_training import exposure_trainer_class,response_collator
    model,_,_,_,_=tiny
    model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,lora_dropout=.05,target_modules=['q_proj'],task_type='CAUSAL_LM'))
    cfg=Config.load('config.smoke_sft13.json')
    exposure=dict(tokens={'English':0,'Hindi':0},examples={'English':0,'Hindi':0},steps=[])
    samples=[dict(input_ids=[3,4,5,6],labels=[-100,-100,5,6],_language='English' if i%2 else 'Hindi') for i in range(8)]
    def collate(rows):
        batch=response_collator(rows,0)
        if '_language' in rows[0]:batch['_languages']=[r['_language'] for r in rows]
        return batch
    cls=exposure_trainer_class(cfg,exposure)
    args=TrainingArguments(output_dir=str(tmp_path),use_cpu=True,num_train_epochs=2,per_device_train_batch_size=1,
        gradient_accumulation_steps=4,save_strategy='no',eval_strategy='epoch',report_to=[],remove_unused_columns=False,
        prediction_loss_only=True,seed=42,data_seed=42,disable_tqdm=True,label_names=['labels'])
    validation=[dict(input_ids=[3,4,5,6],labels=[-100,-100,5,6])]
    trainer=cls(model=model,args=args,train_dataset=samples,eval_dataset={'English':validation,'Hindi':validation},data_collator=collate)
    trainer.train()
    assert trainer.state.epoch==2 and trainer.state.global_step==4
    assert exposure['examples']=={'English':8,'Hindi':8} and exposure['tokens']=={'English':32,'Hindi':32}
    assert any('eval_English_loss' in r for r in trainer.state.log_history)
    assert any('eval_Hindi_loss' in r for r in trainer.state.log_history)


def test_heavy_notebook_archive(tmp_path,monkeypatch):
    n=json.loads(Path('Qwen3_8Block_HeavySFT_7Hour_Kaggle.ipynb').read_text())
    cells=[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
    for c in cells:ast.parse(c)
    assert "run_stage('sft-both' if SFT_CONCURRENT" in ''.join(cells)
    last=cells[-1];assert 'disk_usage' in last and 'len(files)' in last and 'allowZip64=True' in last
    import sys,zipfile
    from types import ModuleType
    display=ModuleType('IPython.display');display.display=lambda *a:None;display.FileLink=lambda *a:None;display.Javascript=lambda *a:None
    monkeypatch.setitem(sys.modules,'IPython.display',display);monkeypatch.chdir(tmp_path)
    (tmp_path/'model.pt').write_bytes(b'fixture');(tmp_path/'source.py').write_text('pass')
    code=last.replace("Path('/kaggle/working')",'Path('+repr(str(tmp_path))+')').replace('SALVAGE_PARTIAL=False','SALVAGE_PARTIAL=True')
    exec(compile(code,'archive','exec'),{})
    with zipfile.ZipFile(tmp_path/'Qwen3_8Block_HeavySFT_WORKING.zip') as z:
        assert set(z.namelist())=={'model.pt','source.py'} and z.getinfo('model.pt').compress_type==zipfile.ZIP_STORED
