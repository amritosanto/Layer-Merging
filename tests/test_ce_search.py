import ast
import json
from dataclasses import asdict
from pathlib import Path
import pytest
from config import Config
from src.pair_selection import candidate_plans,compatible_selection,valid_plan,local_swaps,select_measured
from src.benchmark_evaluation import variants,prepare_benchmarks
from src.utils import file_digest


def test_all_levels_and_budget():
    costs=[float(i%7-3) for i in range(27)]
    for k in range(1,15):
        plan=compatible_selection(costs,k)
        assert len(plan)==k and valid_plan(plan)==plan
    plans=candidate_plans(costs,budget=12)
    assert len(plans)<=12 and len(plans[0])==1 and len(plans[-1])==14
    assert all(valid_plan(p)==p for p in plans)
    assert all(valid_plan(p)==p for p in local_swaps((0,4,8)))


def test_measured_ce_and_storage_select_plan():
    rows=[dict(pairs=[0],ranking_ce=3.,reduction=.45),
          dict(pairs=[2,4],ranking_ce=2.,reduction=.48),
          dict(pairs=[0,2,4],ranking_ce=1.,reduction=.65)]
    assert select_measured(rows)['pairs']==[0,2,4]
    assert select_measured(rows[2:])['pairs']==[0,2,4]
    assert select_measured([dict(pairs=[0],ranking_ce=2.,reduction=.6),dict(pairs=[0,2],ranking_ce=2.,reduction=.7)])['pairs']==[0,2]
    with pytest.raises(ValueError):select_measured([dict(pairs=[0],ranking_ce=float('nan'),reduction=.5)])


def test_smoke_config_and_json_normalization(tmp_path):
    cfg=Config.load('config.smoke.json')
    assert (cfg.epochs,cfg.max_optimizer_steps,cfg.training_hours)==(5,0,0)
    assert (cfg.merger_blocks,cfg.merger_width,cfg.merger_heads)==(6,384,6)
    assert cfg.ce_weight==0 and cfg.quantization=='nf4' and cfg.teacher_precision=='fp16'
    path=tmp_path/'config.json'; cfg.save(path)
    assert json.loads(json.dumps(asdict(cfg)))==json.loads(json.dumps(asdict(Config.load(path))))


def test_only_two_benchmark_variants(tmp_path):
    cfg=Config(output_dir=str(tmp_path))
    folder=tmp_path/'manifests'; folder.mkdir()
    weight=tmp_path/'model.safetensors'; weight.write_bytes(b'fixture')
    (folder/'final_export.json').write_text(json.dumps(dict(remaining_layers=22,files={'model.safetensors':file_digest(weight)})))
    baseline=tmp_path/'reference_models/original_nf4'; baseline.mkdir(parents=True)
    (baseline/'weights').write_bytes(b'NF4 fixture')
    from src.quantization import nf4_config
    (folder/'original_nf4.json').write_text(json.dumps(dict(identity=dict(model_id=cfg.model_id,revision='immutable',quantization=nf4_config().to_dict()),files={'weights':file_digest(baseline/'weights')})))
    specs=variants(cfg,{'model_revision':'immutable'})
    assert [(s['name'],s['nf4']) for s in specs]==[('original_nf4',True),('best_learned_nf4',True)]
    weight.write_bytes(b'changed')
    with pytest.raises(RuntimeError):variants(cfg,{'model_revision':'immutable'})


def test_launcher_syntax():
    n=json.loads(Path('Qwen3_5Epoch_Multilingual_NF4_Kaggle_Launcher.ipynb').read_text())
    assert n['nbformat']==4
    for c in n['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    text=json.dumps(n)
    assert 'git' in text and 'clone' in text and 'config.smoke.json' in text
    assert 'sentencepiece==' not in text and 'only-binary' in text


@pytest.mark.parametrize('configuration,count',[('config.smoke.json',10),('config.smoke10.json',100)])
def test_smoke_benchmarks_load_only_selected_configs(tmp_path,monkeypatch,configuration,count):
    from types import SimpleNamespace
    import src.benchmark_evaluation as b
    cfg=Config.load(configuration); cfg.output_dir=str(tmp_path)
    # Tuple input must compare identically with JSON's cached lists.
    cfg.benchmark_languages['MMMLU']=('HI_IN','JA_JP')
    calls=[]
    class Split(list):
        _fingerprint='fixture'
    def load(dataset,language,**kwargs):
        calls.append((dataset,language))
        if dataset=='juletxara/mgsm':
            return dict(train=Split(dict(question=f'support {i}',answer_number=str(i),answer='reason '+str(i)) for i in range(8)),
                        test=Split(dict(question=f'test {i}',answer_number=str(i)) for i in range(250)))
        return dict(test=Split(dict(question=f'{language} question {i}',choices=['one','two'],label=i%2,
                                    id=str(i),subject='s',domain='d',regional='r') for i in range(120)))
    monkeypatch.setattr(b,'load_dataset',load)
    monkeypatch.setattr(b,'get_dataset_config_names',lambda *a,**kw:['Hindi','Japanese','Albanian','Dutch-Flemish'])
    monkeypatch.setattr(b,'HfApi',lambda:SimpleNamespace(list_repo_files=lambda *a,**kw:[f'{lang}/test-0.parquet' for lang in ['Hindi','Japanese','Albanian']]))
    monkeypatch.setattr(b.mmmlu,'adapt',lambda r,i:r)
    monkeypatch.setattr(b.include,'adapt',lambda r,i:r)
    tokenizer=SimpleNamespace(encode=lambda *a,**kw:[1,2,3])
    lock=dict(model_revision='model',dataset_revisions={d:'revision' for d in ['openai/MMMLU',cfg.include_dataset,'juletxara/mgsm']})
    result=prepare_benchmarks(cfg,lock,tokenizer)
    assert len(calls)==9 and len(result['jobs'])==9
    assert all(len(j['examples'])==count for j in result['jobs'])
    assert all(len(e['support_ids'])==8 for j in result['jobs'] if j['benchmark']=='MGSM' for e in j['examples'])
    if count==100:assert all(len(e['support_ids'])==8 for j in result['jobs'] for e in j['examples'])
    assert prepare_benchmarks(cfg,lock,tokenizer)==result
    assert len(calls)==9  # Frozen prompt cache reused, no extra dataset loads.
