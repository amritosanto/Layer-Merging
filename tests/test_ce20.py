import ast
import json
from pathlib import Path
import pytest
from config import Config
from src.pair_selection import candidate_plans,valid_plan,task_for_batch,select_feasible,pareto_frontier
from src.wiki40b_loader import balanced_epoch


def test_configuration_and_all_depths():
    c=Config.load('config.smoke10.json')
    assert (c.epochs,c.max_optimizer_steps,c.training_hours)==(10,0,0)
    assert (c.train_tokens_per_language,c.validation_tokens_per_language,c.ranking_tokens_per_language)==(10000,3000,2048)
    assert c.max_relative_ce_increase==.20 and c.accuracy_loss_target==.05
    assert (c.mmmlu_few_shot,c.include_few_shot,c.mgsm_few_shot,c.pilot_questions)==(8,8,8,100)
    plans=candidate_plans(list(range(27)),budget=c.max_candidate_plans-c.max_refinement_rounds)
    assert {len(p) for p in plans}==set(range(1,15))
    assert all(valid_plan(p)==p for p in plans)
    tasks=[task_for_batch(42,0,i) for i in range(56)]
    assert {len(p) for p in tasks}==set(range(1,15))
    assert all(valid_plan(p)==p for p in tasks)
    c.max_final_nf4_candidates=3
    with pytest.raises(AssertionError):c.validate()


def row(k,ce,reduction=.6):
    return dict(pairs=list(range(0,2*k,2)),ranking_ce=ce,reduction=reduction)


def test_relative_limit_depth_ties_and_diagnostic():
    rows=[row(1,1.5),row(4,2.4),row(5,2.40001),row(7,1.9,.39)]
    assert select_feasible(rows,2.) is rows[1]
    assert select_feasible([row(1,1.),row(2,1.5)],2.)['pairs']==[0,2]
    assert select_feasible([row(4,2.3),row(4,2.2)],2.)['ranking_ce']==2.2
    assert select_feasible([row(4,2.3),dict(row(4,2.3),pairs=[1,3,5,7])],2.)['pairs']==[0,2,4,6]
    assert select_feasible([row(1,3.),row(4,4.)],2.)['ranking_ce']==3.
    with pytest.raises(ValueError):select_feasible(rows,0.)


def test_pareto():
    rows=[row(0,2.),row(1,1.9),row(2,2.1),row(3,2.1),row(4,2.5)]
    assert [len(r['pairs']) for r in pareto_frontier(rows)]==[1,3,4]


def test_ten_complete_epochs_and_all_tokens(tmp_path,monkeypatch):
    from test_multilingual_upgrade import training_fixture
    e=training_fixture(tmp_path,monkeypatch)
    e.cfg=Config.load('config.smoke10.json');e.cfg.output_dir=str(tmp_path)
    e.data={l:dict(train=list(range(10000))) for l in e.cfg.training_languages}
    windows=balanced_epoch(e.data,128,42,0)
    for language in e.data:
        values=[int(i) for l,w in windows if l==language for i in w]
        assert sorted(values)==list(range(10000))
    e.train()
    done=json.loads((tmp_path/'manifests/training_complete.json').read_text())
    assert done['state']['epoch']==10 and done['state']['step']==1390 and done['stop_reason']=='epochs'
    metrics=json.loads((tmp_path/'logs/training_metrics.json').read_text())
    assert set(metrics['actual_training_tokens_per_language'].values())=={100000}


def test_notebook_and_archive(tmp_path,monkeypatch):
    n=json.loads(Path('Qwen3_10Epoch_CE20_Kaggle_Launcher.ipynb').read_text())
    cells=[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
    for s in cells:ast.parse(s)
    assert 'config.smoke10.json' in ''.join(cells)
    last=cells[-1]
    assert 'allowZip64=True' in last and 'ZIP_STORED' in last and 'SALVAGE_PARTIAL' in last
    # Execute the archive body against tiny files, suppressing UI/download calls.
    import sys
    from types import ModuleType
    display=ModuleType('IPython.display')
    display.display=lambda *a:None
    display.FileLink=lambda *a:None
    display.Javascript=lambda *a:None
    monkeypatch.setitem(sys.modules,'IPython.display',display)
    monkeypatch.chdir(tmp_path)
    (tmp_path/'weights.pt').write_bytes(b'fixture')
    (tmp_path/'readme.txt').write_text('source')
    code=last.replace("Path('/kaggle/working')",'Path('+repr(str(tmp_path))+')').replace('SALVAGE_PARTIAL=False','SALVAGE_PARTIAL=True')
    exec(compile(code,'archive','exec'),{})
    import zipfile
    with zipfile.ZipFile(tmp_path/'Qwen3_10Epoch_CE20_working_archive.zip') as z:
        assert set(z.namelist())=={'weights.pt','readme.txt'}
        assert z.getinfo('weights.pt').compress_type==zipfile.ZIP_STORED


def test_report_and_pareto_output(tmp_path):
    from src.reporting import make_report,plot_pareto
    cfg=Config.load('config.smoke10.json');cfg.output_dir=str(tmp_path)
    for d in ('rankings','manifests','logs','benchmark_results/pilot/original_nf4','benchmark_results/pilot/best_learned_nf4'):
        (tmp_path/d).mkdir(parents=True,exist_ok=True)
    selected=dict(completed_epochs=10,training_updates=1390,pairs=[0],remaining_layers=27,searched_plans=16,nf4_finalists=14,
        selection='maximum feasible depth',ranking_ce=2.2,ce_increase_original=.2,final_nf4_validation_ce=2.3,final_nf4_ce_increase=.3,
        serialized_bytes=400,actual_reduction=.6,reduction_vs_original_nf4=.2,storage_target_met=True,target_50_met=True,format='NF4',
        ce_target_met=True,feasible=True,max_relative_ce_increase=.2,
        baseline=dict(nf4_ranking=2.,nf4_validation=2.,nf4_serialized_bytes=500,reference_storage=dict(bytes=1000,basis='fixture')))
    (tmp_path/'rankings/selected_plan.json').write_text(json.dumps(selected))
    (tmp_path/'logs/training_metrics.json').write_text(json.dumps(dict(state=dict(seconds=1.),actual_training_tokens_per_language={'English':100000})))
    (tmp_path/'manifests/language_coverage.json').write_text('[]')
    examples=[dict(id=str(i),prompt_hash=str(i),support_ids=['support'],target=0,skip_reason=None) for i in range(4)]
    (tmp_path/'manifests/benchmark_prompts_pilot.json').write_text(json.dumps(dict(jobs=[dict(benchmark='MMMLU',language='en',examples=examples)])))
    for variant in ('original_nf4','best_learned_nf4'):
        rows=[dict(e,benchmark='MMMLU',language='en',prediction=0,invalid=False,correct=True) for e in examples]
        (tmp_path/f'benchmark_results/pilot/{variant}/MMMLU_en.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=make_report(cfg).read_text()
    assert 'relative ratio' in report and 'five-percentage-point' in report and 'Total saving vs FP16' in report
    import pandas as pd
    table=pd.read_csv(tmp_path/'benchmark_results/pilot/per_language_accuracy.csv')
    assert table.paired_ci95_lower_pp.iloc[0]==0 and table.accuracy_delta_pp.iloc[0]==0
    r=dict(row(1,2.2),relative_ce_increase_percent=10,on_pareto_frontier=True)
    plot_pareto([r],r,.2,tmp_path/'chart.png')
    assert (tmp_path/'chart.png').stat().st_size>0
