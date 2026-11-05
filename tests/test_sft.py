import ast
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from config import Config
from src.utils import digest,atomic_json,file_digest
from src.sft_data import ContaminationIndex,prepare_sft,load_prepared,response_tokens
from src.sft_training import response_collator,inference_precision,complete_checkpoint,verified_specs
from src.pair_selection import select_feasible


def frozen():
    jobs=[dict(benchmark='MGSM',language='en',examples=[dict(id='0',audit_texts=['What is the sum of thirty and fifty apples?','80','A demonstration about bananas','12'],support_ids=['train:0'])])]
    protocol={'fixture':True}
    return dict(protocol=protocol,jobs=jobs,signature=digest(dict(protocol=protocol,jobs=jobs)))


def test_eight_blocks_and_old_checkpoint_rejected(tiny):
    from src.merger import PairWeightMerger
    _,six,_,_,families=tiny
    cfg=Config.load('config.smoke_sft13.json')
    assert (cfg.epochs,cfg.max_optimizer_steps,cfg.training_hours)==(10,0,0)
    assert (cfg.train_tokens_per_language,cfg.validation_tokens_per_language,cfg.ranking_tokens_per_language)==(100000,20000,4096)
    eight=PairWeightMerger(cfg,families)
    assert len(eight.blocks)==8 and eight.blocks[0].heads==6 and eight.family.embedding_dim==384
    with pytest.raises(RuntimeError):eight.load_state_dict(six.state_dict())
    assert cfg.ce_reference=='original_fp16' and cfg.max_relative_ce_increase==.13
    rows=[dict(pairs=[0],ranking_ce=2.,reduction=.5),dict(pairs=[0,2],ranking_ce=2.26,reduction=.6),dict(pairs=[0,2,4],ranking_ce=2.261,reduction=.7)]
    assert select_feasible(rows,2.,.13)['pairs']==[0,2]


def test_audit_exact_near_and_short_answers():
    index=ContaminationIndex(['What is the sum of thirty and fifty apples?','80'],.45)
    assert index.match(' WHAT is the sum of thirty and fifty apples!!!')=='exact_normalized_hash'
    assert index.match('What is the sum of thirty and fifty pears?')
    assert index.match('８０')=='exact_normalized_hash'
    assert index.match('Describe the emotional tone of a beautiful sunset') is None


def test_paired_sft_preparation_and_fail_closed(tmp_path,monkeypatch):
    import src.sft_data as s
    cfg=Config.load('config.smoke_sft13.json');cfg.output_dir=str(tmp_path);cfg.training_languages=['English'];cfg.sft_examples_per_language=1
    lock={'dataset_revisions':{cfg.sft_dataset:'immutable'}};manifest=frozen()
    class Stream(list):
        def shuffle(self,**kw):return self
    rows=Stream([
        dict(inputs='What is the sum of thirty and fifty apples?',targets='80',annotation_type='original_annotations',language_code='eng'),
        dict(inputs='Unknown generated source',targets='Untrusted response',annotation_type='re-annotations',language_code='eng'),
        dict(inputs='Describe a sunset',targets='The sky glows with warm colours',annotation_type='original_annotations',language_code='eng')])
    def loader(*args,**kw):
        assert kw['streaming'] and kw['revision']=='immutable' and kw['split']=='train'
        return rows
    monkeypatch.setattr(s,'load_dataset',loader)
    tok=SimpleNamespace(encode=lambda text,**kw:[3,4,5],eos_token_id=2)
    data,audit=prepare_sft(cfg,lock,tok,manifest)
    assert audit['complete'] and len(data['examples'])==1
    assert audit['exclusion_counts']=={'exact_normalized_hash':1,'non_original_or_unknown_provenance':1}
    assert data['examples'][0]['labels']==[-100]*3+[3,4,5,2]
    assert prepare_sft(cfg,lock,tok,manifest)==(data,audit)
    bad=dict(manifest,jobs=[])
    with pytest.raises(RuntimeError):load_prepared(cfg,lock,bad)
    cfg.output_dir=str(tmp_path/'short');cfg.sft_examples_per_language=2
    with pytest.raises(RuntimeError,match='Insufficient'):prepare_sft(cfg,lock,tok,manifest)
    assert not json.loads((Path(cfg.output_dir)/'manifests/sft_contamination_audit.json').read_text())['complete']


def test_response_mask_padding_and_tiny_lora_roundtrip(tiny,tmp_path):
    from peft import LoraConfig,get_peft_model,PeftModel
    teacher,_,_,ids,_=tiny
    model=get_peft_model(teacher,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    batch=response_collator([dict(input_ids=[3,4,5,6],labels=[-100,-100,5,6]),dict(input_ids=[3,4,5],labels=[-100,4,5])],0)
    assert batch['labels'][1,-1]==-100 and batch['attention_mask'][1,-1]==0
    assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    loss=model(**batch).loss;loss.backward();optimizer.step()
    assert torch.isfinite(loss)
    model.save_pretrained(tmp_path/'adapter')
    model=model.merge_and_unload(safe_merge=True).half()
    inference_precision(model,False)
    assert not any('lora_' in n for n,_ in model.named_parameters())
    model.float()
    with pytest.raises(RuntimeError,match='FP16'):inference_precision(model,False)


def test_checkpoint_boundary_integrity(tmp_path):
    for i in (10,20):
        p=tmp_path/f'checkpoint-{i}';p.mkdir();(p/'optimizer.pt').write_bytes(b'fixture')
    p=tmp_path/'checkpoint-10'
    atomic_json(dict(files={'optimizer.pt':file_digest(p/'optimizer.pt')}),p/'complete.json')
    assert complete_checkpoint(tmp_path)==str(p)
    (p/'optimizer.pt').write_bytes(b'changed')
    assert complete_checkpoint(tmp_path) is None


def test_benchmark_gate_and_paired_specs(tmp_path,monkeypatch):
    import src.sft_training as s
    cfg=Config.load('config.smoke_sft13.json');cfg.output_dir=str(tmp_path)
    (tmp_path/'manifests').mkdir();weight=tmp_path/'weights';weight.write_bytes(b'fixture')
    audit={'accepted_digest':'shared'}
    monkeypatch.setattr(s,'context',lambda *a:(tmp_path,None,audit,None,'identity'))
    manifest=frozen()
    with pytest.raises(RuntimeError,match='Both SFT'):verified_specs(cfg,{},manifest)
    records=[]
    for name,nf4 in zip(s.NAMES,[False,True]):
        r=dict(identity='identity',completed_epochs=2,accepted_digest='shared',optimizer_steps=20,files={'weights':file_digest(weight)},spec=dict(name=name,nf4=nf4))
        atomic_json(r,tmp_path/f'manifests/{name}.json');records.append(r)
    with pytest.raises(RuntimeError,match='sft-verify'):verified_specs(cfg,{},manifest)
    atomic_json(dict(records_digest=digest(records)),tmp_path/'manifests/sft_verified.json')
    specs=verified_specs(cfg,{},manifest)
    assert [(s['name'],s['nf4']) for s in specs]==[('original_fp16_sft',False),('merged_nf4_sft',True)]
    weight.write_bytes(b'corrupt')
    with pytest.raises(RuntimeError):verified_specs(cfg,{},manifest)


def test_sft_notebook_stages():
    n=json.loads(Path('Qwen3_8Block_CE13_100k_20k_SFT_Kaggle.ipynb').read_text())
    cells=[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
    for c in cells:ast.parse(c)
    stages=[c.strip().split("'")[1] for c in cells if c.strip().startswith("run_stage('")]
    assert stages==['prepare','preflight','train','search','export','sft-prepare','sft-original','sft-merged','sft-verify','benchmark','report']
    assert 'ZIP_STORED' in cells[-1] and 'allowZip64=True' in cells[-1] and 'archive.resolve()' in cells[-1]
    assert 'peft==0.15.2' in ''.join(cells) and 'config.smoke_sft13.json' in ''.join(cells)


def test_post_sft_report(tmp_path,monkeypatch):
    import src.sft_reporting as report
    cfg=Config.load('config.smoke_sft13.json');cfg.output_dir=str(tmp_path)
    specs=[dict(name='original_fp16_sft',stored_bytes=1000,adapter_bytes=0,layers=28),dict(name='merged_nf4_sft',stored_bytes=600,adapter_bytes=50,layers=20)]
    monkeypatch.setattr(report,'prepare_run',lambda c:{})
    monkeypatch.setattr(report,'verified_specs',lambda *a:specs)
    examples=[dict(id=str(i),prompt_hash=str(i),support_ids=['train:0'],target=0,skip_reason=None) for i in range(4)]
    atomic_json(dict(signature='fixture',jobs=[dict(benchmark='MMMLU',language='en',examples=examples)]),tmp_path/'manifests/benchmark_prompts_pilot.json')
    atomic_json(dict(weight_hashes={s['name']:'weights' for s in specs}),tmp_path/'manifests/sft_verified.json')
    for spec in specs:
        folder=tmp_path/'benchmark_results/pilot'/spec['name'];folder.mkdir(parents=True)
        rows=[dict(e,identity=digest(['fixture',spec,'weights','evaluation']),correct=True,invalid=False,prediction=0) for e in examples]
        (folder/'MMMLU_en.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    selected=dict(completed_epochs=10,training_updates=10,baseline=dict(reference_ranking=2.,reference_validation=2.1),ranking_ce=2.2,
        relative_ce_increase=.1,feasible=True,ce_target_met=True,final_nf4_validation_ce=2.25,pairs=[0,2],remaining_layers=26,actual_reduction=.5,reduction_vs_original_nf4=.1)
    atomic_json(selected,tmp_path/'manifests/final_export.json')
    atomic_json(dict(examples_per_language={'English':100},tokens_per_language={'English':1000},limitations='Synthetic fixture.'),tmp_path/'manifests/sft_contamination_audit.json')
    result=report.make_sft_report(cfg).read_text()
    assert 'quantization, merging, and SFT' in result and 'pre-SFT diagnostics' in result
    import pandas as pd
    rows=pd.read_csv(tmp_path/'benchmark_results/pilot/per_language_accuracy.csv')
    assert rows.original_fp16_sft_correct.iloc[0]==4 and rows.relative_accuracy_retention.iloc[0]==1
    assert rows.paired_ci95_lower_pp.iloc[0]==0
