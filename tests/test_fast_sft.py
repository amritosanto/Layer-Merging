import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from config import Config
from src.utils import atomic_json,diagnostic_json,file_digest
from src.sft_runtime import choose_batch,profile_candidates,finite_gradients,completed_state,child_environments


def test_fast_config_and_common_effective_batch():
    c=Config.load('config.smoke_sft2h.json');later=Config.load('config.smoke_sft2h_eval100.json')
    assert (c.epochs,c.max_optimizer_steps,c.training_hours,c.merger_blocks,c.max_relative_ce_increase)==(10,0,0,8,.13)
    assert (c.sft_epochs,c.sft_examples_per_language,c.sft_validation_examples_per_language)==(1,200,25)
    assert (c.sft_effective_batch_size,c.pilot_questions,c.mgsm_max_new_tokens)==(8,30,256)
    assert later.output_dir==c.output_dir and later.pilot_questions==100 and later.benchmark_audit_questions==100
    for micro in (1,2,4,8):
        r=choose_batch([dict(status='ok',microbatch=micro,tokens_per_second=1,peak_bytes=1)],8)
        assert r['microbatch']*r['gradient_accumulation']==8
    environments=child_environments({'PRESERVED':'value'})
    assert [e['CUDA_VISIBLE_DEVICES'] for e in environments]==['0','1']
    assert all(e['PRESERVED']=='value' for e in environments)


def test_autotune_oom_and_instability_rejection():
    def probe(batch):
        if batch==2:raise torch.cuda.OutOfMemoryError()
        if batch==4:raise FloatingPointError('nonfinite')
        return dict(tokens_per_second=batch,peak_bytes=batch)
    result=profile_candidates([1,2,4,8],probe,8)
    assert result['selected']['microbatch']==8 and result['selected']['gradient_accumulation']==1
    assert [r['status'] for r in result['candidates']]==['ok','oom','nonfinite','ok']
    with pytest.raises(RuntimeError):choose_batch([dict(status='oom',microbatch=1)],8)


def test_diagnostic_nonfinite_serializer_is_scoped(tmp_path):
    path=tmp_path/'metrics.json'
    value=dict(loss_history=[dict(loss=.25,eval_loss=float('nan'),nested=[float('inf'),float('-inf'),7])])
    names=diagnostic_json(value,path);saved=json.loads(path.read_text())
    assert saved['loss_history'][0]==dict(loss=.25,eval_loss=None,nested=[None,None,7])
    assert names==['loss_history[0].eval_loss','loss_history[0].nested[0]','loss_history[0].nested[1]']
    with pytest.raises(ValueError):atomic_json(value,tmp_path/'scientific_manifest.json')
    model=torch.nn.Linear(1,1);model.weight.grad=torch.full_like(model.weight,float('nan'))
    with pytest.raises(FloatingPointError,match='actual SFT gradients'):finite_gradients(model)


def test_actual_nonfinite_training_loss_stops(tmp_path):
    from transformers import TrainingArguments
    from src.sft_training import exposure_trainer_class,response_collator
    class Bad(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
        def forward(self,input_ids,attention_mask=None,labels=None):return {'loss':self.weight.sum()*float('nan')}
    cfg=Config.load('config.smoke_sft2h.json');exposure=dict(tokens={'English':0},examples={'English':0})
    cls=exposure_trainer_class(cfg,exposure)
    args=TrainingArguments(output_dir=str(tmp_path),use_cpu=True,num_train_epochs=1,report_to=[],remove_unused_columns=False,save_strategy='no',disable_tqdm=True)
    def collate(rows):return dict(response_collator(rows,0),_languages=['English']*len(rows))
    model=Bad();trainer=cls(model=model,args=args,train_dataset=[dict(input_ids=[1,2],labels=[-100,2])],data_collator=collate)
    with pytest.raises(RuntimeError,match='Nonfinite SFT training loss'):trainer.train()
    assert model.weight.item()==1 and exposure['examples']['English']==0


def test_completed_checkpoint_finalizes_without_training(tiny,tmp_path,monkeypatch):
    from peft import get_peft_model,LoraConfig
    from transformers import Trainer
    import src.sft_training as s
    cfg=Config.load('config.smoke_sft2h.json');cfg.output_dir=str(tmp_path);cfg.training_languages=['English']
    base=copy.deepcopy(tiny[0]);source=copy.deepcopy(base)
    model=get_peft_model(source,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none',task_type='CAUSAL_LM'))
    folder=tmp_path/'sft/original_fp16_sft';checkpoint=folder/'checkpoint-1';checkpoint.mkdir(parents=True)
    model.save_pretrained(checkpoint)
    state=dict(epoch=1.,global_step=1,log_history=[dict(eval_loss=float('nan'),loss=.1)])
    (checkpoint/'trainer_state.json').write_text(json.dumps(state))
    exposure=dict(examples={'English':4},tokens={'English':16},steps=[])
    atomic_json(exposure,checkpoint/'exposure.json')
    atomic_json(dict(files={p.name:file_digest(p) for p in checkpoint.iterdir() if p.is_file()}),checkpoint/'complete.json')
    atomic_json(dict(identity='expected'),folder/'identity.json')
    data=dict(examples=[dict(language='English',input_ids=[3,4,5,6],labels=[-100,-100,5,6]) for _ in range(4)])
    audit=dict(tokens_per_language={'English':16},accepted_digest='data');final={'remaining_layers':12}
    monkeypatch.setattr(s,'context',lambda *a:(tmp_path,data,audit,final,'expected'))
    monkeypatch.setattr(s,'load_model',lambda *a,**kw:copy.deepcopy(base))
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(Trainer,'train',lambda *a,**kw:pytest.fail('Completed optimizer steps must not run again'))
    tokenizer=SimpleNamespace(save_pretrained=lambda p:(p/'tokenizer.json').write_text('{}'))
    record=s.train_sft(cfg,{'model_revision':'fixture'},tokenizer,{'signature':'audit'},'original_fp16_sft')
    assert record['finalized_from_checkpoint'] and record['completed_epochs']==1 and record['optimizer_steps']==1
    assert record['nonfinite_diagnostic_metrics']==['loss_history[0].eval_loss']
    assert json.loads((folder/'training_metrics.json').read_text())['loss_history'][0]['eval_loss'] is None
    assert s.train_sft(cfg,{'model_revision':'fixture'},tokenizer,{'signature':'audit'},'original_fp16_sft')==record
    assert s.complete_checkpoint(folder)==str(checkpoint)
    (checkpoint/'adapter_config.json').write_text('{}')
    assert s.complete_checkpoint(folder) is None  # Recovery never bypasses checkpoint hashes.


def test_evaluation_profile_reuses_training_identity(tmp_path,monkeypatch):
    import src.trainer as t
    api=SimpleNamespace(dataset_info=lambda *a,**kw:SimpleNamespace(sha='pinned-data'),model_info=lambda *a,**kw:SimpleNamespace(sha='pinned-model'))
    monkeypatch.setattr(t,'HfApi',lambda:api)
    fast=Config.load('config.smoke_sft2h.json');later=Config.load('config.smoke_sft2h_eval100.json')
    fast.output_dir=later.output_dir=str(tmp_path)
    first=t.prepare_run(fast)
    monkeypatch.setattr(t,'HfApi',lambda:pytest.fail('Existing run lock must be reused'))
    assert t.prepare_run(later)==first


def test_fast_notebook_archive(tmp_path,monkeypatch):
    import ast,sys,zipfile
    from types import ModuleType
    n=json.loads(Path('Qwen3_8Block_FastSFT_2Hour_Kaggle.ipynb').read_text())
    cells=[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
    for c in cells:ast.parse(c)
    assert 'sft-both' in ''.join(cells) and 'preflight.json' in ''.join(cells)
    assert 'config.smoke_sft2h.json' in ''.join(cells)
    display=ModuleType('IPython.display');display.display=lambda *a:None;display.FileLink=lambda *a:None;display.Javascript=lambda *a:None
    monkeypatch.setitem(sys.modules,'IPython.display',display);monkeypatch.chdir(tmp_path)
    (tmp_path/'weights.pt').write_bytes(b'fixture')
    code=cells[-1].replace("Path('/kaggle/working')",'Path('+repr(str(tmp_path))+')').replace('SALVAGE_PARTIAL=False','SALVAGE_PARTIAL=True')
    exec(compile(code,'archive','exec'),{})
    with zipfile.ZipFile(tmp_path/'Qwen3_8Block_FastSFT_2Hour_WORKING.zip') as z:
        assert z.namelist()==['weights.pt'] and z.getinfo('weights.pt').compress_type==zipfile.ZIP_STORED
