import json
from types import SimpleNamespace
from pathlib import Path
import pytest
import torch
from config import Config
from src.multilingual_data import coverage,MAPPING_PATH
from src.wiki40b_loader import prepare_data,balanced_epoch
from benchmarks.mgsm import extract_answer


def test_verified_mapping_and_dialect_proxies():
    cfg=Config.load('config.smoke.json')
    catalog=json.loads(MAPPING_PATH.read_text(encoding='utf-8'))
    assert catalog['official_entry_count']==119
    assert sum(r['official_qwen3'] for r in catalog['languages'])==119
    rows=coverage(['en','hi','ja','ar','fa'],['ben_Beng','swh_Latn','yor_Latn','tel_Telu'],cfg)
    byname={r['language']:r for r in rows}
    for name in ('English','Hindi','Japanese'):assert byname[name]['source']==cfg.wiki_dataset
    for name in ('Bengali','Swahili','Yoruba','Telugu'):assert byname[name]['source']==cfg.fineweb_dataset
    assert byname['Najdi Arabic']['coverage_status']=='Related-language or dialect proxy'
    assert byname['Najdi Arabic']['config']=='ar'
    assert not byname['Yoruba']['official_qwen3']


def test_streaming_mixed_sources_and_disjoint_partitions(tmp_path,monkeypatch):
    import src.multilingual_data as helper
    import src.wiki40b_loader as loader
    cfg=Config(mode='smoke',output_dir=str(tmp_path),training_languages=('English','Bengali'),train_seq_len=4,
               train_tokens_per_language=16,validation_tokens_per_language=8,ranking_tokens_per_language=8)
    monkeypatch.setattr(helper,'train_configs',lambda dataset,rev:['en'] if dataset==cfg.wiki_dataset else ['ben_Beng'])
    calls=[]
    class Stream(list):
        def shuffle(self,**kwargs):return self
    def load(dataset,**kwargs):
        calls.append((dataset,kwargs))
        assert kwargs['streaming'] and kwargs['split']=='train'
        return Stream(dict(text=f'{dataset} content {i}',wikidata_id=str(i),id=str(i)) for i in range(100))
    monkeypatch.setattr(loader,'load_dataset',load)
    tok=SimpleNamespace(name_or_path='tiny',eos_token_id=2,encode=lambda *a,**kw:[3,4,5,6])
    rev={cfg.wiki_dataset:'wiki-pinned',cfg.fineweb_dataset:'fw-pinned'}
    data,manifest=prepare_data(cfg,tok,rev)
    assert len(calls)==2 and {c[0] for c in calls}==set(rev)
    assert all(c[1]['revision']==rev[c[0]] for c in calls)
    assert set(data)=={'English','Bengali'}
    ids={k:set() for k in ('train','validation','ranking')}
    hashes={k:set() for k in ids}
    for item in manifest['languages'].values():
        assert item['tokens']==dict(train=16,validation=8,ranking=8)
        for part,docs in item['documents'].items():
            ids[part].update(d['id'] for d in docs); hashes[part].update(d['sha256'] for d in docs)
    for a,b in [('train','validation'),('train','ranking'),('validation','ranking')]:
        assert not ids[a]&ids[b] and not hashes[a]&hashes[b]
    assert len(balanced_epoch(data,4,42,0))==8
    assert prepare_data(cfg,tok,rev)[0]==data and len(calls)==2
    with pytest.raises(RuntimeError,match='cache settings'):
        prepare_data(cfg,tok,dict(rev,**{cfg.fineweb_dataset:'changed'}))


@pytest.mark.parametrize('text,complete,answer',[
    ('reason\n#### 42\nQuestion: unrelated\n#### 99',True,'42'),
    ('reason\n#### 42\n#### 99',True,'42'),
    ('Question: unrelated\n#### 99',True,None),
    ('reason\n#### 123',False,None),
    ('reason\n#### 42\nmore',False,'42'),
    ('reason\n#### 42',True,'42'),
    ('reason\n42',True,'42'),
    ('reason\n4 or 5',True,None),
])
def test_first_completed_mgsm_answer(text,complete,answer):
    assert extract_answer(text,complete=complete)==answer


def training_fixture(tmp_path,monkeypatch,max_steps=0):
    from src.trainer import AnchoredExperiment
    e=AnchoredExperiment.__new__(AnchoredExperiment)
    cfg=Config.load('config.smoke.json');cfg.output_dir=str(tmp_path);cfg.max_optimizer_steps=max_steps
    e.cfg=cfg;e.out=tmp_path;e.signature='test'
    for folder in ('checkpoints','logs','manifests'): (tmp_path/folder).mkdir()
    e.data={lang:dict(train=list(range(512))) for lang in cfg.training_languages}
    e.merger=torch.nn.Linear(1,1);e.optimizer=torch.optim.SGD(e.merger.parameters(),lr=.01)
    e.state=dict(epoch=0,cursor=0,step=0,seconds=0.,best=None,batch_size=4,last_validation=0.)
    e.logs=[];e.validation=[]
    e.restore=lambda:False;e.preflight=lambda:None;e.tune_batch=lambda:None;e.integrity=lambda:None
    e.validate=lambda:e.validation.append(dict(step=e.state['step'],teacher_ce=1.,student_ce=1.))
    e.save=lambda:torch.save({'state':e.state},tmp_path/'checkpoints/latest.pt')
    def loss(batch,plan):
        value=(e.merger.weight-1).square().sum()
        return value,{'loss':value.detach()},[len(batch)//2,len(batch)-len(batch)//2]
    e.dual=SimpleNamespace(workers=[None,None],sync=lambda:None,clear=lambda:e.optimizer.zero_grad(),
                           loss=loss,reduce=lambda:None,synchronize=lambda:None)
    e.scaler=SimpleNamespace(scale=lambda x:x,unscale_=lambda o:None,get_scale=lambda:1.,step=lambda o:o.step(),update=lambda:None)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *a:0)
    return e


def test_five_complete_epochs_without_step_cap(tmp_path,monkeypatch):
    e=training_fixture(tmp_path,monkeypatch);e.train()
    done=json.loads((tmp_path/'manifests/training_complete.json').read_text())
    assert done['state']['epoch']==5 and done['state']['step']==35
    metrics=json.loads((tmp_path/'logs/training_metrics.json').read_text())
    assert set(metrics['actual_training_tokens_per_language'].values())=={5*512}
    assert (tmp_path/'checkpoints/merger_final.pt').is_file()


def test_step_cap_is_incomplete_and_checkpointed(tmp_path,monkeypatch):
    e=training_fixture(tmp_path,monkeypatch,max_steps=1)
    with pytest.raises(RuntimeError,match='Training incomplete'):e.train()
    assert not (tmp_path/'manifests/training_complete.json').exists()
    assert not (tmp_path/'checkpoints/merger_final.pt').exists()
    assert (tmp_path/'checkpoints/latest.pt').exists()
