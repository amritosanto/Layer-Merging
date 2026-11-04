"""Metadata-only discovery and conservative Qwen3 language/source mapping."""
import json
from pathlib import Path
import yaml
from huggingface_hub import hf_hub_download

MAPPING_PATH=Path(__file__).with_name('language_coverage.json')
SMOKE_LANGUAGES=('English','Hindi','Japanese','Bengali','Swahili','Yoruba','Telugu')
WIKI_SCRIPTS={'en':'Latn','hi':'Deva','ja':'Jpan','ar':'Arab','fa':'Arab','he':'Hebr','zh-cn':'Hans','zh-tw':'Hant',
              'ru':'Cyrl','bg':'Cyrl','uk':'Cyrl','be':'Cyrl','sr':'Cyrl','el':'Grek','ko':'Hang','th':'Thai'}


def train_configs(dataset,revision):
    # Read one small dataset card, never enumerate remote shards for discovery.
    path=hf_hub_download(dataset,'README.md',repo_type='dataset',revision=revision)
    metadata=yaml.safe_load(Path(path).read_text(encoding='utf-8').split('---',2)[1])
    return sorted(c['config_name'] for c in metadata.get('configs',[])
                  if any(s.get('split')=='train' for s in c.get('data_files',[])))


def coverage(wiki_configs,fineweb_configs,cfg):
    mapping=json.loads(MAPPING_PATH.read_text(encoding='utf-8'))
    rows=[]
    def exact(r):
        if r['wiki_config'] in wiki_configs:
            script=WIKI_SCRIPTS.get(r['wiki_config'])
            if script is None and r['fineweb_configs']:script=r['fineweb_configs'][0].rsplit('_',1)[-1]
            return cfg.wiki_dataset,r['wiki_config'],script or 'not specified'
        fw=[c for c in r['fineweb_configs'] if c in fineweb_configs]
        return (cfg.fineweb_dataset,fw[0],fw[0].rsplit('_',1)[-1]) if fw else (None,None,None)
    byname={r['language']:r for r in mapping['languages']}
    for r in mapping['languages']:
        source,config,script=exact(r); status='Exact coverage' if source else 'Unavailable'
        proxy=None
        if not source:
            parent=byname.get(r.get('proxy_language'))
            if parent:source,config,script=exact(parent); proxy=r['proxy_language']
            if not source and r.get('proxy_wiki_config') in wiki_configs:
                source,config,script=cfg.wiki_dataset,r['proxy_wiki_config'],'Latn';proxy='generic Norwegian'
            if source:status='Related-language or dialect proxy'
        rows.append(dict(language=r['language'],official_qwen3=r['official_qwen3'],source=source,
                         config=config,script=script,coverage_status=status,proxy_for=proxy,
                         mapping_note=r['mapping_note'],sampled=False,requested_tokens=0,
                         actual_train_tokens=0,actual_validation_tokens=0,actual_ranking_tokens=0))
    return rows
