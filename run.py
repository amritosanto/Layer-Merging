import argparse
import json
from pathlib import Path
import src
from config import Config

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','preflight','train','rank-export','benchmark-preflight','benchmark','report'])
    parser.add_argument('--config',default='config.json'); parser.add_argument('--variant',default='all')
    parser.add_argument('--benchmark',choices=['MMMLU','INCLUDE','MGSM','all'],default='all')
    args=parser.parse_args(); cfg=Config.load(args.config)
    if args.stage=='report':
        from src.reporting import make_report
        print(make_report(cfg)); return
    from src.trainer import AnchoredExperiment,prepare_run
    if args.stage in ('preflight','train','rank-export'):
        experiment=AnchoredExperiment(cfg)
        if args.stage=='preflight':experiment.preflight()
        elif args.stage=='train':experiment.train()
        else:experiment.rank_and_export()
        return
    from transformers import AutoTokenizer
    from src.wiki40b_loader import prepare_data
    from src.benchmark_evaluation import prepare_benchmarks,evaluate_variant,variants
    lock=prepare_run(cfg); tokenizer=AutoTokenizer.from_pretrained(cfg.model_id,revision=lock['model_revision'])
    if args.stage=='prepare':prepare_data(cfg,tokenizer,lock['dataset_revisions'][cfg.wiki_dataset])
    manifest=prepare_benchmarks(cfg,lock,tokenizer)
    if args.stage=='prepare':return
    if args.stage=='benchmark-preflight':
        spec=dict(name='original_fp16',path=cfg.model_id,revision=lock['model_revision'],nf4=False,layers=28)
        evaluate_variant(cfg,lock,manifest,spec,smoke=True)
        from src.utils import atomic_json
        atomic_json(dict(passed=True,signature=manifest['signature']),Path(cfg.output_dir)/f'manifests/benchmark_preflight_{cfg.evaluation_mode}.json'); return
    preflight=Path(cfg.output_dir)/f'manifests/benchmark_preflight_{cfg.evaluation_mode}.json'
    if not preflight.exists() or json.loads(preflight.read_text())['signature']!=manifest['signature']:
        raise RuntimeError('Run benchmark-preflight before benchmark execution')
    # Filtering jobs leaves the complete frozen protocol signature intact.
    if args.benchmark!='all':manifest=dict(manifest,jobs=[j for j in manifest['jobs'] if j['benchmark']==args.benchmark])
    specs=variants(cfg,lock)
    if args.variant!='all':
        specs=[v for v in specs if v['name']==args.variant]
        if not specs:raise ValueError('Unknown variant')
    for spec in specs:evaluate_variant(cfg,lock,manifest,spec)

if __name__=='__main__':main()
