import argparse
import json
from pathlib import Path
import src
from config import Config

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['smoke','search','export','prepare','preflight','train','rank-export','benchmark-preflight','benchmark','report'])
    parser.add_argument('--config',default='config.json'); parser.add_argument('--variant',default='all')
    parser.add_argument('--benchmark',choices=['MMMLU','INCLUDE','MGSM','all'],default='all')
    args=parser.parse_args(); cfg=Config.load(args.config)
    if args.stage=='smoke':
        if cfg.mode!='smoke':raise ValueError('Use config.smoke.json for the smoke command')
        import subprocess,sys
        for stage in ('train','search','export','benchmark','report'):
            subprocess.run([sys.executable,__file__,stage,'--config',args.config],check=True)
        return
    if args.stage=='report':
        from src.reporting import make_report
        print(make_report(cfg)); return
    from src.trainer import AnchoredExperiment,prepare_run
    if args.stage=='train' and (Path(cfg.output_dir)/'manifests/training_complete.json').exists():
        from src.utils import file_digest,digest
        import torch
        lock=prepare_run(cfg); out=Path(cfg.output_dir)
        done=json.loads((out/'manifests/training_complete.json').read_text())
        checkpoint=out/'checkpoints/merger_final.pt'
        if done['state']['epoch']!=cfg.epochs:raise RuntimeError('Training completion marker contains incomplete epochs')
        if file_digest(checkpoint)!=done['checkpoint_sha256']:raise RuntimeError('Final checkpoint changed')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        expected=digest(dict(run=lock,data=saved['data_manifest'],quantization=saved['quantization']))
        if expected!=saved['signature'] or expected!=done['signature']:raise RuntimeError('Completed training signature mismatch')
        print('Training already complete; reusing verified final checkpoint'); return
    if args.stage in ('preflight','train','rank-export','search','export'):
        experiment=AnchoredExperiment(cfg)
        if args.stage=='preflight':experiment.preflight()
        elif args.stage=='train':experiment.train()
        elif args.stage in ('search','export'):
            from src.ce_search import search_and_export
            search_and_export(experiment,export=args.stage=='export')
        else:experiment.rank_and_export()
        return
    from transformers import AutoTokenizer
    from src.wiki40b_loader import prepare_data
    from src.benchmark_evaluation import prepare_benchmarks,evaluate_variant,variants
    lock=prepare_run(cfg); tokenizer=AutoTokenizer.from_pretrained(cfg.model_id,revision=lock['model_revision'])
    if args.stage=='prepare':
        prepare_data(cfg,tokenizer,lock['dataset_revisions']); return
    manifest=prepare_benchmarks(cfg,lock,tokenizer)
    if args.stage=='prepare':return
    if args.stage=='benchmark-preflight':
        print('Benchmark prompts validated:',len(manifest['jobs']),'language jobs'); return
    # Filtering jobs leaves the complete frozen protocol signature intact.
    if args.benchmark!='all':manifest=dict(manifest,jobs=[j for j in manifest['jobs'] if j['benchmark']==args.benchmark])
    specs=variants(cfg,lock)
    if args.variant!='all':
        specs=[v for v in specs if v['name']==args.variant]
        if not specs:raise ValueError('Unknown variant')
    for spec in specs:evaluate_variant(cfg,lock,manifest,spec)

if __name__=='__main__':main()
