import argparse
import json
from pathlib import Path
import src
from config import Config

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['smoke','search','export','prepare','preflight','train','rank-export','benchmark-preflight','benchmark','report','sft-prepare','sft-original','sft-merged','sft-verify'])
    parser.add_argument('--config',default='config.json'); parser.add_argument('--variant',default='all')
    parser.add_argument('--benchmark',choices=['MMMLU','INCLUDE','MGSM','all'],default='all')
    args=parser.parse_args(); cfg=Config.load(args.config)
    if args.stage in ('sft-original','sft-merged'):
        import os
        # One independent adapter fit per stage, avoiding QLoRA DataParallel.
        os.environ['CUDA_VISIBLE_DEVICES']='0' if args.stage=='sft-original' or cfg.single_gpu_debug else '1'
    if args.stage=='smoke':
        if cfg.mode!='smoke':raise ValueError('Use config.smoke.json for the smoke command')
        import subprocess,sys
        stages=['train','search','export']+(['sft-prepare','sft-original','sft-merged','sft-verify'] if cfg.post_sft else [])+['benchmark','report']
        for stage in stages:
            subprocess.run([sys.executable,__file__,stage,'--config',args.config],check=True)
        return
    if args.stage=='report':
        if cfg.post_sft:
            from src.sft_reporting import make_sft_report
            print(make_sft_report(cfg));return
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
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    if args.stage=='prepare':
        prepare_data(cfg,tokenizer,lock['dataset_revisions']); return
    manifest=prepare_benchmarks(cfg,lock,tokenizer)
    if args.stage.startswith('sft-'):
        if not cfg.post_sft:raise ValueError('Use config.smoke_sft13.json')
        from src.sft_data import prepare_sft
        from src.sft_training import train_sft,verify_sft
        if args.stage=='sft-prepare':prepare_sft(cfg,lock,tokenizer,manifest)
        elif args.stage=='sft-verify':verify_sft(cfg,lock,tokenizer,manifest)
        else:train_sft(cfg,lock,tokenizer,manifest,'original_fp16_sft' if args.stage=='sft-original' else 'merged_nf4_sft')
        return
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
