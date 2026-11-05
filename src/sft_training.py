"""Paired LoRA/QLoRA stages, immutable audited data, and post-SFT evaluation gate."""
import gc
import json
import math
from pathlib import Path
import torch
from .utils import atomic_json,digest,file_digest,seed_all,module_hash
from .quantization import load_model,verify_nf4
from .sft_data import load_prepared

NAMES=('original_fp16_sft','merged_nf4_sft')


def file_map(root,folder):
    return {str(p.relative_to(root)):file_digest(p) for p in folder.rglob('*') if p.is_file()}


def check_files(out,files):
    if not files or any(not (out/p).is_file() or file_digest(out/p)!=h for p,h in files.items()):
        raise RuntimeError('SFT/source files missing or changed')


def inference_precision(model,nf4):
    if nf4:return verify_nf4(model)
    if getattr(model,'is_loaded_in_4bit',False):raise RuntimeError('Original SFT model must be FP16')
    if any(p.is_floating_point() and p.dtype!=torch.float16 for p in model.parameters()):
        raise RuntimeError('Original SFT inference weights are not FP16')


def load_inference(spec,device):
    model=load_model(spec['path'],spec.get('revision'),device,spec['nf4'])
    if spec.get('adapter'):
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,spec['adapter'],is_trainable=False).get_base_model()
    model.eval().requires_grad_(False);inference_precision(model,spec['nf4'])
    return model


def response_collator(rows,pad_token_id):
    width=max(len(r['input_ids']) for r in rows)
    ids=torch.full((len(rows),width),pad_token_id,dtype=torch.long)
    labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids)
    for i,row in enumerate(rows):
        n=len(row['input_ids']);ids[i,:n]=torch.tensor(row['input_ids']);labels[i,:n]=torch.tensor(row['labels']);mask[i,:n]=1
    return dict(input_ids=ids,labels=labels,attention_mask=mask)


def context(cfg,lock,manifest):
    out=Path(cfg.output_dir);data,audit=load_prepared(cfg,lock,manifest)
    final=json.loads((out/'manifests/final_export.json').read_text())
    check_files(out,final['files'])
    if final['completed_epochs']!=cfg.epochs:raise RuntimeError('Incomplete merger training')
    identity=digest([lock,audit['identity'],audit['accepted_digest'],final['files'],'paired-sft-v1'])
    return out,data,audit,final,identity


def complete_checkpoint(directory):
    candidates=[]
    for p in directory.glob('checkpoint-*'):
        marker=p/'complete.json'
        if marker.exists():
            record=json.loads(marker.read_text())
            if all((p/name).is_file() and file_digest(p/name)==sha for name,sha in record['files'].items()):candidates.append(p)
    return str(max(candidates,key=lambda p:int(p.name.split('-')[-1]))) if candidates else None


def train_sft(cfg,lock,tokenizer,manifest,name):
    if name not in NAMES:raise ValueError(name)
    out,data,audit,final,identity=context(cfg,lock,manifest)
    done=out/f'manifests/{name}.json'
    if done.exists():
        record=json.loads(done.read_text())
        if record['identity']!=identity:raise RuntimeError('SFT run identity changed')
        check_files(out,record['files']);return record
    from transformers import Trainer,TrainingArguments,TrainerCallback
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    if torch.cuda.device_count()!=1:raise RuntimeError('Run each SFT stage with one visible GPU; CLI sets CUDA_VISIBLE_DEVICES automatically')
    seed_all(cfg.seed);nf4=name=='merged_nf4_sft'
    model=load_model(str(out/'model_exports/best_learned_nf4') if nf4 else cfg.model_id,
                     None if nf4 else lock['model_revision'],'cuda:0',nf4)
    if nf4:model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    else:model.enable_input_require_grads()
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=cfg.sft_lora_rank,lora_alpha=cfg.sft_lora_alpha,lora_dropout=0.,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none',task_type='CAUSAL_LM'))
    if any(p.requires_grad and 'lora_' not in n for n,p in model.named_parameters()):raise RuntimeError('Only LoRA may be trained')
    folder=out/'sft'/name;folder.mkdir(parents=True,exist_ok=True)
    identity_path=folder/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())['identity']!=identity:raise RuntimeError('Stale SFT checkpoint')
    atomic_json(dict(identity=identity),identity_path)
    class SafeBoundary(TrainerCallback):
        def on_epoch_end(self,args,state,control,**kwargs):
            control.should_save=True;return control
        def on_save(self,args,state,control,**kwargs):
            path=Path(args.output_dir)/f'checkpoint-{state.global_step}'
            required=('trainer_state.json','optimizer.pt','scheduler.pt','rng_state.pth','adapter_model.safetensors')
            if not all((path/p).is_file() for p in required):raise RuntimeError('Incomplete SFT optimizer checkpoint')
            atomic_json(dict(files={str(p.relative_to(path)):file_digest(p) for p in path.rglob('*') if p.is_file() and p.name!='complete.json'}),path/'complete.json')
    args=TrainingArguments(output_dir=str(folder),num_train_epochs=cfg.sft_epochs,max_steps=-1,
        per_device_train_batch_size=cfg.sft_batch_size,gradient_accumulation_steps=cfg.sft_gradient_accumulation,
        learning_rate=cfg.sft_learning_rate,lr_scheduler_type='linear',warmup_ratio=.03,weight_decay=0.,
        fp16=True,bf16=False,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        save_strategy='steps',save_steps=cfg.sft_save_steps,save_total_limit=3,logging_steps=10,
        eval_strategy='no',load_best_model_at_end=False,seed=cfg.seed,data_seed=cfg.seed,
        report_to=[],remove_unused_columns=False,dataloader_num_workers=0,optim='adamw_torch',label_names=['labels'])
    samples=[dict(input_ids=r['input_ids'],labels=r['labels']) for r in data['examples']]
    trainer=Trainer(model=model,args=args,train_dataset=samples,
                    data_collator=lambda rows:response_collator(rows,tokenizer.pad_token_id),callbacks=[SafeBoundary()])
    trainer.train(resume_from_checkpoint=complete_checkpoint(folder))
    if trainer.state.epoch is None or trainer.state.epoch<cfg.sft_epochs:raise RuntimeError('SFT epochs incomplete')
    adapter=folder/'adapter';model.save_pretrained(adapter,safe_serialization=True)
    epochs=trainer.state.epoch;steps=trainer.state.global_step
    if nf4:
        spec=dict(name=name,path=str(out/'model_exports/best_learned_nf4'),revision=None,nf4=True,
                  adapter=str(adapter),layers=final['remaining_layers'])
        files=dict(final['files'],**file_map(out,adapter))
    else:
        merged=model.merge_and_unload(safe_merge=True).half().eval();merged.config.use_cache=True
        inference_precision(merged,False)
        destination=out/'model_exports/original_fp16_sft';destination.mkdir(parents=True,exist_ok=True)
        merged.save_pretrained(destination,safe_serialization=True);tokenizer.save_pretrained(destination)
        spec=dict(name=name,path=str(destination),revision=None,nf4=False,layers=28)
        files=file_map(out,destination);del merged
    spec['stored_bytes']=sum((out/p).stat().st_size for p in files)
    spec['adapter_bytes']=sum(p.stat().st_size for p in adapter.rglob('*') if p.is_file()) if nf4 else 0
    record=dict(identity=identity,files=files,spec=spec,completed_epochs=epochs,optimizer_steps=steps,
                accepted_digest=audit['accepted_digest'],benchmark_signature=manifest['signature'],
                tokens_per_language_per_epoch=audit['tokens_per_language'],stop_reason='epochs',
                inference_format='NF4 base plus unmerged LoRA adapter' if nf4 else 'merged FP16')
    atomic_json(record,done)
    del trainer,model;gc.collect();torch.cuda.empty_cache()
    return record


def verified_specs(cfg,lock,manifest=None,require_gate=True):
    out=Path(cfg.output_dir)
    if manifest is None:manifest=json.loads((out/f'manifests/benchmark_prompts_{cfg.evaluation_mode}.json').read_text(encoding='utf-8'))
    _,_,audit,_,identity=context(cfg,lock,manifest)
    records=[]
    for name in NAMES:
        path=out/f'manifests/{name}.json'
        if not path.exists():raise RuntimeError('Both SFT models must complete before evaluation')
        r=json.loads(path.read_text())
        if r['identity']!=identity or r['completed_epochs']<cfg.sft_epochs or r['accepted_digest']!=audit['accepted_digest']:
            raise RuntimeError('SFT completion/data identity mismatch')
        check_files(out,r['files']);records.append(r)
    if records[0]['optimizer_steps']!=records[1]['optimizer_steps']:raise RuntimeError('Paired SFT schedules differ')
    if require_gate:
        gate=out/'manifests/sft_verified.json'
        if not gate.exists() or json.loads(gate.read_text()).get('records_digest')!=digest(records):
            raise RuntimeError('Run sft-verify successfully before benchmarking')
    return [r['spec'] for r in records]


def verify_sft(cfg,lock,tokenizer,manifest):
    specs=verified_specs(cfg,lock,manifest,require_gate=False);out=Path(cfg.output_dir)
    hashes={}
    for spec in specs:
        model=load_inference(spec,'cuda:0')
        ids=tokenizer('A short verification.',return_tensors='pt').input_ids.to('cuda:0')
        with torch.no_grad():
            logits=model(ids,use_cache=False).logits
            if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite post-SFT inference')
            if model.generate(ids,max_new_tokens=2,do_sample=False,pad_token_id=tokenizer.pad_token_id).shape[1]<=ids.shape[1]:raise RuntimeError('Post-SFT generation failed')
        if len(model.model.layers)!=spec['layers']:raise RuntimeError('Post-SFT layer mismatch')
        hashes[spec['name']]=module_hash(model)
        del model;gc.collect();torch.cuda.empty_cache()
    records=[json.loads((out/f'manifests/{name}.json').read_text()) for name in NAMES]
    atomic_json(dict(records_digest=digest(records),weight_hashes=hashes,passed=True),out/'manifests/sft_verified.json')
    return specs
