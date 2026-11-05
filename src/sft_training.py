"""Paired LoRA/QLoRA stages, immutable audited data, and post-SFT evaluation gate."""
import gc
import json
import math
import time
from pathlib import Path
import torch
from .utils import atomic_json,digest,file_digest,seed_all,module_hash,diagnostic_json
from .quantization import load_model,verify_nf4
from .sft_data import load_prepared

NAMES=('original_fp16_sft','merged_nf4_sft')


class PairedSampler(torch.utils.data.Sampler):
    """Order independent of model depth, dropout RNG and other global RNG use."""
    def __init__(self,data,seed,bucket=False):self.data=data;self.seed=seed;self.epoch=0;self.bucket=bucket
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return len(self.data)
    def __iter__(self):
        order=torch.randperm(len(self.data),generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
        if self.bucket:
            order=[i for start in range(0,len(order),64) for i in sorted(order[start:start+64],key=lambda j:len(self.data[j]['input_ids']),reverse=True)]
        return iter(order)


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
    identity=digest([lock,audit['identity'],audit['accepted_digest'],final['files'],'paired-heavy-sft-v2'])
    return out,data,audit,final,identity


def complete_checkpoint(directory):
    candidates=[]
    for p in directory.glob('checkpoint-*'):
        marker=p/'complete.json'
        if marker.exists():
            record=json.loads(marker.read_text())
            if all((p/name).is_file() and file_digest(p/name)==sha for name,sha in record['files'].items()):candidates.append(p)
    return str(max(candidates,key=lambda p:int(p.name.split('-')[-1]))) if candidates else None


def exposure_trainer_class(cfg,exposure):
    from transformers import Trainer
    class ExposureTrainer(Trainer):
        def _get_train_sampler(self):return PairedSampler(self.train_dataset,cfg.seed,cfg.sft_bucket_by_length)
        def training_step(self,model,inputs,num_items_in_batch=None):
            languages=inputs.pop('_languages');counts=inputs['attention_mask'].sum(1).tolist()
            loss=super().training_step(model,inputs,num_items_in_batch)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite SFT training loss')
            for language,count in zip(languages,counts):
                exposure['tokens'][language]+=int(count);exposure['examples'][language]+=1
            return loss
    return ExposureTrainer


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
    if nf4:model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=cfg.sft_gradient_checkpointing,gradient_checkpointing_kwargs={'use_reentrant':False})
    else:model.enable_input_require_grads()
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=cfg.sft_lora_rank,lora_alpha=cfg.sft_lora_alpha,lora_dropout=cfg.sft_lora_dropout,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none',task_type='CAUSAL_LM'))
    if any(p.requires_grad and 'lora_' not in n for n,p in model.named_parameters()):raise RuntimeError('Only LoRA may be trained')
    if cfg.sft_gradient_checkpointing:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    folder=out/'sft'/name;folder.mkdir(parents=True,exist_ok=True)
    identity_path=folder/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())['identity']!=identity:raise RuntimeError('Stale SFT checkpoint')
    atomic_json(dict(identity=identity),identity_path)
    from .sft_runtime import autotune,completed_state,restore_adapter,finite_gradients,gpu_sample
    resume=complete_checkpoint(folder)
    completed=completed_state(resume,cfg,data,audit)
    if completed:
        state,exposure=completed
        restore_adapter(model,resume)
        from types import SimpleNamespace
        print('Finalizing verified completed SFT checkpoint; no optimizer steps will run',flush=True)
        return finalize_sft(cfg,tokenizer,manifest,name,out,data,audit,final,identity,model,exposure,SimpleNamespace(**state),recovered=True)
    selected_batch=autotune(cfg,model,data,tokenizer.pad_token_id,folder,identity)
    seed_all(cfg.seed)
    exposure=dict(tokens={l:0 for l in cfg.training_languages},examples={l:0 for l in cfg.training_languages},steps=[])
    if resume:
        exposure_path=Path(resume)/'exposure.json'
        if not exposure_path.exists():raise RuntimeError('Missing SFT exposure checkpoint')
        exposure=json.loads(exposure_path.read_text())
    ExposureTrainer=exposure_trainer_class(cfg,exposure)
    class SafeBoundary(TrainerCallback):
        last_monitor=0.
        def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):finite_gradients(model)
        def on_step_begin(self,args,state,control,**kwargs):
            self.started=time.monotonic();self.tokens=sum(exposure['tokens'].values())
        def on_step_end(self,args,state,control,**kwargs):
            torch.cuda.synchronize();elapsed=time.monotonic()-self.started
            exposure['steps'].append(dict(step=state.global_step,seconds=elapsed,tokens=sum(exposure['tokens'].values())-self.tokens,
                peak_bytes=torch.cuda.max_memory_allocated()))
            if time.monotonic()-self.last_monitor>=cfg.sft_monitor_interval:
                sample=dict(step=state.global_step,step_seconds=elapsed,input_tokens_per_second=exposure['steps'][-1]['tokens']/max(elapsed,1e-9),optimizer_updates_per_second=1/max(elapsed,1e-9),**gpu_sample());exposure.setdefault('gpu_samples',[]).append(sample);self.last_monitor=time.monotonic()
                print('GPU telemetry:',sample,flush=True)
            if len(exposure['steps'])==10:
                seconds=sum(r['seconds'] for r in exposure['steps'])/10
                print('SFT estimate from ten updates (not a cutoff):',seconds*state.max_steps/3600,'hours per variant; planning target',cfg.planning_hours,'hours for the entire experiment',flush=True)
                if seconds*state.max_steps/3600>cfg.planning_hours:print('Projected overrun: continuing all configured epochs and preserving checkpoints',flush=True)

        def on_epoch_end(self,args,state,control,**kwargs):
            control.should_save=True;return control
        def on_save(self,args,state,control,**kwargs):
            path=Path(args.output_dir)/f'checkpoint-{state.global_step}'
            atomic_json(exposure,path/'exposure.json')
            required=('trainer_state.json','optimizer.pt','scheduler.pt','rng_state.pth','adapter_model.safetensors')
            if not all((path/p).is_file() for p in required):raise RuntimeError('Incomplete SFT optimizer checkpoint')
            atomic_json(dict(files={str(p.relative_to(path)):file_digest(p) for p in path.rglob('*') if p.is_file() and p.name!='complete.json'}),path/'complete.json')
    args=TrainingArguments(output_dir=str(folder),num_train_epochs=cfg.sft_epochs,max_steps=-1,
        per_device_train_batch_size=selected_batch['microbatch'],gradient_accumulation_steps=selected_batch['gradient_accumulation'],
        learning_rate=cfg.sft_learning_rate,lr_scheduler_type='linear',warmup_ratio=.03,weight_decay=0.,
        fp16=True,bf16=False,gradient_checkpointing=cfg.sft_gradient_checkpointing,gradient_checkpointing_kwargs={'use_reentrant':False},
        save_strategy='steps',save_steps=cfg.sft_save_steps,save_total_limit=3,logging_steps=10,
        eval_strategy='epoch' if data.get('validation') else 'no',per_device_eval_batch_size=selected_batch['microbatch'],prediction_loss_only=True,load_best_model_at_end=False,seed=cfg.seed,data_seed=cfg.seed,
        report_to=[],remove_unused_columns=False,dataloader_num_workers=0,optim='adamw_torch',label_names=['labels'])
    samples=[dict(input_ids=r['input_ids'],labels=r['labels'],_language=r['language']) for r in data['examples']]
    evaluation={l:[dict(input_ids=r['input_ids'],labels=r['labels']) for r in data.get('validation',[]) if r['language']==l] for l in cfg.training_languages}
    def collate(rows):
        batch=response_collator(rows,tokenizer.pad_token_id)
        if '_language' in rows[0]:batch['_languages']=[r['_language'] for r in rows]
        return batch
    trainer=ExposureTrainer(model=model,args=args,train_dataset=samples,eval_dataset=evaluation if data.get('validation') else None,
                    data_collator=collate,callbacks=[SafeBoundary()])
    trainer.train(resume_from_checkpoint=resume)
    return finalize_sft(cfg,tokenizer,manifest,name,out,data,audit,final,identity,model,exposure,trainer.state)


def finalize_sft(cfg,tokenizer,manifest,name,out,data,audit,final,identity,model,exposure,state,recovered=False):
    folder=out/'sft'/name;done=out/f'manifests/{name}.json';nf4=name=='merged_nf4_sft'
    if state.epoch is None or state.epoch<cfg.sft_epochs:raise RuntimeError('SFT epochs incomplete')
    expected={l:cfg.sft_epochs*sum(r['language']==l for r in data['examples']) for l in cfg.training_languages}
    if exposure['examples']!=expected:raise RuntimeError('SFT exposure incomplete or duplicated')
    if exposure['tokens']!={l:cfg.sft_epochs*audit['tokens_per_language'][l] for l in cfg.training_languages}:raise RuntimeError('SFT effective token budget mismatch')
    bad_metrics=diagnostic_json(dict(exposure=exposure,loss_history=state.log_history),folder/'training_metrics.json')
    if any(not bool(torch.isfinite(p).all()) for n,p in model.named_parameters() if 'lora_' in n):raise FloatingPointError('Nonfinite trained adapter weights')
    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    from transformers import AutoModelForCausalLM
    import copy
    dense_config=copy.deepcopy(model.config)
    if hasattr(dense_config,'quantization_config'):del dense_config.quantization_config
    with torch.device('meta'):dense=AutoModelForCausalLM.from_config(dense_config)
    architecture_parameters=sum(p.numel() for p in dense.parameters());del dense
    adapter=folder/'adapter';model.save_pretrained(adapter,safe_serialization=True)
    epochs=state.epoch;steps=state.global_step
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
    spec.update(architecture_parameters=architecture_parameters,trainable_adapter_parameters=trainable_parameters)
    record=dict(finalized_from_checkpoint=recovered,nonfinite_diagnostic_metrics=bad_metrics,identity=identity,files=files,spec=spec,completed_epochs=epochs,optimizer_steps=steps,
                accepted_digest=audit['accepted_digest'],benchmark_signature=manifest['signature'],
                tokens_per_language_per_epoch=audit['tokens_per_language'],actual_exposure=exposure,trainable_adapter_parameters=trainable_parameters,stop_reason='epochs',
                inference_format='NF4 base plus unmerged LoRA adapter' if nf4 else 'merged FP16')
    atomic_json(record,done)
    del model;gc.collect();torch.cuda.empty_cache()
    return record


def verified_specs(cfg,lock,manifest=None,require_gate=True):
    out=Path(cfg.output_dir)
    if manifest is None:
        from .benchmark_evaluation import prompt_path
        manifest=json.loads(prompt_path(cfg,audit=True).read_text(encoding='utf-8'))
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
