import copy
import gc
from threading import RLock
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from .utils import module_hash, tensor_hash,atomic_json,file_digest
from pathlib import Path
import json

# Transformers/Accelerate temporarily patch process-global initialization state.
# All model construction in benchmark threads must share this lock; inference
# itself remains parallel across devices.
MODEL_LOAD_LOCK = RLock()

def nf4_config():
    return BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                             bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)

def load_model(model_id, revision, device, quantized=False):
    kwargs=dict(revision=revision,torch_dtype=torch.float16,attn_implementation='sdpa',device_map={'':str(device)})
    if quantized: kwargs['quantization_config']=nf4_config()
    with MODEL_LOAD_LOCK:
        return AutoModelForCausalLM.from_pretrained(model_id,**kwargs).eval().requires_grad_(False)

@torch.no_grad()
def dequantized_copy(quantized, device):
    """Materialize only tensors actually read from NF4 storage, including unchanged embeddings/norms."""
    import bitsandbytes as bnb
    config=copy.deepcopy(quantized.config)
    # The new model stores dense FP16 tensors, so neither NF4 settings nor its
    # runtime dtype marker belongs in its config. The latter is a torch.dtype
    # that Transformers 4.51 cannot serialize in save_pretrained.
    for name in ('quantization_config','_pre_quantization_dtype'):
        if hasattr(config,name): delattr(config,name)
    config._attn_implementation='sdpa'
    # CPU construction avoids holding a second initialization on the GPU.
    with MODEL_LOAD_LOCK:
        dense=AutoModelForCausalLM.from_config(config).half().to(device)
    quant_count=0; hashes={}
    originals=dict(quantized.named_parameters())
    for name,target in dense.named_parameters():
        source=originals[name]
        if isinstance(source,bnb.nn.Params4bit):
            if source.quant_state is None: raise RuntimeError(f'NF4 tensor was never quantized: {name}')
            value=bnb.functional.dequantize_4bit(source.data,source.quant_state)
            quant_count+=1
        else: value=source
        if value.shape!=target.shape: raise ValueError(f'Dequantized shape mismatch: {name}')
        target.copy_(value.to(device=device,dtype=torch.float16)); hashes[name]=tensor_hash(target)
    if quant_count==0: raise RuntimeError('No quantized source matrices found; refusing FP16-first merging')
    return dense.eval().requires_grad_(False),dict(nf4_matrices=quant_count,source_tensor_hashes=hashes,
                                                   dense_hash=module_hash(dense))

def load_training_models(cfg,revision,device):
    teacher=load_model(cfg.model_id,revision,device)
    baseline_dir=Path(cfg.output_dir)/'reference_models/original_nf4'
    baseline_manifest=Path(cfg.output_dir)/'manifests/original_nf4.json'
    identity=dict(model_id=cfg.model_id,revision=revision,quantization=nf4_config().to_dict())
    if baseline_manifest.exists():
        info=json.loads(baseline_manifest.read_text())
        if info['identity']!=identity or any(not (baseline_dir/name).is_file() or file_digest(baseline_dir/name)!=sha for name,sha in info['files'].items()):
            raise RuntimeError('Original NF4 cache identity/integrity mismatch')
        quant=load_model(str(baseline_dir),None,device,True)
        verify_nf4(quant)
    else:
        quant=load_model(cfg.model_id,revision,device,True)
        verify_nf4(quant)
        baseline_dir.mkdir(parents=True,exist_ok=True)
        save_nf4_checkpoint(quant,baseline_dir)
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(cfg.model_id,revision=revision).save_pretrained(baseline_dir)
        files={str(p.relative_to(baseline_dir)):file_digest(p) for p in baseline_dir.rglob('*') if p.is_file()}
        info=dict(identity=identity,files=files,serialized_bytes=sum((baseline_dir/name).stat().st_size for name in files))
        atomic_json(info,baseline_manifest)
    dense,manifest=dequantized_copy(quant,device)
    manifest.update(quantization=nf4_config().to_dict(),teacher_hash=module_hash(teacher),
                    teacher_bytes=teacher.get_memory_footprint(),nf4_bytes=quant.get_memory_footprint(),
                    nf4_serialized_bytes=info['serialized_bytes'])
    if manifest['nf4_bytes'] >= manifest['teacher_bytes']:
        raise RuntimeError('NF4 did not reduce measured model storage')
    del quant; gc.collect(); torch.cuda.empty_cache()
    return teacher,dense,manifest


@torch.no_grad()
def quantize_dense_nf4(model):
    """Convert a private dense model in memory; never serialize FP16 candidates."""
    import bitsandbytes as bnb
    from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
    from transformers.utils.quantization_config import QuantizationMethod
    count=0
    for name,module in list(model.named_modules()):
        if not isinstance(module,torch.nn.Linear) or name=='lm_head':continue
        parent_name,_,child=name.rpartition('.')
        parent=model.get_submodule(parent_name) if parent_name else model
        device=module.weight.device
        linear=bnb.nn.Linear4bit(module.in_features,module.out_features,bias=module.bias is not None,
                                compute_dtype=torch.float16,compress_statistics=True,quant_type='nf4')
        linear.weight=bnb.nn.Params4bit(module.weight.detach().cpu(),requires_grad=False,
                                      compress_statistics=True,quant_type='nf4').to(device)
        if module.bias is not None:linear.bias=torch.nn.Parameter(module.bias.detach().clone(),requires_grad=False)
        setattr(parent,child,linear); count+=1
    if not count:raise RuntimeError('No eligible NF4 linear layers')
    config=nf4_config(); model.config.quantization_config=config
    model.hf_quantizer=Bnb4BitHfQuantizer(config)
    model.quantization_method=QuantizationMethod.BITS_AND_BYTES
    model.is_quantized=True; model.is_loaded_in_4bit=True; model.is_4bit_serializable=True
    return model.eval().requires_grad_(False)


def verify_nf4(model):
    """Check the actual eligible weights and compute/storage policy, not a label."""
    import bitsandbytes as bnb
    count=0
    for name,module in model.named_modules():
        if not name.startswith('model.layers.') or not name.endswith(('_proj',)):
            continue
        if not isinstance(module,bnb.nn.Linear4bit) or not isinstance(module.weight,bnb.nn.Params4bit):
            raise RuntimeError(f'Eligible matrix is not NF4: {name}')
        state=module.weight.quant_state
        if state is None or state.quant_type!='nf4' or not state.nested:
            raise RuntimeError(f'Invalid NF4/double-quant state: {name}')
        if module.compute_dtype!=torch.float16:raise RuntimeError('NF4 compute dtype must be FP16')
        count+=1
    if count==0:raise RuntimeError('No NF4 matrices found')
    return count


def save_nf4_checkpoint(model,destination):
    verify_nf4(model)
    marker=getattr(model.config,'_pre_quantization_dtype',None)
    if hasattr(model.config,'_pre_quantization_dtype'):del model.config._pre_quantization_dtype
    try:model.save_pretrained(destination,safe_serialization=True)
    finally:
        if marker is not None:model.config._pre_quantization_dtype=marker
