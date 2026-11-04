import copy
import gc
from threading import RLock
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from .utils import module_hash, tensor_hash

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
    quant=load_model(cfg.model_id,revision,device,True)
    dense,manifest=dequantized_copy(quant,device)
    manifest.update(quantization=nf4_config().to_dict(),teacher_hash=module_hash(teacher),
                    teacher_bytes=teacher.get_memory_footprint(),nf4_bytes=quant.get_memory_footprint())
    if manifest['nf4_bytes'] >= manifest['teacher_bytes']:
        raise RuntimeError('NF4 did not reduce measured model storage')
    del quant; gc.collect(); torch.cuda.empty_cache()
    return teacher,dense,manifest
