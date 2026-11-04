import copy
from contextlib import contextmanager
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModelForCausalLM,AutoTokenizer
from .pair_selection import valid_plan
from .utils import atomic_json,module_hash

@torch.no_grad()
def compress_in_place(source,pairs,method,pair_weights=None):
    """Materialize an evaluation-only control without retaining 20 redundant disk exports."""
    if method not in ('learned','average','drop_first','drop_second'):raise ValueError(method)
    pairs=valid_plan(pairs,len(source.model.layers)); original=source.model.layers; layers=[]; i=0
    while i<len(original):
        if i not in pairs:layers.append(original[i]); i+=1; continue
        if method=='drop_first':block=original[i+1]
        elif method=='drop_second':block=original[i]
        else:
            block=copy.deepcopy(original[i+1]); a,b=dict(original[i].named_parameters()),dict(original[i+1].named_parameters())
            params=pair_weights[i] if method=='learned' else {k:(a[k].float()+b[k].float())/2 for k in a}
            for name,p in block.named_parameters():p.copy_(params[name].to(p.device))
        layers.append(block); i+=2
    source.model.layers=nn.ModuleList(layers); source.config.num_hidden_layers=len(layers)
    for j,block in enumerate(layers):block.self_attn.layer_idx=j
    return source

@torch.no_grad()
def replacement(source,merger,pair,method='learned'):
    a,b=source.model.layers[pair:pair+2]
    if method=='drop_first':return copy.deepcopy(b)
    if method=='drop_second':return copy.deepcopy(a)
    pa,pb=dict(a.named_parameters()),dict(b.named_parameters())
    if method=='learned':params=merger(pa,pb,pair)
    elif method=='average':params={k:(pa[k].float()+pb[k].float())/2 for k in pa}
    else:raise ValueError(method)
    block=copy.deepcopy(b)
    for name,p in block.named_parameters():
        if p.shape!=params[name].shape:raise ValueError(f'Generated tensor shape: {name}')
        p.copy_(params[name])
    return block.eval().requires_grad_(False)

@contextmanager
def variant(source,merger,pairs=(),method='learned'):
    pairs=valid_plan(pairs,len(source.model.layers),allow_empty=True)
    original=source.model.layers; count=source.config.num_hidden_layers
    indices=[block.self_attn.layer_idx for block in original]
    layers=[]; i=0
    while i<count:
        if i in pairs:layers.append(replacement(source,merger,i,method)); i+=2
        else:layers.append(original[i]); i+=1
    try:
        source.model.layers=nn.ModuleList(layers); source.config.num_hidden_layers=len(layers)
        for j,block in enumerate(layers):block.self_attn.layer_idx=j
        yield source
    finally:
        source.model.layers=original; source.config.num_hidden_layers=count
        for block,index in zip(original,indices):block.self_attn.layer_idx=index

@torch.no_grad()
def export_model(source,merger,tokenizer,pairs,destination,method='learned',metadata=None,verify=True):
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    with variant(source,merger,pairs,method) as model:
        model.config.use_cache=True
        model.save_pretrained(destination,safe_serialization=True); tokenizer.save_pretrained(destination)
        count=sum(p.numel() for p in model.parameters())
        manifest=dict(pairs=list(pairs),layers=len(model.model.layers),method=method,representation='FP16 from NF4 source',
                      parameters=count,weight_hash=module_hash(model),merger_hash=module_hash(merger),**(metadata or {}))
    manifest['stored_bytes']=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
    atomic_json(manifest,destination/'compression_manifest.json')
    if verify:
        reloaded=AutoModelForCausalLM.from_pretrained(destination,torch_dtype=torch.float16,attn_implementation='sdpa').to(next(source.parameters()).device)
        assert len(reloaded.model.layers)==manifest['layers']
        assert module_hash(reloaded)==manifest['weight_hash']
        tok=AutoTokenizer.from_pretrained(destination)
        assert tok.get_vocab()==tokenizer.get_vocab()
        ids=tok('A short multilingual test.',return_tensors='pt').input_ids.to(next(source.parameters()).device)
        assert torch.isfinite(reloaded(ids,logits_to_keep=1).logits).all()
        assert reloaded.generate(ids,max_new_tokens=2,do_sample=False,pad_token_id=tok.pad_token_id).shape[1]>ids.shape[1]
        del reloaded
    return manifest


@torch.no_grad()
def export_best_nf4(source,merger,tokenizer,pairs,destination):
    from .quantization import quantize_dense_nf4,load_model,verify_nf4,save_nf4_checkpoint
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    with torch.autocast('cuda',dtype=torch.float16),variant(source,merger,pairs):
        final=quantize_dense_nf4(copy.deepcopy(source))
    verify_nf4(final)
    final.config.use_cache=True
    save_nf4_checkpoint(final,destination); tokenizer.save_pretrained(destination)
    expected=module_hash(final); del final
    reloaded=load_model(str(destination),None,next(source.parameters()).device,True)
    verify_nf4(reloaded)
    if module_hash(reloaded)!=expected:raise RuntimeError('Saved NF4 weights changed on reload')
    size=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
    model_bytes=sum(p.stat().st_size for p in destination.glob('*.safetensors'))
    return reloaded,dict(serialized_bytes=size,model_tensor_file_bytes=model_bytes,tokenizer_config_bytes=size-model_bytes,
                         merged_parameters=sum(p.numel() for p in source.parameters())-sum(sum(p.numel() for p in source.model.layers[i].parameters()) for i in pairs),
                         weight_hash=expected,format='NF4 double quantization; embeddings, norms and lm_head FP16')
