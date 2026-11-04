import inspect
import json
import torch
from torch.func import functional_call

EXPECTED = {'self_attn.'+x+'.weight' for x in ('q_proj','k_proj','v_proj','o_proj','q_norm','k_norm')} | {
    'mlp.'+x+'.weight' for x in ('gate_proj','up_proj','down_proj')} | {
    'input_layernorm.weight','post_attention_layernorm.weight'}

def families_from_block(block):
    params = dict(block.named_parameters())
    if set(params) != EXPECTED:
        raise ValueError(f'Unexpected Qwen3 parameters: missing={EXPECTED-set(params)}, extra={set(params)-EXPECTED}')
    return params

def block_parameters(families): return dict(families)

def _config_json_default(value):
    # Transformers converts torch_dtype, but leaves _pre_quantization_dtype
    # as a torch.dtype after NF4 loading and dequantized_copy.
    if isinstance(value, torch.dtype):
        return str(value).removeprefix('torch.')
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

def inspect_model(model, strict=True):
    c = model.config
    if c.model_type != 'qwen3': raise ValueError('Expected dense Qwen3')
    expected = dict(num_hidden_layers=28, hidden_size=1024, intermediate_size=3072,
                    num_attention_heads=16, num_key_value_heads=8, head_dim=128)
    if strict:
        for key, value in expected.items():
            if getattr(c,key) != value: raise ValueError(f'Unexpected {key}: {getattr(c,key)}')
    shapes = {k: list(v.shape) for k,v in families_from_block(model.model.layers[0]).items()}
    for layer in model.model.layers:
        assert {k:list(v.shape) for k,v in families_from_block(layer).items()} == shapes
    signature = str(inspect.signature(model.model.layers[0].forward))
    if 'position_embeddings' not in signature: raise RuntimeError('Unsupported Qwen3 decoder API')
    config = json.loads(json.dumps(c.to_dict(), default=_config_json_default, allow_nan=False))
    return dict(config=config, shapes=shapes, decoder_signature=signature)

def layer_context(model, h):
    length = h.shape[1]
    positions = torch.arange(length, device=h.device)
    mask = torch.full((length,length), torch.finfo(h.dtype).min, device=h.device, dtype=h.dtype).triu(1)
    return dict(attention_mask=mask[None,None], position_ids=positions[None], cache_position=positions,
                position_embeddings=model.model.rotary_emb(h, positions[None]), use_cache=False)

def run_layer(block, h, context, params=None):
    if params is None:
        out = block(h, **context)
    else:
        if set(params) != set(dict(block.named_parameters())): raise ValueError('Incomplete generated layer')
        # Cast is differentiable. FP16 normalization weights prevent unintended FP32 residual promotion.
        out = functional_call(block, {k:v.to(h.dtype) for k,v in params.items()}, (h,), context, strict=False)
    return out[0] if isinstance(out, tuple) else out

@torch.no_grad()
def teacher_states(model, ids):
    h = model.model.embed_tokens(ids); context = layer_context(model,h); states = [h]
    for block in model.model.layers:
        h = run_layer(block,h,context); states.append(h)
    return states
