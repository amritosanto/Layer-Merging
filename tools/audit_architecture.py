"""Inspect authoritative classes without executing the reference experiment."""
import ast
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config,Qwen3ForCausalLM
from config import Config
from src.qwen3_adapter import families_from_block
from src.weight_patching import make_families
from src.merger import PairWeightMerger
from src.utils import atomic_json

tree=ast.parse(Path('reference/gpt2_reference.py').read_text(encoding='utf-8-sig'))
nodes=[n for n in tree.body if (isinstance(n,ast.ClassDef) and n.name in {'WeightAttentionBlock','PairWeightMerger'}) or
       (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'FAMILIES','PATCH'} for t in n.targets))]
ns=dict(torch=torch,nn=nn,F=F,math=math)
exec(compile(ast.Module(nodes,type_ignores=[]),'<reference>','exec'),ns)
cfg=Config()
with torch.device('meta'):
    original=ns['PairWeightMerger'](cfg)
    c=Qwen3Config(vocab_size=151936,hidden_size=1024,intermediate_size=3072,num_hidden_layers=28,
                  num_attention_heads=16,num_key_value_heads=8,head_dim=128,tie_word_embeddings=True)
    c._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(c)
    families=make_families(families_from_block(model.model.layers[0]))
    merger=PairWeightMerger(cfg,families)
counts=dict(gpt2_merger_parameters=sum(p.numel() for p in original.parameters()),
            qwen3_merger_parameters=sum(p.numel() for p in merger.parameters()),
            qwen3_weight_tokens=2*sum(f.count for f in families),
            families=[dict(name=f.name,shape=f.shape,patch=f.patch,tokens_per_source=f.count) for f in families])
atomic_json(counts,'reference/architecture_counts.json'); print(counts)
