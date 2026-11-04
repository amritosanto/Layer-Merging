"""Numerically compare nonzero corrections with the authoritative reference math."""
import ast
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from config import Config
from src.merger import PairWeightMerger
from src.weight_patching import Family

def test_nonzero_reference_parity():
    tree=ast.parse(Path('reference/gpt2_reference.py').read_text(encoding='utf-8-sig'))
    names={'WeightAttentionBlock','PairWeightMerger','pack','unpack'}
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    families=[('matrix','square',(8,8)),('vector','vector',(8,))]
    patches={'square':(4,4),'vector':(4,)}
    ns=dict(torch=torch,nn=nn,F=F,math=math,FAMILIES=families,PATCH=patches,block_parameters=lambda x:x)
    exec(compile(ast.Module(nodes,type_ignores=[]),'<authoritative reference>','exec'),ns)
    cfg=Config(merger_width=12,merger_heads=3)
    original=ns['PairWeightMerger'](cfg)
    port=PairWeightMerger(cfg,[Family(name,kind,shape,patches[kind]) for name,kind,shape in families],12)
    torch.manual_seed(123)
    for p in original.dec.parameters():nn.init.normal_(p,mean=0,std=.01)
    state=original.state_dict()
    state['row.weight']=state['row.weight'][:port.row.num_embeddings]
    state['col.weight']=state['col.weight'][:port.col.num_embeddings]
    port.load_state_dict(state)
    a={name:torch.randn(shape) for name,_,shape in families}; b={name:torch.randn(shape) for name,_,shape in families}
    old=original(a,b,3); new=port(a,b,3)
    for key in old:torch.testing.assert_close(new[key],old[key],rtol=0,atol=0)
