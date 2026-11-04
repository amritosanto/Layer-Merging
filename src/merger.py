"""Faithful port of the supplied PairWeightMerger; shape registry is runtime-derived."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .weight_patching import pack, unpack

class WeightAttentionBlock(nn.Module):
    def __init__(self, width, heads, multiplier):
        super().__init__(); self.heads = heads
        self.ln1=nn.LayerNorm(width); self.qkv=nn.Linear(width,3*width); self.out=nn.Linear(width,width)
        self.ln2=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,multiplier*width),nn.GELU(),nn.Linear(multiplier*width,width))

    def forward(self,x):
        b,n,d=x.shape
        q,k,v=self.qkv(self.ln1(x)).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
        x=x+self.out(y.transpose(1,2).reshape(b,n,d))
        return x+self.ff(self.ln2(x))

class PairWeightMerger(nn.Module):
    def __init__(self,cfg,families,num_layers=28):
        super().__init__(); self.families=families; d=cfg.merger_width
        patches={f.kind:f.patch for f in families}
        self.enc=nn.ModuleDict({k:nn.Linear(math.prod(p),d) for k,p in patches.items()})
        self.dec=nn.ModuleDict({k:nn.Linear(d,math.prod(p)) for k,p in patches.items()})
        self.family=nn.Embedding(len(families),d); self.source=nn.Embedding(2,d); self.layer=nn.Embedding(num_layers,d)
        self.row=nn.Embedding(max(f.shape[0]//f.patch[0] for f in families),d)
        self.col=nn.Embedding(max(1 if len(f.shape)==1 else f.shape[1]//f.patch[1] for f in families),d)
        self.blocks=nn.Sequential(*[WeightAttentionBlock(d,cfg.merger_heads,cfg.merger_ff_multiplier) for _ in range(cfg.merger_blocks)])
        self.fuse=nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,d),nn.GELU())
        for head in self.dec.values(): nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def forward(self,a,b,left_layer):
        chunks=[[],[]]; scales={}; anchors={}
        for index,f in enumerate(self.families):
            scale=((a[f.name].float().square().mean()+b[f.name].float().square().mean())/2).sqrt().clamp_min(.01)
            scales[f.name]=scale; idx=torch.arange(f.count,device=a[f.name].device)
            columns=1 if len(f.shape)==1 else f.shape[1]//f.patch[1]
            position=self.row(idx//columns)+self.col(idx%columns)+self.family.weight[index]
            for source,weights in enumerate((a,b)):
                patches=pack(weights[f.name],f.patch)
                x=self.enc[f.kind](patches/scale)
                chunks[source].append(x+position+self.source.weight[source]+self.layer.weight[left_layer+source])
                if source==1: anchors[f.name]=patches
        z=self.blocks(torch.cat([torch.cat(c) for c in chunks])[None]).squeeze(0)
        za,zb=z.chunk(2); fused=self.fuse(torch.cat((za,zb),dim=-1))
        outputs={}; offset=0
        for f in self.families:
            delta=self.dec[f.kind](fused[offset:offset+f.count]).float()*scales[f.name]
            outputs[f.name]=unpack(anchors[f.name]+delta,f.patch,f.shape); offset+=f.count
        return outputs

def correction_penalty(generated,a,b):
    terms=[]
    for name,value in generated.items():
        scale2=((a[name].float().square().mean()+b[name].float().square().mean())/2).clamp_min(.01**2)
        terms.append((value.float()-b[name].float()).square().mean()/scale2)
    return torch.stack(terms).mean()
