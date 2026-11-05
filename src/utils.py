import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def atomic_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sanitize_diagnostics(value):
    """Sanitize diagnostics only; scientific manifests still use strict atomic_json."""
    import math
    import numbers
    bad=[]
    def visit(item,path):
        if isinstance(item,dict):return {str(k):visit(v,path+'.'+str(k) if path else str(k)) for k,v in item.items()}
        if isinstance(item,(list,tuple)):return [visit(v,f'{path}[{i}]') for i,v in enumerate(item)]
        if isinstance(item,torch.Tensor) and item.numel()==1:item=item.detach().cpu().item()
        if isinstance(item,numbers.Integral) and not isinstance(item,bool):return int(item)
        if isinstance(item,numbers.Real) and not isinstance(item,(bool,int)):
            if not math.isfinite(float(item)):bad.append(path);return None
            return float(item)
        return item
    return visit(value,''),bad


def diagnostic_json(value,path):
    clean,bad=sanitize_diagnostics(value)
    atomic_json(dict(clean,nonfinite_metrics=bad),path)
    return bad

def atomic_torch(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp); tmp.replace(path)

def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def module_hash(model):
    return digest({k: tensor_hash(v) for k, v in model.state_dict().items()})

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])

def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def file_digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
