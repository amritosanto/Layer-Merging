from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Family:
    name: str
    kind: str
    shape: tuple
    patch: tuple

    @property
    def count(self): return math.prod(self.shape) // math.prod(self.patch)

def pack(x, patch):
    if len(patch) == 1:
        if x.ndim != 1 or x.shape[0] % patch[0]: raise ValueError('Invalid vector patch')
        return x.reshape(-1, patch[0])
    ph, pw = patch; h, w = x.shape
    if h % ph or w % pw: raise ValueError(f'Patch {patch} does not divide {tuple(x.shape)}')
    return x.reshape(h//ph, ph, w//pw, pw).permute(0,2,1,3).reshape(-1, ph*pw)

def unpack(x, patch, shape):
    if len(patch) == 1: return x.reshape(shape)
    ph, pw = patch; h, w = shape
    return x.reshape(h//ph,w//pw,ph,pw).permute(0,2,1,3).reshape(shape)

def make_families(parameters, matrix_patch=128, vector_patch=128):
    families = []
    for name, p in parameters.items():
        if p.ndim not in (1,2): raise ValueError(f'Unsupported parameter: {name} {p.shape}')
        # gcd only adjusts for divisibility (also supports tiny correctness models).
        patch = tuple(math.gcd(int(n), vector_patch if p.ndim == 1 else matrix_patch) for n in p.shape)
        kind = ('vector_' if p.ndim == 1 else 'matrix_') + '_'.join(map(str,patch))
        families.append(Family(name, kind, tuple(p.shape), patch))
    return tuple(families)
