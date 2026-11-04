import functools
import numpy as np

def valid_plan(pairs,n=28,allow_empty=False):
    pairs=tuple(sorted(pairs))
    if (not pairs and not allow_empty) or any(p<0 or p>=n-1 for p in pairs) or any(b<=a+1 for a,b in zip(pairs,pairs[1:])):
        raise ValueError('Require nonoverlapping original adjacent pairs')
    return pairs

def task_for_batch(seed,epoch,batch_index,n=28):
    k=(1,1,2,3,4,5)[batch_index%6]
    if k==1:
        index=(batch_index//6)*2+batch_index%6
        order=np.random.default_rng(seed+epoch*100003+index//(n-1)).permutation(n-1)
        return (int(order[index%(n-1)]),)
    # Bijection: choose k sorted positions in n-k slots, then add 0..k-1.
    # This is uniform over all compatible sets, without materializing 33k plans.
    rng=np.random.default_rng(seed+epoch*1000003+batch_index)
    return tuple(int(x+i) for i,x in enumerate(sorted(rng.choice(n-k,k,replace=False))))

def checkpoint_plans(n=28):
    return [(i,) for i in range(n-1)]+[tuple(range(offset,2*k+offset,2)) for k in range(2,6) for offset in (0,1)]

def compatible_selection(costs,k):
    @functools.lru_cache(None)
    def solve(i,left):
        if left==0:return (0.,())
        if i>=len(costs):return (float('inf'),())
        skip=solve(i+1,left); value,tail=solve(i+2,left-1)
        take=(costs[i]+value,(i,)+tail)
        return min(skip,take,key=lambda v:(v[0],v[1]))
    cost,plan=solve(0,k)
    if len(plan)!=k:raise ValueError('Not enough compatible pairs')
    return plan
