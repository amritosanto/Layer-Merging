import functools
import numpy as np

def valid_plan(pairs,n=28,allow_empty=False):
    pairs=tuple(sorted(pairs))
    if (not pairs and not allow_empty) or any(p<0 or p>=n-1 for p in pairs) or any(b<=a+1 for a,b in zip(pairs,pairs[1:])):
        raise ValueError('Require nonoverlapping original adjacent pairs')
    return pairs

def task_for_batch(seed,epoch,batch_index,n=28,min_pairs=1,max_pairs=None):
    maximum=min(n//2,max_pairs or n//2)
    if not 1<=min_pairs<=maximum:raise ValueError('Invalid training pair range')
    # Every other task is a single pair; the others cycle all compression levels.
    k=min_pairs if batch_index%2==0 else min_pairs+(batch_index//2)%(maximum-min_pairs+1)
    if k==1:
        return (int(np.random.default_rng(seed+epoch).permutation(n-1)[(batch_index//2)%(n-1)]),)
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

def candidate_plans(costs,minimum=1,maximum=14,budget=12):
    """One DP winner per level; bounded, evenly spaced level coverage."""
    maximum=min(maximum,(len(costs)+1)//2)
    levels=list(range(minimum,maximum+1))
    if len(levels)>budget:
        levels=[levels[i] for i in sorted(set(np.linspace(0,len(levels)-1,budget,dtype=int)))]
    return [compatible_selection(costs,k) for k in levels]


def local_swaps(plan,n=28):
    result=set()
    for old in plan:
        for new in (old-1,old+1):
            try:result.add(valid_plan([p for p in plan if p!=old]+[new],n))
            except ValueError:pass
    return sorted(result)


def select_measured(rows,minimum_reduction=.4,preferred_max=.5):
    """CE decides within the preferred range, then among >= minimum savings."""
    finite=[r for r in rows if np.isfinite(r['ranking_ce'])]
    feasible=[r for r in finite if r['reduction']>=minimum_reduction]
    preferred=[r for r in feasible if r['reduction']<=preferred_max]
    pool=preferred or feasible or finite
    if not pool:raise ValueError('No finite candidate CE')
    return min(pool,key=lambda r:(r['ranking_ce'],len(r['pairs']),tuple(r['pairs'])))
