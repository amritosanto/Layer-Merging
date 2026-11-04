import itertools
import numpy as np
from src.pair_selection import task_for_batch,valid_plan,compatible_selection
from src.wiki40b_loader import clean_article,document_partition,balanced_epoch

def test_plan_selection_against_exhaustive():
    costs=[3.,-2.,1.,5.,-1.,-5.,4.]
    for k in (1,2,3):
        possible=[p for p in itertools.combinations(range(len(costs)),k) if all(b>a+1 for a,b in zip(p,p[1:]))]
        expected=min(possible,key=lambda p:(sum(costs[i] for i in p),p))
        assert compatible_selection(costs,k)==expected

def test_joint_task_schedule():
    for i in range(300):
        plan=task_for_batch(42,0,i); assert valid_plan(plan)==plan
        assert len(plan)==(1,1,2,3,4,5)[i%6]
    singles=[task_for_batch(42,0,i)[0] for i in range(81) if i%6 in (0,1)]
    assert set(singles)==set(range(27))

def test_unicode_and_balancing():
    assert clean_article('_START_ARTICLE_\nবাংলা_NEWLINE_日本語')=='বাংলা\n日本語'
    assert document_partition('Q17',42)==document_partition('Q17',42)
    data={l:dict(train=list(range(16))) for l in ('en','bn','ja')}
    a=balanced_epoch(data,4,42,0); b=balanced_epoch(data,4,42,0)
    assert [l for l,_ in a]==[l for l,_ in b]
    assert len(a)==12
    for i in range(0,12,3):assert {l for l,_ in a[i:i+3]}==set(data)
