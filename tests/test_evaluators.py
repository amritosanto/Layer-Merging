import pytest
import torch
from benchmarks.mgsm import normalized_number,extract_answer,prompt
from benchmarks.mmmlu import adapt
from benchmarks.include import adapt as include_adapt
from src.benchmark_evaluation import stratified_sample,mc_prompt,score_choices,read_predictions
from src.reporting import bootstrap_difference

@pytest.mark.parametrize('text,expected',[('১,২৩৪','1234'),('١٢٫٥','25/2'),('-2/4','-1/2'),('1.234,50','2469/2'),('12,5','25/2')])
def test_numeric(text,expected):assert normalized_number(text)==expected

def test_answer_extraction():
    assert extract_answer('First 4 + 5 = 9\n#### 9')=='9'
    assert extract_answer('No answer') is None
    assert extract_answer('4 or 5') is None

def test_support_and_schema():
    rows=[dict(id=str(i),question=f'Question {i}',choices=['yes','no'],label=i%2,subject=str(i%3)) for i in range(20)]
    support=stratified_sample(rows,5,42); held=[r for r in rows if r not in support]
    p=mc_prompt(support,held[0]); assert p.endswith('Answer:')
    assert all(r['id']!=held[0]['id'] for r in support)
    assert len(stratified_sample(held,100,42))==15
    assert adapt(dict(Question='x',A='a',B='b',C='c',D='d',Answer='B'),0)['label']==1
    assert include_adapt(dict(question='x',choices=['a','b'],answer=1),0)['label']==1
    with pytest.raises(ValueError):include_adapt(dict(question='x',choices=['a','b'],answer=5),0)
    assert 'SECRET' not in prompt([dict(question='demo',answer='reason',answer_number=2)],'target','en')

def test_paired_bootstrap():
    assert bootstrap_difference([1,0],[1,0],replicates=100)==(0.,0.)
    assert bootstrap_difference([1]*4,[0]*4,replicates=100)==(100.,100.)

def test_partial_resume_line(tmp_path):
    p=tmp_path/'p.jsonl'; p.write_text('{"id": "1"}\n{"id":')
    assert read_predictions(p)==[{'id':'1'}]

def test_candidate_scores_match_full_logits(tiny):
    model,_,_,_,_=tiny
    class Tok:
        pad_token_id=0
        def encode(self,text,add_special_tokens=False):return [3,4] if text==' A' else [5]
    e=dict(input_ids=[7,8,9],choices=['one','two'])
    result=score_choices(model,Tok(),[e],'cpu',2)[0]
    expected=[]
    for tokens in ([3,4],[5]):
        logits=model(torch.tensor([[7,8,9]+tokens])).logits[0,2:2+len(tokens)].log_softmax(-1)
        expected.append(float(logits.gather(1,torch.tensor(tokens)[:,None]).sum()))
    torch.testing.assert_close(torch.tensor(result['scores']),torch.tensor(expected))
