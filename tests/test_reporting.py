import json
from config import Config
from src.reporting import make_report

def test_empty_report_is_explicit(tmp_path):
    path=make_report(Config(output_dir=str(tmp_path)))
    assert 'Results are pending' in path.read_text()
    assert '17. Conclusions' in path.read_text()

def test_paired_report_and_figures(tmp_path):
    cfg=Config(output_dir=str(tmp_path),bootstrap_samples=20)
    for variant in ('original_fp16','original_nf4','learned_1_fp16','learned_1_nf4'):
        root=tmp_path/'benchmark_results/pilot'/variant; root.mkdir(parents=True)
        for language in ('en','ja'):
            rows=[dict(variant=variant,benchmark='MGSM',language=language,id=str(i),prompt_hash=f'h{i}',
                       correct=i%2==0,invalid=False,skip_reason=None,subject='math',domain='math',regional='unknown') for i in range(4)]
            (root/f'MGSM_{language}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    make_report(cfg)
    import pandas as pd
    diffs=pd.read_csv(tmp_path/'benchmark_results/pilot/paired_differences.csv')
    assert (diffs.delta_pp==0).all() and (diffs.ci_low_pp==0).all()
    assert (tmp_path/'charts/C_heatmap_MGSM.png').exists()
    assert (tmp_path/'charts/D_languages_MGSM.pdf').exists()
