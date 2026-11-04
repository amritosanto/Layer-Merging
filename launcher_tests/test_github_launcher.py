import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

def code_cells():
    notebook = json.loads((ROOT / 'Qwen3Merger_GitHub_Kaggle.ipynb').read_text(encoding='utf-8'))
    return [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']

def test_launcher_cells_compile():
    cells = code_cells()
    assert len(cells) == 8
    for index, source in enumerate(cells):
        compile(source, f'launcher_cell_{index}', 'exec')
    assert 'f59a42245e20a126c80965445de88c5af3f21a52' in cells[0]
    assert 'base64.b64decode' not in ''.join(cells)

def test_snapshot_download_and_modified_source_protection(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for name in ['run.py', 'config.py', 'requirements.txt', 'src/trainer.py']:
            archive.writestr('Layer-Merging-test/' + name, '# fixture\n')
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *args, **kwargs: io.BytesIO(buffer.getvalue()))
    scope = dict(GITHUB_REPOSITORY='amritosanto/Layer-Merging', GITHUB_COMMIT='a' * 40,
                 WORKING=tmp_path, Path=Path)
    exec(code_cells()[1], scope)
    assert (scope['SOURCE'] / 'src/trainer.py').read_text() == '# fixture\n'
    exec(code_cells()[1], scope)  # Exact snapshot is reusable.
    (scope['SOURCE'] / 'run.py').write_text('# local modification\n')
    with pytest.raises(RuntimeError, match='modified'):
        exec(code_cells()[1], scope)
    assert (scope['SOURCE'] / 'run.py').read_text() == '# local modification\n'

def pipeline_scope(tmp_path, command):
    config = tmp_path / 'config.json'
    config.write_text('{}')
    import hashlib
    return dict(LAUNCHER=tmp_path, CONFIG=config, MODE='pilot', GITHUB_COMMIT='a' * 40,
                PYTHON='python', OUTPUT=tmp_path, time=time, json=json, hashlib=hashlib,
                STAGES=['prepare', 'benchmark', 'report'], BENCHMARKS=['MMMLU', 'MGSM'],
                VARIANT='all', run_command=command)

def test_pipeline_order_and_history(tmp_path):
    calls = []
    scope = pipeline_scope(tmp_path, lambda args, log: calls.append(args))
    exec(code_cells()[4], scope)
    assert [args[2] for args in calls] == ['prepare', 'benchmark', 'benchmark', 'report']
    assert calls[1][-4:] == ['--benchmark', 'MMMLU', '--variant', 'all']
    history = json.loads((tmp_path / 'stage_history.json').read_text())
    assert len(history) == 4 and all(row['status'] == 'complete' for row in history)

def test_failed_stage_stops_and_records_failure(tmp_path):
    calls = []
    def fail(args, log):
        calls.append(args)
        raise RuntimeError('deliberate test failure')
    with pytest.raises(RuntimeError, match='deliberate'):
        exec(code_cells()[4], pipeline_scope(tmp_path, fail))
    assert len(calls) == 1
    history = json.loads((tmp_path / 'stage_history.json').read_text())
    assert history[0]['status'] == 'failed_or_interrupted'

def test_installation_uses_kernel_python_even_with_broken_venv(tmp_path, monkeypatch):
    import hashlib
    import importlib.metadata
    import os
    import shutil
    import subprocess
    import sys
    broken = tmp_path / ('qwen3-runtime-' + 'a' * 12) / 'bin'
    broken.mkdir(parents=True)
    (broken / 'python').write_text('incomplete environment without pip')
    calls = []
    class Process:
        stdout = ['mock subprocess output\n']
        def __init__(self, args, **kwargs):
            calls.append(args)
        def wait(self, **kwargs):
            return 0
    monkeypatch.setattr(subprocess, 'Popen', Process)
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '2.6.0+cu124')
    scope = dict(Path=Path, RESTORE_FROM=None, OUTPUT=tmp_path/'outputs', WORKING=tmp_path,
                 SOURCE=tmp_path, GITHUB_REPOSITORY='amritosanto/Layer-Merging', GITHUB_COMMIT='a'*40,
                 payloads={'run.py': b'# fixture'}, hashlib=hashlib, json=json, os=os,
                 shutil=shutil, subprocess=subprocess, sys=sys)
    exec(code_cells()[2], scope)
    assert len(calls) == 3
    assert all(Path(args[0]) == Path(sys.executable) for args in calls)
    assert calls[0][1:] == ['-m', 'pip', '--version']
    assert calls[1][1:4] == ['-m', 'pip', 'install']
    assert '-c' in calls[1]  # Keep the existing CUDA PyTorch version constrained.
    assert (scope['LAUNCHER']/'keep-kaggle-torch.txt').read_text() == 'torch==2.6.0+cu124\n'
