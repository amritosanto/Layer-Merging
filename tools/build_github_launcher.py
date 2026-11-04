"""Generate the small Kaggle launcher; experiment code is fetched from GitHub at runtime."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []

def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))

def code(text):
    cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.splitlines(True)))

md('''# Run Layer-Merging from GitHub on Kaggle

This launcher downloads [amritosanto/Layer-Merging](https://github.com/amritosanto/Layer-Merging),
runs the Qwen3 experiment, and displays its actual tables, charts and report.
It contains no embedded experiment implementation or example scores.

**Before running:** enable **Internet** and select **GPU T4 ×2** in Kaggle settings.
Use a fresh kernel. The default runs the complete pilot pipeline; training has a
seven-hour soft budget, and ranking plus benchmarks take additional time. You can
split the stages across sessions using the controls below and restore saved outputs.

The default GitHub revision is pinned to the uploaded, tested project commit.
Changing the revision requires a new source directory and must not silently reuse
an incompatible experiment. GPU-dependent correctness is checked on Kaggle.
''')
md('## 1. Run controls\nChoose stages before Run All. For a later evaluation session use '
   '`STAGES = ["benchmark-preflight", "benchmark", "report"]` and restore the earlier outputs. '
   'For display only use `STAGES = []`, `RUN_TESTS = False`, and run the results cells. '
   'Set `OVERRIDES["evaluation_mode"] = "full"` to evaluate the same trained models in full mode.')
code('''from pathlib import Path

GITHUB_REPOSITORY = 'amritosanto/Layer-Merging'
GITHUB_COMMIT = 'f59a42245e20a126c80965445de88c5af3f21a52'
WORKING = Path('/kaggle/working')
OUTPUT = WORKING / 'Qwen3Merger'

# Optional: attach the previous Kaggle output as an input dataset and set its directory.
# It must contain manifests/run.json, checkpoints/, and the other experiment outputs.
RESTORE_FROM = None  # Example: '/kaggle/input/my-previous-run/Qwen3Merger'

STAGES = ['prepare', 'benchmark-preflight', 'train', 'rank-export', 'benchmark', 'report']
BENCHMARKS = ['MMMLU', 'INCLUDE', 'MGSM']
VARIANT = 'all'  # Or e.g. 'learned_3_nf4', 'original_fp16', 'original_nf4'.
RUN_TESTS = True

# Defaults are supplied by the GitHub project's Config class.
# Training defaults: 10 epochs, 7 hours, 12,000 Wiki-40B tokens/language.
# Override training settings only for a fresh run (or match the restored run exactly).
OVERRIDES = {
    'evaluation_mode': 'pilot',
    # 'train_tokens_per_language': 100000,
    # 'training_hours': 7.0,
    # 'eval_context': 8192,
}
''')
md('## 2. Download the pinned GitHub project\nSource files are checked against the downloaded snapshot. '
   'An existing modified checkout is left intact and causes an error rather than being overwritten.')
code('''import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', GITHUB_REPOSITORY):
    raise ValueError('Expected GitHub owner/repository')
if not re.fullmatch(r'[0-9a-f]{40}', GITHUB_COMMIT):
    raise ValueError('Use an immutable 40-character Git commit SHA')
WORKING.mkdir(parents=True, exist_ok=True)
SOURCE = WORKING / ('Layer-Merging-source-' + GITHUB_COMMIT[:12])
url = f'https://codeload.github.com/{GITHUB_REPOSITORY}/zip/{GITHUB_COMMIT}'
request = urllib.request.Request(url, headers={'User-Agent': 'Qwen3-Kaggle-Launcher'})
try:
    with urllib.request.urlopen(request, timeout=120) as response:
        snapshot = response.read()
except Exception as error:
    raise RuntimeError('GitHub download failed. Enable Kaggle Internet and check the repository/commit.') from error

with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
    members = [m for m in archive.infolist() if not m.is_dir()]
    roots = {m.filename.split('/')[0] for m in members}
    if len(roots) != 1:
        raise RuntimeError('Unexpected GitHub archive layout')
    payloads = {}
    for member in members:
        relative = Path(*member.filename.split('/')[1:])
        destination = (SOURCE / relative).resolve()
        if not destination.is_relative_to(SOURCE.resolve()):
            raise RuntimeError('Archive path escapes the source directory')
        payloads[relative.as_posix()] = archive.read(member)
    for required in ['run.py', 'config.py', 'requirements.txt', 'src/trainer.py']:
        if required not in payloads:
            raise RuntimeError(f'Missing project file: {required}')
    # Validate all existing files before writing any missing ones.
    for relative, contents in payloads.items():
        path = SOURCE / relative
        if path.exists() and path.read_bytes() != contents:
            raise RuntimeError(f'Local source was modified: {path}. Use another SOURCE directory.')
    for relative, contents in payloads.items():
        path = SOURCE / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(contents)

print(f'Downloaded {len(payloads)} files from {GITHUB_REPOSITORY}@{GITHUB_COMMIT}')
print('Source:', SOURCE)
print('Snapshot SHA256:', hashlib.sha256(snapshot).hexdigest())
''')
md('## 3. Restore outputs and install dependencies\nUse Kaggle’s notebook Python and working pip directly, '
   'with its installed PyTorch version constrained. Research stages run in fresh subprocesses, '
   'so they see installed package versions without importing them into the notebook kernel. '
   'Checkpoint loading is only for your own trusted saved outputs.')
code('''import importlib.metadata

if RESTORE_FROM:
    previous = Path(RESTORE_FROM)
    if not (previous / 'manifests/run.json').is_file():
        raise FileNotFoundError(f'No experiment run manifest in {previous}')
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError('OUTPUT is already populated. Set RESTORE_FROM=None to resume it directly.')
    shutil.copytree(previous, OUTPUT, dirs_exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
LAUNCHER = OUTPUT / 'launcher'
LAUNCHER.mkdir(parents=True, exist_ok=True)

provenance_path = LAUNCHER / 'source.json'
provenance = {
    'repository': GITHUB_REPOSITORY,
    'commit': GITHUB_COMMIT,
    'files_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
}
if provenance_path.exists():
    if json.loads(provenance_path.read_text()) != provenance:
        raise RuntimeError('Restored outputs use different source code. Choose a new OUTPUT directory.')
provenance_path.write_text(json.dumps(provenance, indent=2))

# Kaggle's startup hooks rely on packages in its notebook environment. A newly
# created or interrupted venv can lack both pip and those startup dependencies.
# Never reuse the old qwen3-runtime-* environment; use the live kernel interpreter.
PYTHON = Path(sys.executable)
torch_version = importlib.metadata.version('torch')
constraint = LAUNCHER / 'keep-kaggle-torch.txt'
constraint.write_text(f'torch=={torch_version}\\n')
ENV = os.environ.copy()
ENV.update(USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')

def run_command(arguments, log_name):
    """Stream subprocess progress into the notebook and retain a complete log."""
    log_path = LAUNCHER / log_name
    print('Running:', ' '.join(map(str, arguments)), flush=True)
    with log_path.open('a', encoding='utf-8') as log:
        process = subprocess.Popen(list(map(str, arguments)), cwd=SOURCE, env=ENV,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
        except BaseException:
            # SIGINT lets the trainer write its latest checkpoint in finally.
            import signal
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()
            raise
    if status:
        raise RuntimeError(f'Command failed with exit code {status}. See {log_path}')

run_command([PYTHON, '-m', 'pip', '--version'], 'installation.log')
run_command([PYTHON, '-m', 'pip', 'install', '--disable-pip-version-check',
             '-r', SOURCE / 'requirements.txt', '-c', constraint], 'installation.log')
run_command([PYTHON, '-c',
    "import torch, importlib.metadata; assert importlib.metadata.version('torch') == " + repr(torch_version) +
    "; print('Preserved Kaggle PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"],
    'installation.log')
''')
md('## 4. Configure the experiment and check both GPUs\nOn resume, training settings are recovered from '
   'the run manifest before applying your overrides. Normal GPU stages require two visible GPUs.')
code('''valid_stages = ['prepare', 'benchmark-preflight', 'train', 'rank-export', 'benchmark', 'report']
if len(STAGES) != len(set(STAGES)) or any(s not in valid_stages for s in STAGES):
    raise ValueError('Unknown or duplicate stage')
if any(b not in ['MMMLU', 'INCLUDE', 'MGSM'] for b in BENCHMARKS):
    raise ValueError('Unknown benchmark')

settings = {}
run_lock = OUTPUT / 'manifests/run.json'
if run_lock.exists():
    settings.update(json.loads(run_lock.read_text())['config'])
settings.update(OVERRIDES)
settings['output_dir'] = str(OUTPUT)
CONFIG = LAUNCHER / 'config.json'
CONFIG.write_text(json.dumps(settings, indent=2))
configure = (
    "import json; from config import Config; "
    "from pathlib import Path; p=Path(" + repr(str(CONFIG)) + "); "
    "c=Config(**json.loads(p.read_text())).validate(); c.save(p); print(p.read_text())"
)
run_command([PYTHON, '-c', configure], 'configuration.log')
MODE = json.loads(CONFIG.read_text())['evaluation_mode']
if any(s in STAGES for s in ['benchmark-preflight', 'train', 'rank-export', 'benchmark']):
    check = ('from config import Config; from src.trainer import hardware; '
             'hardware(Config.load(' + repr(str(CONFIG)) + '))')
    run_command([PYTHON, '-c', check], 'hardware.log')
if RUN_TESTS:
    run_command([PYTHON, '-m', 'pytest', '-q', 'tests'], 'tests.log')
''')
md('## 5. Run the selected stages\nEach stage runs in a fresh process to release GPU memory afterward. '
   'Training and benchmark predictions resume through the project’s own checkpoint logic. '
   'A failed stage stops the pipeline and leaves its logs available. Rerunning does not automatically '
   'skip a stage based solely on this launcher’s history.')
code('''history_path = LAUNCHER / 'stage_history.json'
history = json.loads(history_path.read_text()) if history_path.exists() else []

def execute(stage, *extra):
    record = {'stage': stage, 'arguments': list(extra), 'mode': MODE,
              'source_commit': GITHUB_COMMIT, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'config_sha256': hashlib.sha256(CONFIG.read_bytes()).hexdigest(), 'status': 'running'}
    history.append(record)
    history_path.write_text(json.dumps(history, indent=2))
    started = time.monotonic()
    try:
        run_command([PYTHON, 'run.py', stage, '--config', CONFIG, *extra], stage + '.log')
        record['status'] = 'complete'
    except BaseException as error:
        record['status'] = 'failed_or_interrupted'
        record['error'] = str(error)
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        temporary = history_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(history, indent=2))
        temporary.replace(history_path)

for stage in STAGES:
    if stage == 'benchmark':
        for benchmark in BENCHMARKS:
            execute(stage, '--benchmark', benchmark, '--variant', VARIANT)
    else:
        execute(stage)
print('Selected stages finished. Results directory:', OUTPUT)
''')
md('## 6. Show results\nThese tables contain recorded experiment measurements only. Missing outputs '
   'are labelled as pending. You can run this cell after an interrupted pipeline to inspect available results.')
code('''import csv
import html
from IPython.display import HTML, Markdown, Image, FileLink, display

RESULTS = OUTPUT / 'benchmark_results' / MODE
def show_csv(relative, limit=60):
    path = OUTPUT / relative
    display(Markdown('### ' + path.name))
    if not path.is_file() or path.stat().st_size < 2:
        print('Pending:', relative)
        return
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        headings = reader.fieldnames or []
        rows = list(reader)
    table = '<table><thead><tr>' + ''.join('<th>' + html.escape(h) + '</th>' for h in headings) + '</tr></thead><tbody>'
    for row in rows[:limit]:
        table += '<tr>' + ''.join('<td>' + html.escape(str(row.get(h, ''))) + '</td>' for h in headings) + '</tr>'
    display(HTML('<div style="overflow-x:auto">' + table + '</tbody></table></div>'))
    if len(rows) > limit:
        print(f'Showing {limit} of {len(rows)} rows. Download the CSV for all languages/variants.')
    display(FileLink(str(path)))

for relative in [
    'logs/validation.csv', 'rankings/selected_plans.csv',
    f'benchmark_results/{MODE}/aggregate.csv',
    f'benchmark_results/{MODE}/canonical_language_scores.csv',
    f'benchmark_results/{MODE}/macro_paired_differences.csv',
    f'benchmark_results/{MODE}/paired_differences.csv',
    f'benchmark_results/{MODE}/performance.csv',
]:
    show_csv(relative)

if not (RESULTS / 'aggregate.csv').exists():
    prediction_files = list(RESULTS.glob('*/*.jsonl'))
    print(f'{len(prediction_files)} prediction files are available; aggregate tables are pending the report stage.')
''')
md('## 7. Charts and research report\nThe project generates training curves, benchmark comparisons, '
   'language heatmaps, compression plots, selected-pair diagrams and GPU measurements from actual outputs.')
code('''figures = sorted((OUTPUT / 'charts').glob('*.png'))
if not figures:
    print('Charts pending: finish the relevant stages and run the report stage.')
for figure in figures:
    display(Markdown('### ' + figure.stem))
    display(Image(filename=str(figure)))

report = OUTPUT / f'REPORT_{MODE}.md'
if not report.exists():
    report = OUTPUT / 'REPORT.md'
if report.exists():
    # Figures are displayed above; avoid broken relative image paths in notebook Markdown.
    text = '\\n'.join(line for line in report.read_text(encoding='utf-8').splitlines()
                     if not line.startswith('!['))
    display(Markdown(text))
    display(FileLink(str(report)))
else:
    print('Research report pending. No benchmark scores have been fabricated.')
''')
md('## 8. Download results and preserve resume files\nThe analysis ZIP excludes model weights and checkpoints '
   'to avoid duplicating large artifacts. Use **Save Version** and retain the entire `Qwen3Merger` output '
   'folder to resume training or benchmarking. Restore it as a Kaggle input in a later session.')
code('''analysis_zip = OUTPUT / f'analysis_outputs_{MODE}.zip'
with zipfile.ZipFile(analysis_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
    folders = ['logs', 'manifests', 'rankings', 'charts', 'launcher', f'benchmark_results/{MODE}']
    for folder in folders:
        for path in sorted((OUTPUT / folder).rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(OUTPUT))
    for path in OUTPUT.glob('REPORT*.md'):
        archive.write(path, path.name)
display(FileLink(str(analysis_zip)))
print('Analysis archive:', analysis_zip, f'({analysis_zip.stat().st_size / 2**20:.2f} MiB)')
print('Keep these directories separately for resume and deployment:')
print(OUTPUT / 'checkpoints')
print(OUTPUT / 'model_exports')
''')

for index, cell in enumerate(cells):
    cell['id'] = f'github-launcher-{index:02d}'
notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
                metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                              language_info=dict(name='python'),
                              kaggle=dict(isInternetEnabled=True, accelerator='gpu',
                                          isGpuEnabled=True, language='python', sourceType='notebook')))
destination = ROOT / 'Qwen3Merger_GitHub_Kaggle.ipynb'
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print(destination)
