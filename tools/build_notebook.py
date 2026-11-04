"""Build a self-contained notebook and portable source archive from reviewed source files."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
files=[ROOT/name for name in ('config.py','run.py','requirements.txt','README.md','ARCHITECTURE_AUDIT.md','VALIDATION.md')]
files.extend(p for p in (ROOT/'validation').glob('*') if p.suffix in ('.json','.xml'))
if (ROOT/'outputs/REPORT.md').exists():files.append(ROOT/'outputs/REPORT.md')
for folder in ('src','benchmarks','tests','tools','reference'):
    files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.md'))
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(files):bundle.writestr(path.relative_to(ROOT).as_posix(),path.read_bytes())
payload=buffer.getvalue(); encoded=base64.b64encode(payload).decode(); checksum=hashlib.sha256(payload).hexdigest()
cells=[]
def md(text):cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(True)))
def code(text):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.splitlines(True)))
md('# Qwen3 multilingual learned transformer layer merger\n\nSelf-contained Kaggle experiment. Select T4 ×2 and enable Internet. '
   'Training and benchmark results are pending until executed. Source modules and tests are embedded below. '
   'The supplied GPT-2 architecture is preserved; see the extracted ARCHITECTURE_AUDIT.md.')
md('## 1. Environment setup\nRun in a fresh kernel. This cell extracts the complete implementation and installs pinned dependencies without replacing PyTorch.')
code(f'''import base64, hashlib, io, os, subprocess, sys, zipfile
from pathlib import Path
SOURCE = Path('/kaggle/working/qwen3_merger_source')
SOURCE.mkdir(parents=True, exist_ok=True)
payload = base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest() == {checksum!r}
with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
    for name in bundle.namelist():
        target = (SOURCE / name).resolve()
        assert target.is_relative_to(SOURCE.resolve()), name
    bundle.extractall(SOURCE)
os.chdir(SOURCE)
sys.path.insert(0, str(SOURCE))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
''')
md('## 2. Configuration\nKeep the output directory unchanged when resuming. Change token budgets here before a new experiment. Set evaluation_mode="full" later to evaluate the same trained models on full eligible test sets.')
code('''from config import Config
cfg = Config()
cfg.output_dir = '/kaggle/working/Qwen3Merger'
cfg.evaluation_mode = 'pilot'
cfg.save('config.json')
OUT = Path(cfg.output_dir)
OUT.mkdir(parents=True, exist_ok=True)
def stage(name, *extra):
    subprocess.run([sys.executable, 'run.py', name, '--config', 'config.json', *extra], check=True)
print(cfg)
''')
md('## 3. GPU detection\nNormal mode requires two CUDA GPUs. A single-GPU debug run must be explicitly configured.')
code('from src.trainer import hardware\nhardware(cfg)\n')
md('## 4. Qwen3 loading\nResolve the model and datasets to immutable revisions. The runtime reads the installed Qwen3 decoder signature and validates all parameter shapes.')
code('from src.trainer import prepare_run\nlock = prepare_run(cfg)\nprint(lock)\n')
md('## 5. NF4 quantization\nThe training stage loads actual NF4 tensors before constructing a dense frozen source copy. It separately loads the original FP16 teacher. Quantization presence, hashes and measured storage reduction are mandatory preflight checks.')
code('from src.quantization import nf4_config\nprint(nf4_config())\n')
md('## 6. Weight extraction and patching\n128 × 128 matrix patches and 128-element vectors; 978 tokens per source, 1,956 per pair for the expected configuration. All 11 decoder parameter families are covered.')
code('import json\nprint(json.dumps(json.loads(Path("reference/architecture_counts.json").read_text()), indent=2))\n')
md('## 7. Merger architecture\nSix bidirectional attention blocks, shared shape encoders/decoders, matched-patch fusion, and second-block anchoring. Initial corrections are exactly zero. The merger has 23,676,288 parameters versus 39,391,168 in the reference; this capacity change is documented.')
code('from IPython.display import Markdown, display\ndisplay(Markdown(Path("ARCHITECTURE_AUDIT.md").read_text()))\n')
md('## 8. Wiki-40B preparation\nDiscover every available language and prepare balanced token budgets from disjoint TRAIN article partitions. Also freeze benchmark prompts and demonstration IDs before model comparisons.')
code("stage('prepare')\n")
md('## 9. Training setup and preflight tests\nRun the source-level tests and actual benchmark smoke tests first. The training stage then performs all model/NF4/gradient/dual-GPU/export checks before its first optimizer update; a critical failure stops the run.')
code("subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests'], check=True)\nstage('benchmark-preflight')\n")
md('## 10. Dual-GPU training\nBoth GPU replicas contribute gradients. The five-merge batch tuner measures memory with Adam state allocated. Resume uses latest.pt automatically. Save the output directory between Kaggle sessions.')
code("stage('train')\n")
md('## 11. Validation and checkpoint selection\nValidation occurs inside training and uses equal compression-level CE gaps against matching drop-first. Benchmark test answers never select checkpoints.')
code("import pandas as pd\ndisplay(pd.read_csv(OUT / 'logs/validation.csv'))\nprint((OUT / 'manifests/training_complete.json').read_text())\n")
md('## 12. Layer-pair ranking\nRank all 27 original adjacent pairs on the independent holdout. Select raw and guarded compatible plans separately, then measure each whole plan. This stage also exports and reload-checks models; it can run in a later session.')
code("stage('rank-export')\ndisplay(pd.read_csv(OUT / 'rankings/single_pairs.csv'))\n")
md('## 13. Model compression and export\nExports contain ordinary Qwen3 models and the original tokenizer. Control definitions are saved separately and reconstructed during evaluation to save disk space. Evaluation creates separately labelled serialized NF4 deployment models.')
code("exports = json.loads((OUT / 'manifests/exports.json').read_text())\ndisplay(pd.DataFrame(exports)[['name', 'layers', 'parameters', 'stored_bytes', 'representation']])\n")
for number,benchmark in [(14,'MMMLU'),(15,'INCLUDE'),(16,'MGSM')]:
    md(f'## {number}. {benchmark} evaluation\nAll variants use the frozen few-shot prompts and selected IDs. Predictions are incrementally saved; rerunning resumes missing examples. Pilot scores are sampled experimental scores, not full official results.')
    code(f"stage('benchmark', '--benchmark', '{benchmark}')\n")
md('## 17. Result comparison\nWrite per-language and aggregate tables, paired bootstrap intervals, coverage, subgroup scores and measured efficiency. Incomplete comparisons fail rather than silently changing the paired sample.')
code("stage('report')\nRESULTS = OUT / 'benchmark_results' / cfg.evaluation_mode\ndisplay(pd.read_csv(RESULTS / 'aggregate.csv'))\ndisplay(pd.read_csv(RESULTS / 'paired_differences.csv'))\n")
md('## 18. Charts\nFigures use recorded measurements only. PNG (300 DPI), PDF and underlying CSV files are retained.')
code("from IPython.display import Image\nfor path in sorted((OUT / 'charts').glob('*.png')):\n    display(Image(filename=str(path)))\n")
md('## 19. Final report and downloads\nSave the entire output folder with Kaggle Save Version. The small archive below contains analysis artifacts; large model exports and checkpoints remain in the output folder for separate download.')
code("""from IPython.display import FileLink
display(Markdown((OUT / 'REPORT.md').read_text()))
analysis_zip = OUT / 'analysis_outputs.zip'
with zipfile.ZipFile(analysis_zip, 'w', zipfile.ZIP_DEFLATED) as z:
    for folder in ['logs', 'manifests', 'rankings', 'benchmark_results', 'charts']:
        for path in (OUT / folder).rglob('*'):
            if path.is_file(): z.write(path, path.relative_to(OUT))
    z.write(OUT / 'REPORT.md', 'REPORT.md')
display(FileLink(str(analysis_zip)))
print('Save checkpoints/ and model_exports/ separately through Kaggle outputs.')
""")
notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                                      language_info=dict(name='python',version='3.11')),nbformat=4,nbformat_minor=5)
for index,cell in enumerate(cells):cell['id']=f'qwen3-{index:03d}'
(ROOT/'Qwen3Merger.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
with zipfile.ZipFile(ROOT/'Qwen3Merger-source.zip','w',zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(files):bundle.write(path,path.relative_to(ROOT))
    bundle.write(ROOT/'Qwen3Merger.ipynb','Qwen3Merger.ipynb')
print('Built',len(cells),'cells;',len(files),'embedded files; payload SHA256',checksum)
