# Five-epoch multilingual Qwen3 NF4 experiment

Upload `Qwen3_5Epoch_Multilingual_NF4_Kaggle_Launcher.ipynb` to Kaggle, enable Internet, select **T4 x2**, and run its cells in order. It clones this repository, preserves Kaggle CUDA PyTorch, and installs binary wheels compatible with Python 3.13. Use the notebook dependency cell instead of the historical requirements.txt.

```bash
python run.py smoke --config config.smoke.json
# Separately resumable stages: prepare, preflight, train, search, export, benchmark, report
python run.py train --config config.smoke.json
```

Defaults: **five complete epochs**, no optimizer-step or time cap, 128-token windows, and per-language budgets of 512 training / 256 validation / 256 ranking tokens. English, Hindi and Japanese stream from Wiki-40B; Bengali, Swahili, Yoruba and Telugu stream from FineWeb2. Sources use pinned revisions and verified configurations. `manifests/language_coverage.json` records exact/proxy/unavailable coverage for the official Qwen3 119 language/dialect entries plus requested Yoruba, and distinguishes availability from actual sampling. Unavailable requested languages fail explicitly unless a replacement is configured. No whole-corpus download is required.

The six-block, width-384, six-head merger, training objective and dual-GPU training remain unchanged. The teacher stays frozen FP16 during training. Search and benchmarks compare **original_nf4** against **best_learned_nf4**, using the same NF4 double quantization and FP16 compute policy.

Search scans adjacent pairs, then evaluates at most 12 whole-plan surrogate candidates with up to two local refinements. At most three finalists are requantized and scored in memory; only the selected learned NF4 checkpoint is exported. Lowest measured NF4 ranking CE is primary among storage-feasible finalists; near ties prefer more merged layers. At least 40% saving against the published original 16-bit checkpoint is targeted, and savings of 50% or more are welcome. Actual savings against both original FP16-width storage and original NF4 storage are reported separately. This is a budget-limited search, not a global optimum; CE does not guarantee benchmark accuracy.

Pilot benchmarks use identical frozen prompts for both models: ten questions per language, three languages per benchmark, five-shot MMMLU/INCLUDE and eight-shot MGSM. MGSM has a separate 256-new-token budget and uses the first completed final answer before any unrelated next question; invalid and truncated outputs are counted separately. Pilot results cannot establish a five-percentage-point threshold statistically.

Outputs use the fresh directory `/kaggle/working/outputs/smoke_5epoch_multilingual_nf4`. Resume the same stages with unchanged code/configuration and the complete output directory, including manifests, token cache and the original NF4 reference. Interrupted training saves its latest state; only five completed epochs produce the completion marker and final checkpoint. A capped run fails explicitly rather than being labeled complete. Changed scientific settings require a fresh output directory. Search, predictions and exports are reused only when their identities match.

Artifacts include `logs/training_metrics.json`, `manifests/language_coverage.json`, `rankings/{candidate_plans.csv,nf4_finalists.csv,selected_plan.json}`, `reference_models/original_nf4`, `model_exports/best_learned_nf4`, paired pilot predictions, accuracy CSV files and `REPORT.md`. The report records completed epochs, tokens, CE, serialized sizes, compression, correct/invalid/truncated counts and limitations. Do not commit datasets, model weights or output caches.

Local validation uses synthetic data and small models. Full pretrained NF4 serialization, dual-T4 execution and benchmark accuracy remain unverified until the notebook runs on Kaggle. No model weights or corpus shards were downloaded for local checks.

---
## Historical implementation notes (superseded defaults)

# Qwen3 multilingual learned layer merger

An executable port of the supplied GPT-2 second-block-anchored merger to
`Qwen/Qwen3-0.6B-Base`. Read [the architecture audit](ARCHITECTURE_AUDIT.md) for the
preserved mathematics, explicit changes and parameter counts.

**Execution status:** implementation and local synthetic-model tests are provided.
Pretrained training, NF4 GPU preflight, dual-T4 validation and benchmark results
remain pending until run on Kaggle. No research scores or GPU performance numbers
are fabricated. `outputs/REPORT.md` is an empty reporting template.

## Run on Kaggle

1. Upload `Qwen3Merger.ipynb`, enable **Internet**, and select **GPU T4 ×2**.
2. Run the environment/bootstrap and configuration cells. The notebook embeds
   every required source/test file; it needs no external repository checkout.
   Existing Kaggle PyTorch is retained. Start a fresh kernel before installation.
3. Run data preparation and benchmark preflight, then training. All GPU
   correctness tests run before optimizer updates. The full training budget is
   ten epochs with a seven-hour soft limit; validation and safe checkpoint writes
   can exceed that limit. Training stops truthfully at the budget reached.
4. Run ranking/export in a separate session if needed. It loads the same best
   Wiki-40B-selected checkpoint for all levels.
5. Run the three benchmark sections. They use both GPUs across languages. Full
   evaluation can take much longer than training; resume it in later sessions.
6. Generate tables, figures and the report. Save the complete output folder before
   the Kaggle session expires. Exports and deployment NF4 files require substantial
   disk space; check available disk before the export stage. Only the five primary
   dense learned models and NF4 deployment models are retained. Controls are
   reconstructed from frozen source/pair weights and temporarily serialized for
   size measurement, avoiding roughly twenty redundant full-model exports.

Source is extracted to `/kaggle/working/qwen3_merger_source`; generated experiment
outputs go to **`/kaggle/working/Qwen3Merger/`**.

## Command-line stages

With the sources in the current directory, install `requirements.txt` without
reinstalling CUDA-enabled PyTorch. Generate a config using
`python -c "from config import Config; Config().save('config.json')"`.

```bash
python run.py prepare --config config.json
python run.py benchmark-preflight --config config.json
python run.py preflight --config config.json
python run.py train --config config.json
python run.py rank-export --config config.json
python run.py benchmark --config config.json --benchmark MMMLU
python run.py benchmark --config config.json --benchmark INCLUDE
python run.py benchmark --config config.json --benchmark MGSM
python run.py report --config config.json
```

`--variant learned_3_fp16`, `--variant learned_3_nf4`, `original_fp16`,
`original_nf4`, or a control name limits benchmark work to one variant. Check
`manifests/exports.json` and `manifests/controls.json` for variant definitions. All five raw learned levels
are evaluated in FP16 and NF4. Drop-first, drop-second, averaging and guarded
controls are evaluated in the same dense representation as raw C.

Set `evaluation_mode='full'` and rerun preparation/benchmark preflight/evaluation
to use all eligible test questions. Full and pilot prompts/predictions are stored
separately and reuse the training checkpoint. Other training settings must remain
unchanged. By default, pilot evaluates up to 100 questions per language.

## Data and evaluation protocol

- Discover Wiki-40B configs from the pinned dataset revision; default budgets are
  12,000 / 1,024 / 1,024 Qwen tokens per language for train/validation/ranking.
  Set the training budget to 100,000 for later experiments. No corpus substitution.
  If the HF distribution is unavailable, install `tensorflow-datasets==4.9.8`
  alongside Kaggle's TensorFlow and set `wiki_backend='tfds'` in a fresh run. This
  explicitly uses official Wiki-40B 1.3.0 and disables TensorFlow GPU allocation.
  TFDS may download complete language shards, so plan disk and preparation time.
- Only official TRAIN articles supply optimizer, validation and ranking data.
  A seeded global Wikidata-ID hash defines disjoint partitions across languages;
  exact content duplicates cannot cross partitions. Unicode is retained, article
  markers removed and documents separated by the tokenizer EOS token. Article
  offsets and actual token counts are recorded. The final training window tail is
  reported; no short corpus is duplicated to meet the requested budget.
- The scheduler interleaves shuffled per-language windows without replacement.
  Equal token budgets make training approximately balanced; scarce languages can
  contribute fewer windows and are reported as such.
- MMMLU: 14 translated configs, five fixed disjoint demonstrations. INCLUDE:
  all discovered language configs, five demonstrations from validation when
  present, otherwise a reserved test support subset. Subject-stratified selection
  is deterministic. These support protocols are **not official benchmark scores**.
  The checked INCLUDE release exposes 45 configs for 44 languages (Dutch and
  Dutch-Flemish are separate configs). Both are evaluated and shown individually;
  primary language macro-averages pool them under `nl` before averaging languages.
- MGSM: all 11 languages, eight official translated training demonstrations, same
  official test row indices across languages (answer sequences checked), greedy
  generation up to 512 new tokens. Demonstrations retain their rationale and add
  a `#### number` final-answer delimiter. The parser prefers this delimiter, then
  a single number on the final nonempty line; ambiguity is invalid, never guessed.
- MC scoring sums conditional log probabilities of explicit ` A`, ` B`, … label
  token sequences appended to the cached prompt tokens. It supports multi-token
  labels and batches candidates. MGSM generation batches examples with left padding.
- Prompt tokens, IDs, demonstrations and protocol hashes are shared across models.
  An over-context prompt is recorded as skipped for **all** variants; neither the
  question nor options nor mandatory demonstrations are silently truncated.
  Full mode attempts all eligible examples and reports any context exclusions.
- Predictions are flushed per example. Results include accuracy, macro and
  weighted aggregates, invalid rates, subject/domain/regional breakdowns, training
  coverage, retention ratios and paired bootstrap intervals for per-language
  accuracy differences. Benchmark answers never select checkpoints or pairs.

## Resume and outputs

Restore the entire output folder to the same Kaggle working path before running
the same stage again. `latest.pt` contains merger, optimizer, AMP scaler, all GPU
RNG states, Python/NumPy/CPU RNG, epoch/window cursor, selected microbatch and data
signature. Checkpoints are written only at optimizer boundaries, so no unsaved
partial gradient accumulation is resumed. `initial.pt`, `best.pt`, and `final.pt`
are separate. Load only your own trusted checkpoint files.

The run lock stores immutable Hugging Face revisions. Data/source/config mismatches
stop execution, including stale GPT-2 checkpoints. Benchmarks hash model tensors
and prompt protocols; changed weights cannot silently reuse previous predictions.
One interrupted final JSONL record is discarded on resume. Raw single-pair ranking
and whole-plan guard decisions are retained. The additive cost is a selection
heuristic, not an assertion about jointly merged error.

Main outputs: `checkpoints/`, `logs/`, `manifests/`, `rankings/`, `model_exports/`,
`benchmark_results/pilot/` (or `full/`), `charts/` (300-DPI PNG and vector PDF), and
`REPORT.md`. Serialized size and parameter reduction are reported separately.
GPU throughput describes measured execution, not a promised utilization level.

## Tests

```bash
python -m pytest -q tests
python tools/audit_architecture.py
python tools/build_notebook.py
```

Synthetic CPU tests require no model downloads. CUDA-only tests skip explicitly
when unavailable. Actual GPU checks are mandatory inside `run.py preflight` and
the training stage. The pinned package set is tested locally with CPU PyTorch
2.6.0; compatibility with Kaggle's CUDA/T4 environment remains a runtime check.
The notebook does not install a CPU PyTorch wheel.

Common failures: enable Internet for Hugging Face downloads; choose T4 ×2 for GPU
count errors; use a new output directory after changing training settings; inspect
the exact language/config in a dataset error. On OOM, measured tuning tries smaller
batches; if none is safe, reduce sequence length explicitly in a fresh run. The
program does not replace the merger or loss to work around a failure.
