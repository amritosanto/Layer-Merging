# Heavy multilingual SFT (seven-hour planning target)

Use `Qwen3_8Block_HeavySFT_7Hour_Kaggle.ipynb` with Kaggle Internet and **T4 x2**. It clones this repo and uses `config.smoke_sft13.json` in a fresh `qwen3_8block_sft13_7h` output directory. CUDA PyTorch is preserved. Seven hours is a planning target, never a cutoff or promised runtime.

The eight-block, width-384, six-head merger and losses are unchanged. Ten complete merger epochs use **10,000 / 3,000 / 2,048** training/validation/ranking tokens per language. Equal positive `merge_count_weights` balance depth 1–14 exposure; counts are logged. Progress validation uses a fixed 1,024-token prefix; final validation uses all 3,000. CE selection remains at most 13% above original FP16, with true NF4 measurements at every depth.

Heavy SFT targets **900 training + 100 independent validation examples per language**, seven languages, two epochs. Targets: 40% original Aya annotations, 35% independent verified math word problems, 25% project-authored numeric academic MC (arithmetic, geometry, motion, probability). Generated correct positions are balanced after screening. This numeric curriculum is not a broad factual academic corpus. Aya shortfalls are recorded and reallocated proportionally to verified math/MC; inability to fill total quotas stops SFT. Examples are never duplicated to fill quotas or silently truncated.

Every candidate is audited against frozen benchmark questions/options/answers/supports using normalized hashes, character-shingle similarity and conservative numeric fingerprints for possible translations. Benchmark-source markers and unknown Aya provenance are rejected. The audit addresses detectable added-SFT overlap, not historical pretraining or undetectable semantic contamination. Accepted training/validation IDs, rejected records, actual composition and pinned sources are recorded under `sft/`. Arabic and Albanian benchmark languages receive no dedicated SFT; their prior effective subsets are now explicit, replacing unavailable-English fallbacks.

Both variants share examples, response masks, an independent deterministic epoch sampler, rank 16 / alpha 32 / dropout .05, learning rate 2e-4, accumulation four and sequence limit 512. A longest-sequence GPU preflight tests the common microbatch and reserves optimizer memory. Per-language validation loss is recorded each epoch without benchmark checkpoint selection. Original inference is merged FP16; compressed inference is NF4 plus its separately counted QLoRA adapter.

```bash
python run.py smoke --config config.smoke_sft13.json
# Independent stages: prepare, preflight, train, search, export, sft-prepare,
# sft-original, sft-merged, sft-verify, benchmark, report
python run.py sft-both --config config.smoke_sft13.json
```

`sft-both` trains concurrently on separate GPUs when two GPUs and at least 12 GiB free host memory are available; otherwise sequentially. Checkpoints restore optimizer, scheduler, RNG and exposure counts. Resume with unchanged code/config and all artifacts retained; new settings require a fresh identity. Only verified `original_fp16_sft` and `merged_nf4_sft` are benchmarked. Their accuracy difference combines quantization, merging and SFT.

Reports include correct/incorrect/invalid/truncated counts, paired intervals, option frequencies, label-score margins and charts; an >=80% dominant-option flag diagnoses possible option collapse. Required CSVs are mirrored under `benchmark_results/`, with `rankings/ce_compression_frontier.png`. Storage separates inference files, adapters and architecture parameters. Runtime estimates and actual throughput/memory are recorded, not fabricated.

The last cell checks disk capacity and archives all working files into `Qwen3_8Block_HeavySFT_WORKING.zip` with ZIP64, excluding itself and storing model binaries without recompression. It prints file count/size and attempts browser download. Use **Save Version / View Output** if blocked, or `SALVAGE_PARTIAL=True` for manual recovery. Source artifacts remain intact.

Local tests use synthetic fixtures and tiny models only. No pretrained weights or datasets are downloaded locally. Full T4 execution, runtime and model quality remain unverified.

---
## Earlier instructions (superseded defaults)

# Eight-block merger with paired post-SFT evaluation

Upload `Qwen3_8Block_CE13_100k_20k_SFT_Kaggle.ipynb` to Kaggle, enable Internet and **T4 x2**, then run its cells in order. It clones this repository, installs binary wheels including PEFT 0.15.2, and preserves CUDA PyTorch. Use `config.smoke_sft13.json` and its fresh output directory; six-block checkpoints are incompatible.

The merger uses **eight** existing weight-attention blocks (width 384, six heads), unchanged anchored corrections and losses, and the frozen original FP16 teacher. Ten complete epochs use 100,000 training / 20,000 validation / 4,096 ranking tokens per language across the existing seven Wiki-40B/FineWeb2 languages. No step/time cap is imposed.

CE selection uses `(merged NF4 CE - original FP16 CE) / original FP16 CE <= 0.13`, plus >=40% estimated storage reduction, verified on the selected export. Every merge depth 1–14 receives true NF4 CE; select the deepest feasible evaluated plan, then lower CE. These are **pre-SFT internal diagnostics**, not accuracy benchmarks; no global optimum is claimed. Infeasible diagnostic exports remain explicitly labeled.

```bash
python run.py prepare --config config.smoke_sft13.json
# Other independent stages, in order:
# preflight, train, search, export, sft-prepare, sft-original,
# sft-merged, sft-verify, benchmark, report
python run.py smoke --config config.smoke_sft13.json
```

`sft-prepare` freezes benchmark questions, answers and authentic few-shot supports for auditing only. It streams pinned `CohereLabs/aya_dataset` original annotations; rejects re-annotations/unknown provenance, benchmark-source markers, normalized exact overlaps and conservative character-shingle near/ambiguous matches. All seven language budgets must succeed. The audit fails closed and writes `manifests/sft_contamination_audit.json`, accepted IDs and one response-only token/label dataset shared by both fits. Lexical and available provenance checks cannot prove absence of semantic or translated contamination.

Default SFT: two epochs, rank 16 / alpha 32, learning rate 2e-4, 100 accepted examples per language, maximum 50,000 tokens per language and 512 tokens per example. Oversize examples are rejected, not truncated. Budgets are configurable; actual accepted token counts are identical between models. Both fits use the same seed, schedule and response-only masking, without benchmark checkpoint selection. Original FP16 uses LoRA and exports merged FP16 weights; merged NF4 uses QLoRA and retains its separate adapter for inference. Storage includes that adapter. Each SFT stage uses one T4; merger training keeps dual-GPU execution. Complete optimizer checkpoints resume both fits; incomplete checkpoint writes are ignored.

Only `original_fp16_sft` and `merged_nf4_sft` can be benchmarked, after both finish and `sft-verify` reloads them, checks precision/files/layers and runs synthetic inference. Both receive identical frozen prompts: 100 questions per language, eight-shot MMMLU/INCLUDE/MGSM, 384-token MGSM generation limit. The final language table includes correct counts, accuracy differences, retention, invalid/truncated rates and paired bootstrap intervals. The comparison includes **quantization, merging and SFT together**. The five-point accuracy diagnostic is separate from the 13% CE selection threshold.

The last cell archives all `/kaggle/working` with ZIP64, excludes itself and attempts a browser download. If blocked, use **Save Version / View Output**. Set `SALVAGE_PARTIAL=True` only for manual partial-run recovery. Full checkpoints and adapters remain on disk. Resume with unchanged configuration/code and the complete output folder.

Local checks use synthetic fixtures and tiny randomly initialized models only. No Qwen weights, Aya samples or corpus shards are downloaded locally; full dual-T4 training and pretrained NF4+adapter inference remain unverified until Kaggle execution.

---
## Earlier experiments

# Ten-epoch CE-constrained Qwen3 merging

Upload `Qwen3_10Epoch_CE20_Kaggle_Launcher.ipynb` to Kaggle, enable Internet and **T4 x2**, and run the cells in order. The notebook clones this repository and preserves installed CUDA PyTorch. It uses `config.smoke10.json` and a fresh output directory.

Defaults: ten complete epochs with no step/time cap; seven languages from verified Wiki-40B/FineWeb2 configurations; **10,000 training, 3,000 validation and 2,048 ranking tokens per language**. Partial training windows are retained without duplication. Source revisions, content hashes and full experimental identities govern resume. The frozen FP16 teacher, six-block merger and objective are unchanged.

```bash
python run.py smoke --config config.smoke10.json
# Or run prepare, preflight, train, search, export, benchmark, report separately:
python run.py train --config config.smoke10.json
```

The search evaluates DP candidates at every merge count 1–14, with at most two additional surrogate refinements (16 whole plans total). At least one candidate per count receives true NF4 ranking CE in memory. Select the **most merged pairs** with relative NF4 CE increase `(merged-base)/base <= 0.20` and >=40% estimated storage reduction against original 16-bit storage; ties use lower CE then pair indices. Final serialized size verifies storage feasibility. If no plan qualifies, one lowest-CE learned diagnostic is exported and explicitly marked infeasible. Savings above 50% are welcome. This is the best evaluated plan, not a global optimum. Separate full validation uses all 3,000 tokens per language.

`rankings/nf4_finalists.csv` includes accepted and rejected plans plus original NF4 (zero merges); `pareto_frontier.csv` and `charts/ce_compression_pareto.png` show non-dominated measured candidates. Surrogate CE and estimated storage are labeled separately from NF4 measurements and final serialized sizes. Only one learned NF4 checkpoint is exported.

Benchmarks compare only original NF4 and learned NF4 with identical quantization, questions and prompts: 100 questions per language, prior three-language subsets, eight-shot MMMLU/INCLUDE/MGSM, and a 384-token MGSM generation cap. Counts, invalid/truncated rates, answer-position counts and paired bootstrap 95% accuracy-difference intervals are reported. The five-percentage-point accuracy diagnostic is independent of the 20% CE constraint. Benchmark results do not select the plan.

Resume unchanged stages with the complete output directory; changed code/configuration requires a fresh experiment. Retain checkpoints, manifests, original NF4 reference and cached samples/predictions. A partial run is never labeled complete. Abrupt process termination resumes from the last persisted optimizer boundary; intervening uncommitted work may be repeated.

The last notebook cell archives **all `/kaggle/working`** with ZIP64, excluding itself and storing model binaries without recompression. It requires a completed report by default; manually set `SALVAGE_PARTIAL=True` to archive an interrupted run. It attempts a browser download, which Kaggle/browser policies may block. The archive remains available under **Save Version / View Output**. Ensure sufficient free disk space for a second copy of working files; originals are not deleted.

Local checks use tiny synthetic fixtures only. Full pretrained NF4 export, dual-T4 execution and benchmark accuracy remain unverified until Kaggle runs. No models or corpus shards are downloaded by local tests.

---
## Earlier five-epoch experiment

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
