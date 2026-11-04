# Validation status

Local checks were run on 2026-11-04 using an isolated Python 3.12 environment,
CPU PyTorch 2.6.0, Transformers 4.51.3 and the pinned project dependencies.
The installed system PyTorch had a CUDA DLL import failure; it was not modified.

## Executed

- **25 tests passed; one CUDA-only NF4 test skipped.** Machine-readable results:
  `validation/tests.xml`.
- Nonzero correction outputs match the authoritative reference implementation
  exactly under matching test families, parameters and inputs.
- Synthetic Qwen3 tests cover GQA/non-square projections, Q/K RMSNorm, RoPE,
  patch reconstruction, initial anchors, simultaneous replacements, nonzero
  gradients, frozen teacher/source integrity, chunked KL values and gradients,
  saved-model reload, KV-cache generation and reconstructed control equivalence.
- Evaluator tests check multi-token candidate likelihood against a full-logit
  calculation, numeric normalization, invalid outputs, disjoint support selection,
  partial-line resume, paired bootstrap and reporting/chart generation.
- All 19 notebook code cells compile, and the embedded source archive matches the
  project files byte-for-byte. Run `python tools/validate_notebook.py` to verify.
- Live Hugging Face checks loaded the actual Qwen3 configuration and tokenizer,
  discovered 41 Wiki-40B configs, 14 MMMLU translated configs plus its combined
  default, and 45 INCLUDE configs representing 44 languages. Representative
  Wiki-40B/MMMLU/INCLUDE rows passed their adapters. All 11 MGSM configs loaded;
  each had 8 training and 250 test examples, with matching test answer sequences
  across languages. Details and immutable revisions: `validation/live_schemas.json`.

## Not executed here

- Pretrained model weight loading/inference or training.
- CUDA NF4 extraction/requantization and storage checks.
- Dual-T4 forward/backward reduction, actual batch tuning and throughput.
- Full multilingual Wiki-40B preparation, GPU export preflight or few-shot
  benchmark inference. Live schema checks are not benchmark results.
- The optional official TFDS fallback (normal HF sampling was checked).

These checks are implemented as runtime stages, with failures stopping execution.
Kaggle accelerator compatibility and memory capacity must be established there;
the notebook is not claimed to have completed a fresh dual-T4 run.

`outputs/REPORT.md` deliberately contains pending results. Synthetic reporting
fixtures are confined to temporary test directories and are not research outputs.
