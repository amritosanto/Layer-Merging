# Qwen3 CE-selected NF4 merger (experimental v2)

This version evaluates **one learned, CE-selected NF4 model** on MMMLU, INCLUDE, and MGSM.
It never benchmarks averaging, drop-first, drop-second, original FP16 or original NF4 as competing variants.
The frozen original FP16 Qwen3 remains the teacher for Wiki-40B merger training; merged source weights come from a pre-quantized NF4 model (dequantized for differentiable merging). No teacher weights are updated.

## Run on Kaggle (two T4 GPUs, Internet enabled)

```bash
python -m pip install -r requirements.txt
python run.py prepare --config config.json
python run.py preflight --config config.json
python run.py train --config config.json
python run.py rank-export --config config.json
python run.py benchmark-preflight --config config.json
python run.py benchmark --config config.json --benchmark MMMLU
python run.py benchmark --config config.json --benchmark INCLUDE
python run.py benchmark --config config.json --benchmark MGSM
python run.py report --config config.json
```

`rank-export` is now the single-model **CE search and NF4 selection** stage. It must finish before benchmark-preflight. There is only one export, `model_exports/selected_learned_nf4`, and only one evaluation variant, `selected_learned_nf4`.

### Selection: what is and is not guaranteed

The code measures CE on held-out Wiki-40B data. It ranks all 27 possible adjacent original-layer pairs, uses an exact dynamic program to find top-N **additive surrogate** combinations for every merge count from 1 through 14, and re-evaluates each shortlisted combination with actual whole-model CE. The best shortlisted models are serialized and verified in NF4 for CE and on-disk size. The combinatorial search is **not an exhaustive global optimizer** of full-model CE, because interactions between merged layers are non-additive. `max_search_pairs`, `additive_candidates_per_count`, and `nf4_verify_limit` control the search budget.

Selection requires <=5% **perplexity increase** relative to the original FP16 teacher as a *surrogate constraint*, not a validated <=5% benchmark accuracy loss. The latter cannot be guaranteed before running the benchmark. After selecting a model, the program benchmarks only that NF4 model. Existing previous original-model benchmark predictions can be compared later only if the question IDs and prompt protocol match.

The nominal size target is 40–50% smaller **serialized NF4 model** compared with the original serialized FP16 model. NF4 alone may already save more than 50%. When all measured candidates exceed the 50% upper reduction bound, `allow_nf4_size_overshoot=True` permits a clearly labeled, smallest-excess fallback *only if the CE surrogate holds*. Set it to false for strict 40–50% acceptance. If no measured candidate satisfies the CE surrogate, ranking stops without an export or benchmark rather than claiming success.

**Training distribution caveat:** The existing learner's training plans and validation still use up to five simultaneous merged pairs. Searches with 6–14 merged pairs therefore extrapolate beyond training. Deeper compression needs a separately optimized dual-T4 training schedule if these candidates are to be competitive.

### Important artifacts

- `rankings/single_pairs.csv`: CE of each learned single-pair replacement.
- `rankings/whole_plan_candidates.csv`: whole-model CE before final NF4 requantization.
- `rankings/verified_nf4_candidates.json`: actually measured CE and size for verified NF4 candidates.
- `manifests/selection.json`: final decision, target attainment and explicit infeasibility when applicable.
- `manifests/exports.json`: exactly one selected NF4 artifact.
- `benchmark_results/pilot/selected_learned_nf4/`: the sole benchmark prediction group.

Use a **fresh output directory**. v2 deliberately rejects old v1 training checkpoints/config locks so that previous results cannot be mistaken for the new protocol. Source-level tests and CUDA preflight remain required; pretrained training, NF4 export, and benchmarks are *not yet validated on Kaggle*.
