# Benchmark meta-tensor crash: diagnosis and recovery

## Observed failure

The 2-hour SFT Kaggle pilot (source revision `0752fbdc72e972195b39e745336b8a1f82323c46`) completed both SFT runs and `sft-verify`, then failed during `benchmark` with:

```
NotImplementedError: Cannot copy out of meta tensor; no data!
```

The old benchmark code loaded two FP16/QLoRA inference replicas concurrently using `ThreadPoolExecutor`. During the merged NF4 + PEFT adapter load, warnings showed that LoRA tensors were being copied into **meta** parameters (a no-op). The later `module_hash(model)` raised when hashing an unmaterialized parameter. This is an evaluation-time model-loading failure, not evidence that two completed SFT fits failed.

## Fix for new runs

The corrected `run.py` launches independent GPU-specific **subprocesses** (`benchmark-worker --worker-rank 0/1`) for each variant instead of shared Python threads. Each worker sees one GPU using `CUDA_VISIBLE_DEVICES`, loads its model independently, evaluates its assigned languages, and writes its own resumable prediction files. `src/benchmark_evaluation.py` explicitly rejects unmaterialized meta tensors before hashing or inference. The 13% CE selection rule, merger training, SFT datasets, and benchmark prompts are not changed.

```bash
python run.py benchmark --config /kaggle/working/qwen-smoke-run.json
python run.py report --config /kaggle/working/qwen-smoke-run.json
```

**Important: old checkpoints are source-locked.** `prepare_run` fingerprints implementation files. Updating the repository can invalidate an existing run's implementation lock. Do **not** edit or overwrite `manifests/run.json` or disable integrity checks.

## Recovery of an already-finished old run

1. Restore all saved files from `/kaggle/working/outputs/qwen3_8block_sft_2hour`. The Kaggle notebook alone does not contain model weights.
2. Use the original source revision **`0752fbdc72e972195b39e745336b8a1f82323c46`**, so run/source signatures remain valid.
3. In a separate launcher process, import `src.benchmark_evaluation` and override **only in memory** `ThreadPoolExecutor` with a serial context-manager executor, then run the original `run.py benchmark` via `runpy.run_path`. This runs GPU 0 then GPU 1 without overlapping model initialization. It does not rewrite any source file.
4. The benchmark's existing prediction cache reuses already completed question IDs, then `run.py report` summarizes results.
5. Use the provided `Qwen3_8Block_SFT_Benchmark_Recovery_Kaggle.ipynb` notebook for these steps.

This recovery workaround uses two GPU devices **sequentially**. New runs use the permanent two-process fix and benchmark both GPUs concurrently.

## Verification limitations

The fix was reviewed against the uploaded traceback and the existing GitHub implementation. It has not been GPU-tested on a live Kaggle T4 x2 instance. The only definitive test is that the updated notebook runs its benchmark and verifies both models have no meta tensors.
