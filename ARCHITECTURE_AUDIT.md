# Reference audit and port decisions

Authoritative input: `GPT2Merger2.0.ipynb`, supplied from
`C:/Users/amrit/OneDrive/Desktop/gpt2-layermerging-main/Assignment2/Amrita/`.
All notebook cells were inspected before implementation. The main implementation
cell is preserved in `reference/gpt2_reference.py`; the original notebook is unchanged.
Reference comments and installation commands were treated as source material,
not as permission to execute the original experiment.

| Reference behavior | Qwen3 implementation |
|---|---|
| `WeightAttentionBlock`: pre-LN, explicit QKV, unmasked SDPA, GELU FFN | Same operations, six blocks, width 384, six heads, multiplier 4 |
| Shape-shared encoders and decoders | Shared by actual patch shape; not independent heads for every family |
| Scale = sqrt(mean squared A/B), minimum .01 | Same FP32 formula |
| Source A tokens followed by source B tokens | Same ordering and matched-token fusion |
| Learned family, source, layer, row and column embeddings | Same roles; tables sized to Qwen3 shapes and 28 layers |
| Zero decoder heads; `B + delta` | Same initialization and anchor; no averaging initialization |
| GPT-2 combined QKV, Conv1D orientation | Qwen3 separate Linear matrices, output-by-input orientation |
| Every block matrix, bias and LayerNorm vector | All 11 actual Qwen3 block parameters; no invented biases |
| `task_for_batch`: (1,1,2,3,4,5) repeating mixture | Same mixture; uniform compatible-set sampling on 27 edges |
| Teacher raw output after second block | Same original-index boundary, before final model normalization |
| Joint student accumulates previous merge errors | Same sequential joint execution |
| `T² KL(teacher || student)` | Same direction/scaling; checkpointed token chunks, FP32 log-softmax |
| Family-normalized correction MSE, then mean over pairs | Same formula |
| Equal-level checkpoint CE gaps vs matching drop-first | Same criterion; all singles plus two fixed plans per joint level |
| Shard-weighted losses, scaled gradient sum, unscale, clip | Same synchronous two-device method, with gradient accumulation |
| Additive guarded single-pair PPL-change ranking | Retained as control; separate raw learned ranking is primary |
| Whole-plan fallback to drop-first | Retained and labelled; raw exports are always produced |

## Parameter and token accounting

Counts below are architecture calculations, **not experimental performance**.
`tools/audit_architecture.py` instantiates the reference merger and the port on the
meta device. The full machine-readable table is `reference/architecture_counts.json`.

| Family | Tensor shape | Patch | Tokens per source |
|---|---|---|---:|
| Q | 2048 × 1024 | 128 × 128 | 128 |
| K | 1024 × 1024 | 128 × 128 | 64 |
| V | 1024 × 1024 | 128 × 128 | 64 |
| O | 1024 × 2048 | 128 × 128 | 128 |
| Q norm | 128 | 128 | 1 |
| K norm | 128 | 128 | 1 |
| Gate | 3072 × 1024 | 128 × 128 | 192 |
| Up | 3072 × 1024 | 128 × 128 | 192 |
| Down | 1024 × 3072 | 128 × 128 | 192 |
| Input norm | 1024 | 128 | 8 |
| Post-attention norm | 1024 | 128 | 8 |

978 tokens per source, **1,956 tokens per pair**. The reference uses 2,040 tokens.
The original merger has **39,391,168 parameters**; this port has **23,676,288**,
approximately **39.9% fewer**. This is a material change in merger capacity. The
reference has three distinct matrix-patch shapes; all Qwen3 matrices here share
one 128 × 128 patch shape, so the existing shape-sharing rule yields fewer
encoder/decoder parameters. The six attention blocks and fusion mechanism are
preserved. The merger is a training/model-generation tool and is excluded from
exported compressed-model parameter counts.

## Numerical representations

1. Teacher: original frozen FP16 Qwen3.
2. Source: actual bitsandbytes NF4 load, with double quantization and FP16 compute.
3. Training source: complete dense FP16 reconstruction of those NF4 tensors.
4. Generated corrections: FP32 reconstruction arithmetic, as in the reference;
   differentiable FP16 casts at layer execution match the saved FP16 representation.
5. C: all-dense FP16 compressed model derived from the quantized source.
6. D: a fresh NF4 conversion of C, separately evaluated and serialized.

Manual Qwen3 execution supplies causal masks, RoPE embeddings, positions and
cache positions to the actual installed decoder implementation. Output type is
checked for tuple/tensor compatibility. Runtime preflight compares manual teacher
execution with the normal Transformers forward path. Export reindexes attention
layer indices for KV-cache generation and updates `num_hidden_layers`.

## Tests and remaining execution requirements

CPU tests use synthetic small Qwen3 models and include a numerical comparison of
nonzero corrections against the extracted reference implementation. They do not
claim pretrained accuracy, NF4 GPU correctness, or dual-T4 throughput. The runtime
preflight checks actual NF4 presence/storage reduction, source hashes, all-family
reconstruction, all 27 zero anchors, joint levels 1–5, teacher matching, gradient
flow/reduction, and export/reload/generation. Benchmark preflight executes actual
few-shot samples separately before the training stage in the notebook.

Primary implementation/schema references:

- [Pinned Qwen3 implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py)
- [Original model configuration](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json)
- [Wiki-40B](https://huggingface.co/datasets/google/wiki40b)
- [MMMLU](https://huggingface.co/datasets/openai/MMMLU)
- [INCLUDE lite](https://huggingface.co/datasets/CohereLabs/include-lite-44)
- [MGSM](https://huggingface.co/datasets/juletxara/mgsm)
