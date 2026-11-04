# Multilingual Capability Preservation Through Quantization-Aware Learned Transformer Layer Merging in Qwen3

Results are pending; training and benchmark execution have not been established.

## 1. Research objective and hypothesis

Test whether the supplied second-block-anchored weight-space merger preserves multilingual capabilities after NF4 source quantization. No benchmark scores select weights or layer pairs.

## 2. Original GPT-2 merger architecture

Six bidirectional pre-LN weight-attention blocks; shared shape encoders/decoders; RMS scaling; learned source, layer, family and patch-position embeddings; matched-patch fusion. Zero correction heads reproduce drop-first.

## 3. Adaptation to Qwen3

Runtime-derived Qwen3 families include GQA matrices, Q/K RMSNorm, SwiGLU and both block RMSNorm vectors. Linear weights use output-by-input orientation. See manifests/architecture.json and ARCHITECTURE_AUDIT.md.

## 4. Quantization methodology

Frozen original FP16 teacher. NF4 with double quantization and FP16 compute is applied before extracting source matrices. The entire student training representation is dequantized FP16; generated corrections use FP32 arithmetic and differentiable FP16 casts for decoder execution. C and D are separately labelled. Dense controls share the same dequantized source.

## 5. Wiki-40B data and language coverage

Results pending. No experimental measurements available.

## 6. Training configuration

Configuration: config.json and manifests/run.json. Defaults: 10 epochs, seven-hour soft training budget, learning rate 1e-4, KL weight 1, correction weight .01, CE weight 0, T=2. Validation gives each compression level equal weight.

## 7. Dual-GPU implementation

Synchronous replica training uses shard-weighted losses and sums scaled gradients before AMP unscale and clipping. Measurements, when present, are in logs/batch_tuning.csv and logs/training.csv. VRAM allocation is not a compute-utilization measurement.

## 8. Layer-pair selection

Raw and guarded additive single-pair rankings select separate nonoverlapping plans. Every selected whole plan is measured independently. Raw learned models are always exported. The guard accepts a complete plan only if ranking CE improves over matching drop-first by the configured margin.

## 9. MMMLU evaluation

Results pending. No experimental measurements available.

## 10. INCLUDE evaluation

Results pending. No experimental measurements available.

## 11. MGSM evaluation

Results pending. No experimental measurements available.

## 12. Multilingual performance comparison

Results pending. No experimental measurements available.

## 13. Compression efficiency

Results pending. No experimental measurements available.

## 14. Analysis of language-specific failures

Results pending. No experimental measurements available.

## 15. Limitations

Pilot scores are sampled, non-official few-shot evaluations. MMMLU/INCLUDE test-derived support is excluded from scoring. Context-overflow questions are explicitly skipped for all variants; coverage is reported. Checkpoint validation may exceed the soft time budget. This experiment does not establish generalization to other checkpoints. GPU execution is only verified when runtime preflight manifests exist.

## 16. Reproducibility information

Immutable model/dataset revisions, prompt hashes, source hashes, RNG states and software versions are saved in manifests and checkpoints. Resume requires the same configuration and data. Local tests use synthetic small Qwen3 models; they do not establish NF4 or dual-T4 correctness.

## 17. Conclusions and possible future modifications

Results pending. No experimental measurements available.

## Figures

