from dataclasses import dataclass, asdict
from pathlib import Path
import json

VERSION = 'qwen3-eight-block-sft-v5'

@dataclass
class Config:
    output_dir: str = '/kaggle/working/Qwen3Merger'
    model_id: str = 'Qwen/Qwen3-0.6B-Base'
    model_revision: str = 'main'  # Resolved once to immutable SHA in run lock.
    mode: str = 'standard'
    quantization: str = 'nf4'
    teacher_precision: str = 'fp16'
    max_optimizer_steps: int = 0
    min_merged_pairs: int = 1
    max_merged_pairs: int = 14
    max_candidate_plans: int = 12
    max_refinement_rounds: int = 2
    max_final_nf4_candidates: int = 3
    ce_reference: str = 'original_nf4'
    post_sft: bool = False
    sft_dataset: str = 'CohereLabs/aya_dataset'
    sft_revision: str = 'main'  # Resolved to SHA in the immutable run lock.
    sft_epochs: int = 2
    sft_lora_rank: int = 16
    sft_lora_alpha: int = 32
    sft_learning_rate: float = 2e-4
    sft_examples_per_language: int = 100
    sft_tokens_per_language: int = 50000  # Maximum; identical accepted tokens for both models.
    sft_max_length: int = 512
    sft_batch_size: int = 1
    sft_gradient_accumulation: int = 4
    sft_save_steps: int = 10
    sft_near_duplicate_threshold: float = 0.45  # Conservative ambiguous-match rejection.
    max_relative_ce_increase: float = None  # None retains the earlier low-CE experiment.
    ce_comparable_tolerance: float = 0.01
    min_storage_reduction: float = 0.40
    target_storage_reduction: float = 0.50
    accuracy_loss_target: float = 0.05
    benchmark_languages: dict = None
    seed: int = 42
    epochs: int = 10
    training_hours: float = 7.0
    train_seq_len: int = 256
    max_global_batch_size: int = 16
    gradient_accumulation: int = 1
    peak_memory_fraction: float = 0.85
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    kd_weight: float = 1.0
    correction_weight: float = 0.01
    ce_weight: float = 0.0
    temperature: float = 2.0
    logit_chunk_tokens: int = 32
    merger_width: int = 384
    merger_heads: int = 6
    merger_blocks: int = 6
    merger_ff_multiplier: int = 4
    matrix_patch: int = 128
    vector_patch: int = 128
    train_tokens_per_language: int = 12000
    validation_tokens_per_language: int = 1024
    ranking_tokens_per_language: int = 1024
    wiki_dataset: str = 'google/wiki40b'
    fineweb_dataset: str = 'HuggingFaceFW/fineweb-2'
    fineweb_revision: str = 'af9c13333eb981300149d5ca60a8e9d659b276b9'
    training_languages: tuple = ()
    language_replacements: dict = None
    wiki_backend: str = 'hf'  # 'tfds' explicitly selects the official fallback release.
    wiki_languages: tuple = ()  # Empty discovers all available configs.
    wiki_shuffle_buffer: int = 128
    max_articles_per_language: int = 100000
    allow_missing_languages: bool = False
    single_gpu_debug: bool = False
    save_every_steps: int = 50
    validation_interval_minutes: float = 30.0
    accept_margin_ce: float = 0.001
    evaluation_mode: str = 'pilot'
    pilot_questions: int = 100
    mmmlu_few_shot: int = 5
    include_few_shot: int = 5
    mgsm_few_shot: int = 8
    include_dataset: str = 'CohereLabs/include-lite-44'
    eval_context: int = 4096
    eval_batch_size: int = 4
    max_new_tokens: int = 512
    mgsm_max_new_tokens: int = 256
    bootstrap_samples: int = 2000

    def validate(self):
        assert self.mode in ('smoke','standard')
        assert self.quantization == 'nf4' and self.teacher_precision == 'fp16'
        assert 1 <= self.min_merged_pairs <= self.max_merged_pairs <= 14
        assert self.max_candidate_plans >= 1 and self.max_refinement_rounds >= 0
        assert self.max_optimizer_steps >= 0
        if self.max_relative_ce_increase is not None:
            assert 0 <= self.max_relative_ce_increase < float("inf")
            levels=self.max_merged_pairs-self.min_merged_pairs+1
            assert self.max_candidate_plans >= levels+self.max_refinement_rounds
            assert self.max_final_nf4_candidates >= levels
        assert self.train_seq_len >= 2 and self.pilot_questions > 0
        assert self.save_every_steps > 0 and self.bootstrap_samples > 0
        assert self.mgsm_few_shot == 8
        assert 1 <= self.max_final_nf4_candidates <= self.max_candidate_plans
        assert self.ce_comparable_tolerance >= 0 and self.mgsm_max_new_tokens > 0
        assert 0 < self.min_storage_reduction <= self.target_storage_reduction < 1
        assert 0 <= self.accuracy_loss_target <= 1
        assert self.ce_weight == 0, 'Preserve hidden + KD + correction loss'
        assert self.merger_width % self.merger_heads == 0
        assert self.merger_blocks in (6,8)
        assert self.ce_reference in ('original_nf4','original_fp16')
        assert min(self.sft_epochs,self.sft_lora_rank,self.sft_lora_alpha,self.sft_examples_per_language,self.sft_tokens_per_language,self.sft_batch_size,self.sft_gradient_accumulation,self.sft_save_steps)>0
        assert self.sft_max_length>=2 and self.sft_learning_rate>0
        assert 0<self.sft_near_duplicate_threshold<=1
        if self.post_sft:
            assert self.merger_blocks==8 and self.ce_reference=='original_fp16'
            assert self.merger_width==384 and self.merger_heads==6
            assert self.max_relative_ce_increase is not None
            assert self.sft_dataset=='CohereLabs/aya_dataset'
        assert self.evaluation_mode in ('pilot', 'full')
        assert self.wiki_backend in ('hf','tfds')
        assert self.gradient_accumulation >= 1 and self.temperature > 0
        assert self.epochs > 0 and self.training_hours >= 0
        assert self.max_global_batch_size >= (1 if self.single_gpu_debug else 2)
        assert 0 < self.peak_memory_fraction < 1
        assert min(self.mmmlu_few_shot, self.include_few_shot, self.mgsm_few_shot) > 0
        assert min(self.train_tokens_per_language,self.validation_tokens_per_language,self.ranking_tokens_per_language) >= 2
        return self

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8'))).validate()
