from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.quantization import load_model, MODEL_LOAD_LOCK


def test_concurrent_loading_preserves_checkpoint_weights(tmp_path):
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=8, tie_word_embeddings=True)
    original = Qwen3ForCausalLM(config).eval().half()
    original.save_pretrained(tmp_path)
    expected = original.state_dict()
    original_dtype = torch.get_default_dtype()
    # Repeated simultaneous entry reproduces the old meta-tensor race on CPU.
    for _ in range(6):
        barrier = Barrier(2)
        def load(_):
            barrier.wait(timeout=10)
            return load_model(str(tmp_path), None, 'cpu')
        with ThreadPoolExecutor(max_workers=2) as pool:
            models = list(pool.map(load, range(2)))
        for model in models:
            assert model.lm_head.weight is model.model.embed_tokens.weight
            for name, value in model.state_dict().items():
                assert not value.is_meta
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        assert torch.get_default_dtype() == original_dtype


def test_failed_load_releases_lock(monkeypatch):
    from src import quantization
    def fail(*args, **kwargs):
        raise RuntimeError('simulated load failure')
    monkeypatch.setattr(quantization.AutoModelForCausalLM, 'from_pretrained', fail)
    with pytest.raises(RuntimeError, match='simulated load failure'):
        load_model('unused', None, 'cpu')
    def acquire_from_another_thread():
        acquired = MODEL_LOAD_LOCK.acquire(timeout=2)
        if acquired:
            MODEL_LOAD_LOCK.release()
        return acquired
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(acquire_from_another_thread).result(timeout=5)
