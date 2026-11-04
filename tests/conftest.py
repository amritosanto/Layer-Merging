import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import src
import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from config import Config
from src.qwen3_adapter import families_from_block
from src.weight_patching import make_families
from src.merger import PairWeightMerger

@pytest.fixture
def tiny():
    torch.set_num_threads(2); torch.manual_seed(42)
    c=Qwen3Config(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=12,
                  num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=128,
                  attention_dropout=0.,eos_token_id=2,pad_token_id=0,bos_token_id=1)
    c._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(c).eval().requires_grad_(False)
    cfg=Config(merger_width=12,merger_heads=3,matrix_patch=8,vector_patch=8,logit_chunk_tokens=3)
    families=make_families(families_from_block(model.model.layers[0]),8,8)
    merger=PairWeightMerger(cfg,families,12)
    ids=torch.randint(3,64,(2,7))
    return model,merger,cfg,ids,families
