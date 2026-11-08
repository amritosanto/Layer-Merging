"""CPU regression checks for the four-term fixed-pair experiment."""
import torch
from config import Config
from src.objectives import local_hidden_distribution_kl


def fixed_config():
    return Config(
        post_sft=True,
        merger_blocks=8,
        ce_reference='original_fp16',
        fixed_merge_pairs=(2,4,6,8),
        min_merged_pairs=4,
        max_merged_pairs=4,
        max_relative_ce_increase=None,
        local_kl_weight=1.0,
        local_kl_temperature=2.0,
    ).validate()


def test_fixed_plan_and_no_ce_gate():
    cfg=fixed_config()
    assert tuple(cfg.fixed_merge_pairs)==(2,4,6,8)
    assert cfg.max_relative_ce_increase is None
    assert cfg.epochs==10
    assert cfg.ce_weight==0.0
    assert cfg.sft_epochs==2  # dataclass default; notebook pins original fast-SFT value 1


def test_local_hidden_kl_zero_for_identical_states():
    h=torch.randn(2,3,128)
    result=local_hidden_distribution_kl(h,h,temperature=2.0)
    assert torch.isfinite(result)
    assert abs(result.item()) < 1e-6


def test_local_hidden_kl_has_finite_nonzero_gradients():
    torch.manual_seed(42)
    teacher=torch.randn(2,4,128)
    student=(teacher.detach()+torch.randn_like(teacher)*0.5).requires_grad_()
    loss=local_hidden_distribution_kl(student,teacher,temperature=2.0)
    assert loss.item() > 0 and torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert student.grad.abs().sum().item()>0
    assert teacher.grad is None


def test_overlapping_fixed_plan_is_rejected():
    try:
        Config(post_sft=True,merger_blocks=8,ce_reference='original_fp16',
               fixed_merge_pairs=(2,3,4,6),min_merged_pairs=4,max_merged_pairs=4,
               max_relative_ce_increase=None).validate()
    except AssertionError:
        return
    raise AssertionError('Overlapping pair plan incorrectly accepted')
