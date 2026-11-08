"""Explicit fixed-pair export, with CE measured only as a diagnostic.

This path must never run automatic pair selection or enforce CE/storage cutoffs.
The optimization objective is four-term teacher/student preservation, not CE search.
"""
import json
import math
from pathlib import Path

import torch

from .model_export import export_best_nf4
from .pair_selection import valid_plan
from .utils import atomic_json, file_digest


def export_fixed_plan(experiment, saved, cache_key, checkpoint_hash, export=True):
    e=experiment
    cfg=e.cfg
    out=Path(cfg.output_dir)
    pairs=valid_plan(tuple(cfg.fixed_merge_pairs),len(e.source.model.layers))
    if len(pairs)!=4:
        # Allow other fixed plans in future, but the present experiment is four pairs.
        # Validation of the user-requested (2,3),(4,5),(6,7),(8,9) occurs in config.
        assert len(pairs)==cfg.min_merged_pairs==cfg.max_merged_pairs
    from .ce_search import reference_storage, storage_estimate
    original=reference_storage(cfg,e.lock)
    reference_ranking=e.score(partition='ranking',model=e.teacher)[0]
    reference_validation=e.score(partition='validation',model=e.teacher)[0]
    surrogate_ranking=e.score(pairs=pairs,partition='ranking')[0]
    parameters,estimate_bytes=storage_estimate(e.source,pairs)
    baseline=dict(reference_precision=cfg.ce_reference,
                  reference_ranking=reference_ranking,
                  reference_validation=reference_validation,
                  nf4_serialized_bytes=e.quant_manifest['nf4_serialized_bytes'],
                  reference_storage=original)
    diagnostics=dict(selection='Fixed by user; no CE threshold, pair search or storage gate',
                     objective='normalized hidden MSE + local hidden feature KL + final vocabulary KL + correction',
                     fixed_pairs_zero_based=list(pairs),
                     original_layer_pairs=[[p,p+1] for p in pairs],
                     pairs=list(pairs),merged_pairs=len(pairs),
                     remaining_layers=len(e.source.model.layers)-len(pairs),
                     completed_epochs=saved['state']['epoch'],
                     training_updates=saved['state']['step'],
                     checkpoint_hash=checkpoint_hash,cache_key=cache_key,
                     max_relative_ce_increase=None,
                     baseline=baseline,surrogate_ranking_ce=surrogate_ranking,
                     estimated_parameters=parameters,estimated_nf4_bytes=estimate_bytes,
                     local_kl_weight=cfg.local_kl_weight,
                     local_kl_temperature=cfg.local_kl_temperature,
                     kd_weight=cfg.kd_weight,
                     correction_weight=cfg.correction_weight,
                     hidden_mse_weight=1.0)
    atomic_json(diagnostics,out/'rankings/fixed_plan_diagnostics.json')
    if not export:
        return diagnostics
    e.integrity()
    model,info=export_best_nf4(e.source,e.merger,e.tokenizer,pairs,out/'model_exports/best_learned_nf4')
    ranking=e.score(partition='ranking',model=model)[0]
    validation=e.score(partition='validation',model=model)[0]
    if not all(math.isfinite(x) for x in (reference_ranking,reference_validation,ranking,validation)):
        raise FloatingPointError('Nonfinite fixed-plan CE diagnostic')
    original_bytes=original['bytes']
    report=dict(diagnostics,**info,ranking_ce=ranking,
                reference_ce=reference_ranking,
                relative_ce_increase=(ranking-reference_ranking)/reference_ranking,
                final_nf4_validation_ce=validation,
                final_nf4_ce_increase=validation-reference_validation,
                actual_reduction=1-info['serialized_bytes']/original_bytes,
                reduction_vs_original_nf4=1-info['serialized_bytes']/baseline['nf4_serialized_bytes'],
                ce_target_met=None,storage_target_met=None,feasible=None,
                target_50_met=None,ce_diagnostic_only=True,
                inference_format='Fixed four-pair NF4 base before paired SFT')
    del model
    root=out/'model_exports/best_learned_nf4'
    report['files']={str(p.relative_to(out)):file_digest(p) for p in root.rglob('*') if p.is_file()}
    atomic_json(report,out/'rankings/selected_plan.json')
    atomic_json(report,out/'manifests/final_export.json')
    return report
