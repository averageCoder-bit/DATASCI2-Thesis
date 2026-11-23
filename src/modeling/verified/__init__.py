"""LGMMA-X reviewed reusable source for the **frozen-inference** experiment.

This package is import-safe and has no output-writing side effects.
It is not a claim of original Exp04 training reproducibility.
"""

from .calibration import fit_gmm_candidates, negative_log_density, select_validation_threshold
from .evaluation import align_proxy_labels, evaluate_heldout
from .reconstruction import reconstruction_mae, restored_exp04_model, checkpoint_tensors

__all__ = [
    "fit_gmm_candidates", "negative_log_density", "select_validation_threshold",
    "align_proxy_labels", "evaluate_heldout", "reconstruction_mae",
    "restored_exp04_model", "checkpoint_tensors",
]
