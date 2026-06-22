"""Model explainability utilities."""

from .gradcam import compute_gradcam, find_last_conv_layer, gradcam_heatmap, overlay_heatmap
from .vit_rollout import attention_rollout, cls_attention, extract_attention_matrices, upscale_to_image
from .feature_importance import gradient_sensitivity, permutation_importance
from .faithfulness import deletion_auc, insertion_auc
from .protocol import ExplainConfig, ExplainResult, run_explainability

__all__ = [
    "compute_gradcam",
    "find_last_conv_layer",
    "gradcam_heatmap",
    "overlay_heatmap",
    "attention_rollout",
    "cls_attention",
    "extract_attention_matrices",
    "upscale_to_image",
    "ExplainConfig",
    "ExplainResult",
    "run_explainability",
    "deletion_auc",
    "insertion_auc",
    "gradient_sensitivity",
    "permutation_importance",
]
