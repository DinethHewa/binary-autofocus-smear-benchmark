from __future__ import annotations

from typing import Any, Dict, List

from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)


DEFAULT_SPACES: Dict[str, Dict[str, Any]] = {
    "cnn": {
        "num_blocks": (2, 5),
        "filters_base": (16, 64, 8),
        "kernel_size": [3, 5],
        "dropout": [0.0, 0.2, 0.4],
        "batchnorm": [True, False],
        "dense_units": [0, 64, 128, 256],
        "l2_reg": [0.0, 1e-4, 1e-3],
        "optimizer": ["adam", "rmsprop"],
        "learning_rate": [1e-4, 5e-4, 1e-3],
    },
    "cnn_attention": {
        "attention_type": ["se", "cbam", "none"],
        "se_ratio": [4, 8, 16],
        "attention_placement": ["all", "last_two"],
        "spatial_kernel": [3, 7],
        "num_blocks": (2, 5),
        "filters_base": (16, 64, 8),
        "kernel_size": [3, 5],
        "dropout": [0.0, 0.2, 0.4],
        "batchnorm": [True, False],
        "dense_units": [0, 64, 128, 256],
        "l2_reg": [0.0, 1e-4, 1e-3],
        "optimizer": ["adam", "rmsprop"],
        "learning_rate": [1e-4, 5e-4, 1e-3],
    },
    "transfer": {
        "pooling": ["avg", "max"],
        "head_units": [0, 128, 256],
        "dropout": [0.0, 0.2, 0.4],
        "lr": [1e-5, 1e-4, 5e-4],
        "label_smoothing": [0.0, 0.05],
        "base_trainable_blocks": [0, 1, 2, 3],
    },
    "vit": {
        "patch_size": [8, 16],
        "embed_dim": [64, 128, 192, 256],
        "num_heads": [2, 4, 8],
        "depth": [2, 4, 6],
        "mlp_dim": [128, 256, 384, 512],
        "dropout": [0.0, 0.1, 0.2],
        "use_cls_token": [False, True],
    },
    "hybrid_vit": {
        "stem_type": ["cnn", "mobilenetv2"],
        "embed_dim": [64, 96, 128, 192],
        "transformer_depth": [1, 2, 3, 4],
        "num_heads": [2, 4],
        "mlp_dim": [128, 256, 384],
        "dropout": [0.0, 0.1, 0.2],
        "patch_size": [4, 8],
        "stem_blocks": [2, 3],
    },
    "focus_dnn": {
        "num_layers": (1, 5),
        "units": (16, 256, 16),
        "activation": ["relu", "gelu"],
        "dropout": [0.0, 0.2, 0.4],
        "l2": [0.0, 1e-4, 1e-3],
        "optimizer": ["adam", "rmsprop"],
        "learning_rate": [1e-4, 5e-4, 1e-3],
    },
    "cnn_focus_hybrid": {
        "backbone_choice": ["custom_cnn", "mobilenetv2", "efficientnetb0"],
        "focus_units": [0, 32, 64, 128],
        "fusion_layers": [1, 2, 3],
        "fusion_units": [64, 128, 256, 512],
        "fusion_dropout": [0.0, 0.2, 0.4],
        "unfreeze_top": [0, 5, 10],
        "optimizer": ["adam", "rmsprop"],
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "num_blocks": (2, 5),
        "filters_base": (16, 64, 8),
        "kernel_size": [3, 5],
        "dropout": [0.0, 0.2, 0.4],
        "batchnorm": [True, False],
        "l2": [0.0, 1e-4, 1e-3],
    },
    "convnext": {
        "pooling": ["avg", "max"],
        "head_units": [0, 128, 256],
        "dropout": [0.0, 0.2, 0.4],
        "optimizer": ["adam", "rmsprop"],
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "width_mult": [0.75, 1.0],
        "depth_mult": [0.75, 1.0],
    },
    "swin": {
        "patch_size": [4, 8],
        "window_size": [7],
        "embed_dim": [64, 96, 128],
        "depth": [1, 2, 3],
        "num_heads": [2, 4],
        "mlp_dim": [128, 192, 256],
        "dropout": [0.0, 0.1, 0.2],
        "optimizer": ["adam", "rmsprop"],
        "learning_rate": [1e-4, 5e-4, 1e-3],
    },
}


def get_search_space(family: str) -> Dict[str, Any]:
    if family not in DEFAULT_SPACES:
        raise KeyError(f"Family {family} not found. Available: {list(DEFAULT_SPACES)}")
    return dict(DEFAULT_SPACES[family])


def _hp_choice(hp, name: str, values: List[Any], default: Any):
    if hasattr(hp, "Choice"):
        return hp.Choice(name, values=values, default=default)
    return default


def _hp_int(hp, name: str, min_value: int, max_value: int, default: int, step: int = 1):
    if hasattr(hp, "Int"):
        return hp.Int(name, min_value=min_value, max_value=max_value, step=step, default=default)
    return default


def _hp_float(hp, name: str, min_value: float, max_value: float, default: float, step: float | None = None):
    if hasattr(hp, "Float"):
        return hp.Float(name, min_value=min_value, max_value=max_value, step=step, default=default)
    return default


def build_hyperparameters(
    family: str,
    hp,
    backbone_choices: List[str] | None = None,
    light_mode: bool = False,
) -> Dict[str, Any]:
    """Create a family-specific hyperparameter dict from KerasTuner HP."""

    if family == "cnn":
        max_blocks = 4 if light_mode else 5
        return {
            "num_blocks": _hp_int(hp, "num_blocks", 2, max_blocks, default=3),
            "filters_base": _hp_int(hp, "filters_base", 16, 64, default=32, step=8),
            "kernel_size": _hp_choice(hp, "kernel_size", [3, 5], default=3),
            "dropout": _hp_float(hp, "dropout", 0.0, 0.5, default=0.2, step=0.1),
            "batchnorm": _hp_choice(hp, "batchnorm", [True, False], default=True),
            "dense_units": _hp_int(hp, "dense_units", 0, 256, default=0, step=64),
            "l2_reg": _hp_float(hp, "l2_reg", 0.0, 1e-3, default=0.0),
            "optimizer": _hp_choice(hp, "optimizer", ["adam", "rmsprop"], default="adam"),
            "learning_rate": _hp_float(hp, "learning_rate", 1e-4, 1e-3, default=5e-4),
        }

    if family == "cnn_attention":
        max_blocks = 4 if light_mode else 5
        return {
            "attention_type": _hp_choice(hp, "attention_type", ["se", "cbam", "none"], default="se"),
            "se_ratio": _hp_int(hp, "se_ratio", 4, 16, default=8, step=4),
            "attention_placement": _hp_choice(hp, "attention_placement", ["all", "last_two"], default="last_two"),
            "spatial_kernel": _hp_choice(hp, "spatial_kernel", [3, 7], default=7),
            "num_blocks": _hp_int(hp, "num_blocks", 2, max_blocks, default=3),
            "filters_base": _hp_int(hp, "filters_base", 16, 64, default=32, step=8),
            "kernel_size": _hp_choice(hp, "kernel_size", [3, 5], default=3),
            "dropout": _hp_float(hp, "dropout", 0.0, 0.5, default=0.2, step=0.1),
            "batchnorm": _hp_choice(hp, "batchnorm", [True, False], default=True),
            "dense_units": _hp_int(hp, "dense_units", 0, 256, default=0, step=64),
            "l2_reg": _hp_float(hp, "l2_reg", 0.0, 1e-3, default=0.0),
            "optimizer": _hp_choice(hp, "optimizer", ["adam", "rmsprop"], default="adam"),
            "learning_rate": _hp_float(hp, "learning_rate", 1e-4, 1e-3, default=5e-4),
        }

    if family == "transfer":
        if not backbone_choices:
            raise ValueError("transfer tuning requires backbone choices")
        choices = backbone_choices
        return {
            "backbone": _hp_choice(hp, "backbone", choices or [], default=(choices or [])[0]),
            "pooling": _hp_choice(hp, "pooling", ["avg", "max"], default="avg"),
            "head_units": _hp_int(hp, "head_units", 0, 512, default=0, step=128),
            "dropout": _hp_float(hp, "dropout", 0.0, 0.5, default=0.2, step=0.1),
            "lr": _hp_float(hp, "lr", 1e-5, 1e-3, default=1e-4),
            "label_smoothing": _hp_float(hp, "label_smoothing", 0.0, 0.1, default=0.0),
            "base_trainable_blocks": _hp_int(hp, "base_trainable_blocks", 0, 3, default=0, step=1),
        }

    if family == "vit":
        embed_choices = [64, 96, 128] if light_mode else [64, 128, 192, 256]
        depth_choices = [2, 4] if light_mode else [2, 4, 6, 8]
        return {
            "patch_size": _hp_choice(hp, "patch_size", [8, 16], default=16),
            "embed_dim": _hp_choice(hp, "embed_dim", embed_choices, default=embed_choices[0]),
            "num_heads": _hp_choice(hp, "num_heads", [2, 4, 8], default=4),
            "depth": _hp_choice(hp, "depth", depth_choices, default=depth_choices[0]),
            "mlp_dim": _hp_choice(hp, "mlp_dim", [128, 256, 384, 512], default=256),
            "dropout": _hp_float(hp, "dropout", 0.0, 0.3, default=0.1, step=0.1),
            "use_cls_token": _hp_choice(hp, "use_cls_token", [False, True], default=False),
        }

    if family == "hybrid_vit":
        embed_choices = [64, 96, 128] if light_mode else [64, 96, 128, 192]
        return {
            "stem_type": _hp_choice(hp, "stem_type", ["cnn", "mobilenetv2"], default="cnn"),
            "embed_dim": _hp_choice(hp, "embed_dim", embed_choices, default=embed_choices[0]),
            "transformer_depth": _hp_choice(hp, "transformer_depth", [1, 2, 3, 4], default=2),
            "num_heads": _hp_choice(hp, "num_heads", [2, 4], default=4),
            "mlp_dim": _hp_choice(hp, "mlp_dim", [128, 256, 384], default=256),
            "dropout": _hp_float(hp, "dropout", 0.0, 0.3, default=0.1, step=0.1),
            "patch_size": _hp_choice(hp, "patch_size", [4, 8], default=4),
            "stem_blocks": _hp_choice(hp, "stem_blocks", [2, 3], default=2),
        }

    if family == "focus_dnn":
        return {
            "num_layers": _hp_int(hp, "num_layers", 1, 5, default=2),
            "units": _hp_int(hp, "units", 16, 256, default=64, step=16),
            "activation": _hp_choice(hp, "activation", ["relu", "gelu"], default="relu"),
            "dropout": _hp_float(hp, "dropout", 0.0, 0.5, default=0.2, step=0.1),
            "l2": _hp_float(hp, "l2", 0.0, 1e-3, default=0.0),
            "optimizer": _hp_choice(hp, "optimizer", ["adam", "rmsprop"], default="adam"),
            "learning_rate": _hp_float(hp, "learning_rate", 1e-4, 1e-3, default=5e-4),
        }

    if family == "cnn_focus_hybrid":
        return {
            "backbone_choice": _hp_choice(
                hp,
                "backbone_choice",
                ["custom_cnn", "mobilenetv2", "efficientnetb0"],
                default="custom_cnn",
            ),
            "focus_units": _hp_int(hp, "focus_units", 0, 128, default=64, step=32),
            "fusion_layers": _hp_int(hp, "fusion_layers", 1, 3, default=1),
            "fusion_units": _hp_int(hp, "fusion_units", 32, 512, default=128, step=32),
            "fusion_dropout": _hp_float(hp, "fusion_dropout", 0.0, 0.5, default=0.2, step=0.1),
            "unfreeze_top": _hp_int(hp, "unfreeze_top", 0, 20, default=0, step=5),
            "optimizer": _hp_choice(hp, "optimizer", ["adam", "rmsprop"], default="adam"),
            "learning_rate": _hp_float(hp, "learning_rate", 1e-4, 1e-3, default=5e-4),
            "num_blocks": _hp_int(hp, "num_blocks", 2, 5, default=3),
            "filters_base": _hp_int(hp, "filters_base", 16, 64, default=32, step=8),
            "kernel_size": _hp_choice(hp, "kernel_size", [3, 5], default=3),
            "dropout": _hp_float(hp, "dropout", 0.0, 0.5, default=0.2, step=0.1),
            "batchnorm": _hp_choice(hp, "batchnorm", [True, False], default=True),
            "l2": _hp_float(hp, "l2", 0.0, 1e-3, default=0.0),
        }

    if family == "convnext":
        return {
            "pooling": _hp_choice(hp, "pooling", ["avg", "max"], default="avg"),
            "head_units": _hp_int(hp, "head_units", 0, 256, default=0, step=128),
            "dropout": _hp_float(hp, "dropout", 0.0, 0.5, default=0.2, step=0.1),
            "optimizer": _hp_choice(hp, "optimizer", ["adam", "rmsprop"], default="adam"),
            "learning_rate": _hp_float(hp, "learning_rate", 1e-4, 1e-3, default=5e-4),
            "width_mult": _hp_float(hp, "width_mult", 0.75, 1.0, default=1.0),
            "depth_mult": _hp_float(hp, "depth_mult", 0.75, 1.0, default=1.0),
        }

    if family == "swin":
        embed_choices = [64, 96, 128] if light_mode else [64, 96, 128]
        depth_choices = [1, 2] if light_mode else [1, 2, 3]
        return {
            "patch_size": _hp_choice(hp, "patch_size", [4, 8], default=4),
            "window_size": _hp_choice(hp, "window_size", [7], default=7),
            "embed_dim": _hp_choice(hp, "embed_dim", embed_choices, default=embed_choices[0]),
            "depth": _hp_choice(hp, "depth", depth_choices, default=depth_choices[0]),
            "num_heads": _hp_choice(hp, "num_heads", [2, 4], default=2),
            "mlp_dim": _hp_choice(hp, "mlp_dim", [128, 192, 256], default=192),
            "dropout": _hp_float(hp, "dropout", 0.0, 0.3, default=0.1, step=0.1),
            "optimizer": _hp_choice(hp, "optimizer", ["adam", "rmsprop"], default="adam"),
            "learning_rate": _hp_float(hp, "learning_rate", 1e-4, 1e-3, default=5e-4),
        }

    raise KeyError(f"Unsupported family: {family}")
