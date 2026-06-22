from __future__ import annotations

from typing import Callable, Dict

from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

ModelBuilder = Callable[..., object]

def _default_builders() -> Dict[str, Dict[str, ModelBuilder]]:
    # Local imports to avoid heavy TF imports during module import.
    from focus_binary.models.cnn_baseline import build_cnn_baseline, build_cnn_minimal
    from focus_binary.models.cnn_attention import build_cnn_attention
    from focus_binary.models.convnext import build_convnext
    from focus_binary.models.focus_dnn import build_focus_dnn
    from focus_binary.models.cnn_focus_hybrid import build_cnn_focus_hybrid
    from focus_binary.models.transfer import build_transfer_model
    from focus_binary.models.vit import build_vit
    from focus_binary.models.swin_tiny import build_swin_tiny
    from focus_binary.models.hybrid import build_hybrid, build_hybrid_vit

    registry: Dict[str, Dict[str, ModelBuilder]] = {}
    registry.setdefault("cnn", {})["tuned"] = build_cnn_baseline
    registry.setdefault("cnn", {})["minimal"] = build_cnn_minimal
    registry.setdefault("cnn", {})["default"] = build_cnn_baseline
    registry.setdefault("cnn_attention", {})["default"] = build_cnn_attention
    registry.setdefault("focus_dnn", {})["mlp_focus_measures"] = build_focus_dnn
    registry.setdefault("focus_dnn", {})["default"] = build_focus_dnn
    registry.setdefault("cnn_focus_hybrid", {})["cnn_plus_focus"] = build_cnn_focus_hybrid
    registry.setdefault("cnn_focus_hybrid", {})["default"] = build_cnn_focus_hybrid
    registry.setdefault("transfer", {})["default"] = build_transfer_model
    registry.setdefault("vit", {})["default"] = build_vit
    registry.setdefault("hybrid", {})["default"] = build_hybrid
    registry.setdefault("hybrid_vit", {})["default"] = build_hybrid_vit
    registry.setdefault("efficient_vit", {})["default"] = build_hybrid  # placeholder mapping for efficient ViT variant
    registry.setdefault("convnext", {})["default"] = build_convnext
    registry.setdefault("swin", {})["default"] = build_swin_tiny
    return registry


_MODEL_REGISTRY: Dict[str, Dict[str, ModelBuilder]] = _default_builders()


def register_model(family: str, name: str, builder_fn: ModelBuilder) -> None:
    logger.info("register model", extra={"family": family, "name": name})
    _MODEL_REGISTRY.setdefault(family, {})[name] = builder_fn


def get_builder(family: str, name: str) -> ModelBuilder:
    if family not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model family: {family}. Available: {list(_MODEL_REGISTRY)}")
    family_builders = _MODEL_REGISTRY[family]
    if name not in family_builders:
        raise KeyError(f"Unknown model name '{name}' for family '{family}'. Available: {list(family_builders)}")
    return family_builders[name]


def get_model(family: str, name: str = "default", **kwargs):
    builder = get_builder(family, name)
    return builder(**kwargs)


def available_models() -> Dict[str, Dict[str, ModelBuilder]]:
    return {family: dict(builders) for family, builders in _MODEL_REGISTRY.items()}
