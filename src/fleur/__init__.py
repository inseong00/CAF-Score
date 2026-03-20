"""
FLEUR — Unified API for audio caption evaluation.

Usage:
    from src.fleur import load_model, get_fleur

    fm = load_model('qwen3omni', args)
    raw_score, smoothed_score = get_fleur(fm, caption, audio_path)
"""

from .base import FleurModel


def load_model(model_name: str, args, **kwargs) -> FleurModel:
    """
    Load a FLEUR model by name.

    Args:
        model_name: One of 'audioflamingo3', 'qwen3omni', 'qwen25omni-3b', 'qwen25omni-7b'
        args: Arguments object (model-specific attributes are accessed via getattr)
        **kwargs: Forwarded to backend load_model (e.g., use_vllm=False for torch mode)
    """
    if model_name == 'audioflamingo3':
        from .af3 import load_model as _load
        return _load(args, **kwargs)

    elif model_name in ('qwen3omni', 'qwen25omni-3b', 'qwen25omni-7b'):
        if not hasattr(args, 'lalm_model'):
            args.lalm_model = model_name
        from .qwen import load_model as _load
        return _load(args, **kwargs)

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported: audioflamingo3, qwen3omni, qwen25omni-3b, qwen25omni-7b"
        )


def get_fleur(fm: FleurModel, caption: str, audio: str, **kwargs) -> tuple:
    """
    Compute FLEUR score using a loaded model.

    Returns:
        (raw_score, smoothed_score)
    """
    return fm._get_fleur_fn(fm, caption, audio, **kwargs)
