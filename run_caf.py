#!/usr/bin/env python3
"""
Compute CAF-Score for a single audio-caption pair.

This script takes an audio file and a caption as input and outputs the CAF-Score,
which combines CLAP similarity and FLEUR score.

"""

import os
import argparse

from src.clap import load_clap


def compute_caf_score(
    audio_path: str,
    caption: str,
    clap_model_name: str,
    lalm_model_name: str,
    alpha: float = 0.8,
    use_slide_window: bool = False,
    pooling: str = 'max',
    use_think_mode: bool = False,
    tensor_parallel_size: int = 2,
    gpu_memory_utilization: float = 0.95,
    verbose: bool = True
) -> dict:
    """
    Compute CAF-Score for a single audio-caption pair.

    Args:
        audio_path: Path to the audio file
        caption: The caption to evaluate
        clap_model_name: CLAP model to use ('msclap', 'laionclap', 'mgaclap', 'm2dclap')
        lalm_model_name: LALM model to use ('audioflamingo3', 'qwen3omni')
        alpha: Weight for CLAP score (1-alpha for FLEUR). Default: 0.5
        use_slide_window: Use sliding window for long audio in CLAP
        pooling: Pooling method for sliding window ('max' or 'mean')
        use_think_mode: Use thinking mode for LALM
        tensor_parallel_size: Tensor parallel size for Qwen3-Omni
        gpu_memory_utilization: GPU memory utilization for Qwen3-Omni
        verbose: Print progress information

    Returns:
        Dictionary containing:
            - clap_score: CLAP similarity score
            - fleur_score: FLEUR score from LALM
            - caf_score: Combined CAF-Score
            - raw_fleur_score: Raw FLEUR score before smoothing
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load CLAP model
    if verbose:
        print(f"Loading CLAP model: {clap_model_name}")
    clap_model = load_clap(clap_model_name)

    # Compute CLAP similarity
    if verbose:
        print("Computing CLAP similarity...")
    clap_similarity = clap_model.get_similarity(
        audio_path,
        caption,
        use_sliding_window=use_slide_window,
        pooling=pooling
    )
    clap_score = max(float(clap_similarity[0, 0].cpu()), 0.0)

    # Create args object for LALM model loading
    class Args:
        pass
    args = Args()
    args.use_think_mode = use_think_mode
    args.tensor_parallel_size = tensor_parallel_size
    args.gpu_memory_utilization = gpu_memory_utilization

    # Load LALM model and compute FLEUR score
    if verbose:
        print(f"Loading LALM model: {lalm_model_name}")

    if lalm_model_name == 'qwen3omni':
        from src.qwen3_fleur import load_model, get_fleur
        llm, processor, tokenizer, sampling_params, rate2token = load_model(args)
        if verbose:
            print("Computing FLEUR score...")
        raw_fleur_score, fleur_score = get_fleur(
            llm, processor, tokenizer, sampling_params, rate2token, caption, audio_path
        )
    elif lalm_model_name == 'audioflamingo3':
        from src.af3_fleur import load_model, get_fleur
        model, processor, rate2token = load_model(args)
        if verbose:
            print("Computing FLEUR score...")
        raw_fleur_score, fleur_score = get_fleur(
            model, processor, rate2token, caption, audio_path
        )
    else:
        raise ValueError(f"Unknown LALM model: {lalm_model_name}")

    # Handle None values
    fleur_score = fleur_score if fleur_score is not None else 0.0
    raw_fleur_score = raw_fleur_score if raw_fleur_score is not None else 0.0

    # Compute CAF-Score
    caf_score = alpha * clap_score + (1 - alpha) * fleur_score

    result = {
        'audio_path': audio_path,
        'caption': caption,
        'clap_model': clap_model_name,
        'lalm_model': lalm_model_name,
        'alpha': alpha,
        'clap_score': clap_score,
        'fleur_score': fleur_score,
        'raw_fleur_score': raw_fleur_score,
        'caf_score': caf_score
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute CAF-Score for a single audio-caption pair",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default models
    python run_caf.py --audio_path audio.wav --caption "A dog barking"

    # With specific models and alpha
    python run_caf.py --audio_path audio.wav --caption "Birds chirping" \\
        --clap_model msclap --lalm_model qwen3omni --alpha 0.6

    # With sliding window for long audio
    python run_caf.py --audio_path long_audio.wav --caption "Music playing" \\
        --use_slide_window --pooling max
        """
    )
    parser.add_argument(
        '--audio_path',
        type=str,
        required=True,
        help='Path to the audio file'
    )
    parser.add_argument(
        '--caption',
        type=str,
        required=True,
        help='Caption to evaluate against the audio'
    )
    parser.add_argument(
        '--clap_model',
        type=str,
        required=True,
        choices=['msclap', 'laionclap', 'mgaclap', 'm2dclap'],
        help='CLAP model to use (default: laionclap)'
    )
    parser.add_argument(
        '--lalm_model',
        type=str,
        required=True,
        choices=['audioflamingo3', 'qwen3omni'],
        help='LALM model for FLEUR scoring (default: audioflamingo3)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.8,
        help='Weight for CLAP score (1-alpha for FLEUR). Default: 0.8'
    )
    parser.add_argument(
        '--use_slide_window',
        action='store_true',
        default=False,
        help='Use sliding window for long audio in CLAP'
    )
    parser.add_argument(
        '--pooling',
        type=str,
        default='max',
        choices=['mean', 'max'],
        help='Pooling method for sliding window (default: max)'
    )
    parser.add_argument(
        '--use_think_mode',
        action='store_true',
        default=False,
        help='Use thinking mode for LALM'
    )
    parser.add_argument(
        '--tensor_parallel_size',
        type=int,
        default=2,
        help='Tensor parallel size for Qwen3-Omni (default: 2)'
    )
    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.95,
        help='GPU memory utilization for Qwen3-Omni (default: 0.95)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        default=False,
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    # Compute CAF-Score
    result = compute_caf_score(
        audio_path=args.audio_path,
        caption=args.caption,
        clap_model_name=args.clap_model,
        lalm_model_name=args.lalm_model,
        alpha=args.alpha,
        use_slide_window=args.use_slide_window,
        pooling=args.pooling,
        use_think_mode=args.use_think_mode,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        verbose=not args.quiet
    )

    # Print results
    print("\n" + "="*60)
    print("CAF-Score Results")
    print("="*60)
    print(f"Audio: {result['audio_path']}")
    print(f"Caption: {result['caption']}")
    print("-"*60)
    print(f"CLAP Model: {result['clap_model']}")
    print(f"LALM Model: {result['lalm_model']}")
    print("-"*60)
    print(f"CLAP Score: {result['clap_score']:.4f}")
    print(f"FLEUR Score: {result['fleur_score']:.4f}")
    print("-"*60)
    print(f"CAF-Score: {result['caf_score']:.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
