"""
FLEUR (Flexible Evaluation Using Language Models) for Qwen3-Omni

This script implements FLEUR metric evaluation for audio captions using Qwen3-Omni models.
FLEUR uses an LLM to evaluate caption quality on a 0.0-1.0 scale with score smoothing
based on token probability distributions.

Reference: FLEUR: An Explainable Reference-Free Evaluation Metric for Image Captioning
           Using a Large Multimodal Model (ACL 2024)
           https://github.com/Yebin46/FLEUR
"""

import os
import re
import time
import math
import tempfile
import argparse

import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor, AutoTokenizer
from qwen_omni_utils import process_mm_info


# Default model path - can be overridden via environment variable or argument
DEFAULT_MODEL_PATH = os.environ.get("QWEN3_OMNI_MODEL_PATH", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
DEFAULT_THINKING_MODEL_PATH = os.environ.get("QWEN3_OMNI_THINKING_MODEL_PATH", "Qwen/Qwen3-Omni-30B-A3B-Thinking")


def make_fleur_prompt(audio_path: str, pred_caption: str) -> list:
    """
    Create FLEUR evaluation prompt for Qwen3-Omni

    Args:
        audio_path: Path to the audio file
        pred_caption: The caption to evaluate

    Returns:
        Messages list for Qwen3-Omni
    """
    # FLEUR instruction: evaluate caption quality on 0.0-1.0 scale
    eval_prompt = f"""Your task is to evaluate and rate the caption on a scale of 0.0 to 1.0 based on the given Grading Criteria. (Print Real Number Score ONLY)

Grading Criteria:

0.0: The caption does not describe the audio at all.
1.0: The caption accurately and clearly describes the audio.

Caption: {pred_caption}

Score(Choose a rating from 0.0 to 1.0):"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": eval_prompt},
            ],
        }
    ]

    return messages


def calculate_smoothed_score(
    output_text: str,
    logprobs_list: list,
    token_ids: list,
    rate2token: dict,
    tokenizer
) -> tuple:
    """
    Calculate score smoothing based on token probability distributions.

    This implements the FLEUR score smoothing algorithm which uses the probability
    distribution over digit tokens (0-9) to compute a weighted average score,
    making the evaluation more robust to minor variations in model output.

    Args:
        output_text: The raw output text from the model
        logprobs_list: List of LogprobsOutput objects for each generated token
        token_ids: List of generated token IDs
        rate2token: Dictionary mapping digits (0-9) to their token IDs
        tokenizer: The tokenizer for decoding

    Returns:
        Smoothed FLEUR score (0.0 to 1.0)
    """
    # Parse the score from output text
    # Remove non-numeric characters except dots
    dotsnumbersdots = re.sub(r'[^\d\.]', '', output_text)
    # Remove leading dots
    numbersdots = re.sub(r'^\.+', '', dotsnumbersdots)
    # Remove trailing dots
    numbers = re.sub(r'\.+$', '', numbersdots)

    if not numbers:
        return None, None

    try:
        score_check = float(numbers)
    except ValueError:
        return None, None

    # Validate score range
    if score_check < 0 or score_check > 1:
        return score_check, None

    # Convert token_ids to tensor-like list for searching
    token_ids_list = list(token_ids)

    if score_check < 1.0:
        # Find the first decimal digit position in the score string
        # e.g., "0.7" -> find position of '7' which is index 2 (after "0.")
        score_str = str(score_check)
        num_index_in_score = score_str.index('.') + 1
        find_num = int(score_str[num_index_in_score])

        # Find the token position in the output
        target_token = rate2token[find_num]
        matching_indices = [i for i, tid in enumerate(token_ids_list) if tid == target_token]

        if not matching_indices:
            # Fallback to raw score if token not found
            return score_check, score_check

        # Handle duplicates: for 0.0 select second 0 (after dot), for others select first
        if find_num == 0 and len(matching_indices) > 1:
            num_index_in_token = matching_indices[1]
        else:
            num_index_in_token = matching_indices[0]

        # Get probability distribution at this position
        if num_index_in_token >= len(logprobs_list):
            return score_check, score_check

        logprobs_at_pos = logprobs_list[num_index_in_token]

        # Calculate weighted score using probabilities for digits 0-9
        # Each digit d contributes: P(d) * d * 0.1 (for first decimal place)
        score = 0.0
        for rate, token in rate2token.items():
            if token in logprobs_at_pos:
                prob = math.exp(logprobs_at_pos[token].logprob)
                score += prob * rate * 0.1
            # If token not in top logprobs, its probability is negligible

        # Handle second decimal place if present (e.g., 0.75)
        if len(score_str) > 3:
            num2_index_in_score = score_str.index('.') + 2
            if num2_index_in_score < len(score_str):
                find_num2 = int(score_str[num2_index_in_score])

                target_token2 = rate2token[find_num2]
                matching_indices2 = [i for i, tid in enumerate(token_ids_list) if tid == target_token2]

                if matching_indices2:
                    # For second decimal, choose the second occurrence
                    if len(matching_indices2) > 1:
                        num2_index_in_token = matching_indices2[1]
                    else:
                        num2_index_in_token = matching_indices2[0]

                    if num2_index_in_token < len(logprobs_list):
                        logprobs_at_pos2 = logprobs_list[num2_index_in_token]

                        # Add contribution from second decimal place
                        for rate, token in rate2token.items():
                            if token in logprobs_at_pos2:
                                prob = math.exp(logprobs_at_pos2[token].logprob)
                                score += prob * rate * 0.01

    else:
        # score_check == 1.0 case
        # Find position of '1' token
        target_token = rate2token[1]
        matching_indices = [i for i, tid in enumerate(token_ids_list) if tid == target_token]

        if not matching_indices:
            return score_check, score_check

        num_index_in_token = matching_indices[0]

        if num_index_in_token >= len(logprobs_list):
            return score_check, score_check

        logprobs_at_pos = logprobs_list[num_index_in_token]

        # For 1.0, calculate as: 0.9 * P(0) + 1.0 * P(1)
        score = 0.0
        if rate2token[0] in logprobs_at_pos:
            prob_0 = math.exp(logprobs_at_pos[rate2token[0]].logprob)
            score += 0.9 * prob_0
        if rate2token[1] in logprobs_at_pos:
            prob_1 = math.exp(logprobs_at_pos[rate2token[1]].logprob)
            score += prob_1

    return score_check, score


def load_model(args):
    """
    Load Qwen3-Omni model with specified configurations.

    Args:
        args: Command line arguments containing:
            - model_path: Path to the Qwen3-Omni model (optional)
            - use_think_mode: Whether to use thinking model variant
            - tensor_parallel_size: GPU parallelism
            - gpu_memory_utilization: GPU memory utilization

    Returns:
        Initialized LLM model, processor, tokenizer, sampling params, and rate2token mapping
    """

    os.environ['VLLM_USE_V1'] = '0'

    # Set model path
    model_path = getattr(args, 'model_path', None) or DEFAULT_MODEL_PATH
    if args.use_think_mode:
        model_path = getattr(args, 'thinking_model_path', None) or DEFAULT_THINKING_MODEL_PATH
        print("Using thinking model variant")

    print(f"Loading model from: {model_path}")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={'image': 0, 'video': 0, 'audio': 1},
        max_num_seqs=8,
        max_model_len=16384,
        seed=1234,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for evaluation
        max_tokens=512,
        logprobs=20,  # Get top 20 logprobs per token (includes all digits)
    )

    rate2token = {}
    for digit in range(10):
        encoded = tokenizer.encode(str(digit), add_special_tokens=False)
        rate2token[digit] = encoded[-1]  # Use the last token ID

    return llm, processor, tokenizer, sampling_params, rate2token


def get_fleur(llm, processor, tokenizer, sampling_params, rate2token, caption, audio):
    """
    Compute FLEUR score for a single audio-caption pair.

    Args:
        llm: vLLM model instance
        processor: Qwen3OmniMoeProcessor
        tokenizer: AutoTokenizer
        sampling_params: SamplingParams for generation
        rate2token: Dictionary mapping digits (0-9) to their token IDs
        caption: The caption to evaluate
        audio: Path to the audio file

    Returns:
        raw_score: The raw score parsed from model output
        score: The smoothed FLEUR score
    """

    # Create FLEUR evaluation prompt
    messages = make_fleur_prompt(audio, caption)

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Process audio
    audios, _, _ = process_mm_info(messages, use_audio_in_video=False)

    # Prepare inputs
    inputs = {
        'prompt': text,
        'multi_modal_data': {},
        'mm_processor_kwargs': {
            'use_audio_in_video': False,
        },
    }

    if audios is not None:
        inputs['multi_modal_data']['audio'] = audios

    try:
        # Generate with logprobs
        outputs = llm.generate([inputs], sampling_params=sampling_params)
        output_obj = outputs[0].outputs[0]
        output_text = output_obj.text.strip()

        # Handle thinking mode output
        if "</think>" in output_text:
            output_text = output_text.split("</think>")[-1].strip()

        # Get token IDs and logprobs for smoothing
        token_ids = output_obj.token_ids
        logprobs_list = output_obj.logprobs  # List of dict {token_id: Logprob}

        # Calculate smoothed score
        raw_score, score = calculate_smoothed_score(
            output_text,
            logprobs_list,
            token_ids,
            rate2token,
            tokenizer
        )

    except Exception as e:
        print(f"Error in generation: {e}")
        score = None
        raw_score = None

    return raw_score, score


def main(args):
    # Load model
    llm, processor, tokenizer, sampling_params, rate2token = load_model(args)

    # Compute FLEUR score
    raw_score, fleur_score = get_fleur(
        llm,
        processor,
        tokenizer,
        sampling_params,
        rate2token,
        args.pred_caption,
        args.audio_path
    )

    print(f"Final FLEUR Score: {fleur_score}, Raw Score: {raw_score}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute FLEUR scores for audio captions using Qwen3-Omni"
    )
    parser.add_argument(
        '--audio_path',
        type=str,
        required=True,
        help='Audio file path to evaluate'
    )
    parser.add_argument(
        '--pred_caption',
        type=str,
        required=True,
        help='Caption to evaluate against the audio'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to Qwen3-Omni model (default: from environment or HuggingFace)'
    )
    parser.add_argument(
        '--thinking_model_path',
        type=str,
        default=None,
        help='Path to Qwen3-Omni thinking model (default: from environment or HuggingFace)'
    )
    parser.add_argument(
        '--tensor_parallel_size',
        type=int,
        default=2,
        help='Tensor parallel size for vLLM'
    )
    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.95,
        help='GPU memory utilization for vLLM'
    )
    parser.add_argument(
        '--use_think_mode',
        action='store_true',
        default=False,
        help='Whether to use thinking model variant'
    )

    args = parser.parse_args()

    print('Computing FLEUR scores with Qwen3-Omni...')
    main(args)
