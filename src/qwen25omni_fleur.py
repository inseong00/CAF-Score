"""
FLEUR (Flexible Evaluation Using Language Models) for Qwen2.5-Omni

This script implements FLEUR metric evaluation for audio captions using Qwen2.5-Omni models.
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
from transformers import Qwen2_5OmniProcessor, AutoTokenizer
from qwen_omni_utils import process_mm_info


# Model paths
MODEL_PATHS = {
    'qwen25omni_7b': "/home/inseong6474/NC-AI/models/Qwen2.5-Omni-7B",
    'qwen25omni_3b': "/home/inseong6474/NC-AI/models/Qwen2.5-Omni-3B",
}


def make_fleur_prompt(audio_path: str, pred_caption: str) -> list:
    """
    Create FLEUR evaluation prompt for Qwen2.5-Omni

    Args:
        audio_path: Path to the audio file
        pred_caption: The caption to evaluate

    Returns:
        Messages list for Qwen2.5-Omni
    """
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

def make_reverse_fleur_prompt(audio_path: str, pred_caption: str) -> list:
    """
    Create reverse FLEUR evaluation prompt for Qwen2.5-Omni
    """
    eval_prompt = f"""Your task is to evaluate and rate the audio on a scale of 0.0 to 1.0 based on the given Grading Criteria. (Print Real Number Score ONLY)

Grading Criteria:

0.0: The audio does not describe the caption at all.
1.0: The audio accurately and clearly describes the caption.

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
    dotsnumbersdots = re.sub(r'[^\d\.]', '', output_text)
    numbersdots = re.sub(r'^\.+', '', dotsnumbersdots)
    numbers = re.sub(r'\.+$', '', numbersdots)

    if not numbers:
        return None, None

    try:
        score_check = float(numbers)
    except ValueError:
        return None, None

    if score_check < 0 or score_check > 1:
        return score_check, None

    token_ids_list = list(token_ids)

    if score_check < 1.0:
        score_str = str(score_check)
        num_index_in_score = score_str.index('.') + 1
        find_num = int(score_str[num_index_in_score])

        target_token = rate2token[find_num]
        matching_indices = [i for i, tid in enumerate(token_ids_list) if tid == target_token]

        if not matching_indices:
            return score_check, score_check

        if find_num == 0 and len(matching_indices) > 1:
            num_index_in_token = matching_indices[1]
        else:
            num_index_in_token = matching_indices[0]

        if num_index_in_token >= len(logprobs_list):
            return score_check, score_check

        logprobs_at_pos = logprobs_list[num_index_in_token]

        score = 0.0
        entropy1 = 0.0
        digit_probs1 = []
        for rate, token in rate2token.items():
            if token in logprobs_at_pos:
                prob = math.exp(logprobs_at_pos[token].logprob)
                score += prob * rate * 0.1
                digit_probs1.append(prob)
            else:
                digit_probs1.append(0.0)
        sum_probs1 = sum(digit_probs1)
        if sum_probs1 > 0:
            for p in digit_probs1:
                if p > 0:
                    entropy1 -= (p / sum_probs1) * math.log10(p / sum_probs1)

        entropy2 = 0.0
        digits_probs2 = []

        if len(score_str) > 3:
            num2_index_in_score = score_str.index('.') + 2
            if num2_index_in_score < len(score_str):
                find_num2 = int(score_str[num2_index_in_score])

                target_token2 = rate2token[find_num2]
                matching_indices2 = [i for i, tid in enumerate(token_ids_list) if tid == target_token2]

                if matching_indices2:
                    if len(matching_indices2) > 1:
                        num2_index_in_token = matching_indices2[1]
                    else:
                        num2_index_in_token = matching_indices2[0]

                    if num2_index_in_token < len(logprobs_list):
                        logprobs_at_pos2 = logprobs_list[num2_index_in_token]

                        for rate, token in rate2token.items():
                            if token in logprobs_at_pos2:
                                prob = math.exp(logprobs_at_pos2[token].logprob)
                                score += prob * rate * 0.01
                                digits_probs2.append(prob)
        sum_probs2 = sum(digits_probs2)
        if sum_probs2 > 0:
            for p in digits_probs2:
                if p > 0:
                    entropy2 -= (p / sum_probs2) * math.log10(p / sum_probs2)

    else:
        # score_check == 1.0 case
        target_token = rate2token[1]
        matching_indices = [i for i, tid in enumerate(token_ids_list) if tid == target_token]

        if not matching_indices:
            return score_check, score_check

        num_index_in_token = matching_indices[0]

        if num_index_in_token >= len(logprobs_list):
            return score_check, score_check

        logprobs_at_pos = logprobs_list[num_index_in_token]

        score = 0.0
        entropy1 = 0.0
        entropy2 = 0.0
        digit_probs = []
        if rate2token[0] in logprobs_at_pos:
            prob_0 = math.exp(logprobs_at_pos[rate2token[0]].logprob)
            score += 0.9 * prob_0
            digit_probs.append(prob_0)
        if rate2token[1] in logprobs_at_pos:
            prob_1 = math.exp(logprobs_at_pos[rate2token[1]].logprob)
            score += prob_1
            digit_probs.append(prob_1)

        sum_probs = sum(digit_probs)
        if sum_probs > 0:
            for p in digit_probs:
                if p > 0:
                    entropy1 -= (p / sum_probs) * math.log10(p / sum_probs)

    return score_check, score, entropy1, entropy2

def load_model(args):
    """
    Load Qwen2.5-Omni model with specified configurations.

    Args:
        args: Command line arguments

    Returns:
        Initialized LLM model, processor, tokenizer, sampling_params, rate2token
    """

    os.environ['VLLM_USE_V1'] = '0'

    model_path = MODEL_PATHS[args.lalm_model]

    print(f"Loading Qwen2.5-Omni model from: {model_path}")
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

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        logprobs=20,
    )

    rate2token = {}
    for digit in range(10):
        encoded = tokenizer.encode(str(digit), add_special_tokens=False)
        rate2token[digit] = encoded[-1]

    return llm, processor, tokenizer, sampling_params, rate2token


def get_fleur(llm, processor, tokenizer, sampling_params, rate2token, caption, audio, is_reverse=False):
    """
    Compute FLEUR score for a single audio-caption pair.

    Args:
        llm: vLLM model
        processor: Qwen2.5-Omni processor
        tokenizer: Tokenizer
        sampling_params: Sampling parameters
        rate2token: Digit to token ID mapping
        caption: Caption to evaluate
        audio: Path to audio file
        is_reverse: Whether to use reverse evaluation

    Returns:
        Tuple of (raw_score, smoothed_score, entropy1, entropy2)
    """

    if not is_reverse:
        messages = make_fleur_prompt(audio, caption)
    else:
        messages = make_reverse_fleur_prompt(audio, caption)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    audios, _, _ = process_mm_info(messages, use_audio_in_video=False)

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
        outputs = llm.generate([inputs], sampling_params=sampling_params)
        output_obj = outputs[0].outputs[0]
        output_text = output_obj.text.strip()

        if "</think>" in output_text:
            output_text = output_text.split("</think>")[-1].strip()

        print(f"Audio path: {audio}")
        print(f"Predicted caption: {caption}")
        print(f"FLEUR output raw text: {output_text}")

        token_ids = output_obj.token_ids
        logprobs_list = output_obj.logprobs

        raw_score, score, entropy1, entropy2 = calculate_smoothed_score(
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
        entropy1 = None
        entropy2 = None

    return raw_score, score, entropy1, entropy2

def main(args):
    llm, processor, tokenizer, sampling_params, rate2token = load_model(args)

    raw_score, fleur_score, entropy1, entropy2 = get_fleur(
        llm,
        processor,
        tokenizer,
        sampling_params,
        rate2token,
        args.pred_caption,
        args.audio_path,
        is_reverse=args.is_reverse
    )

    print(f"Final FLEUR Score: {fleur_score}, Raw Score: {raw_score}, Entropy1: {entropy1}, Entropy2: {entropy2}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute FLEUR scores for audio captions using Qwen2.5-Omni"
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
        help='Caption to evaluate'
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
        '--is_reverse',
        action='store_true',
        default=False,
        help='Whether to use reverse FLEUR evaluation'
    )

    args = parser.parse_args()

    print('Computing FLEUR scores with Qwen2.5-Omni...')
    main(args)
