"""
FLEUR (Flexible Evaluation Using Language Models) — Shared Utilities

Common components for FLEUR metric evaluation across different audio language models.
Includes prompt generation, score parsing, token probability smoothing, and the
unified FleurModel container.

Reference: FLEUR: An Explainable Reference-Free Evaluation Metric for Image Captioning
           Using a Large Multimodal Model (ACL 2024)
           https://github.com/Yebin46/FLEUR
"""

import re
import math
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

import torch


@dataclass
class FleurModel:
    """Unified container for all FLEUR model backends."""
    model: Any
    processor: Any
    rate2token: dict
    tokenizer: Any = None
    sampling_params: Any = None
    backend: str = "torch"  # "torch" or "vllm"
    _get_fleur_fn: Callable = field(default=None, repr=False)


def make_fleur_prompt(audio_path: str, pred_caption: str, audio_key: str = "audio") -> list:
    """
    Create FLEUR evaluation prompt.

    Args:
        audio_path: Path to the audio file
        pred_caption: The caption to evaluate
        audio_key: Key for audio in message content ("path" for AF3, "audio" for Qwen)
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
                {"type": "audio", audio_key: audio_path},
                {"type": "text", "text": eval_prompt},
            ],
        }
    ]
    return messages


def parse_raw_score(output_text: str) -> Optional[float]:
    """Parse a numeric score from model output text. Returns None if unparseable."""
    dotsnumbersdots = re.sub(r'[^\d\.]', '', output_text)
    numbersdots = re.sub(r'^\.+', '', dotsnumbersdots)
    numbers = re.sub(r'\.+$', '', numbersdots)
    if not numbers:
        return None
    try:
        return float(numbers)
    except ValueError:
        return None


def build_rate2token(tokenizer_or_processor) -> dict:
    """Build digit (0-9) to token ID mapping for score smoothing."""
    tokenizer = getattr(tokenizer_or_processor, 'tokenizer', tokenizer_or_processor)
    rate2token = {}
    for digit in range(10):
        encoded = tokenizer.encode(str(digit), add_special_tokens=False)
        rate2token[digit] = encoded[-1]
    return rate2token


# ---------------------------------------------------------------------------
# Score smoothing — torch backend (AF3, Qwen3-single)
# ---------------------------------------------------------------------------

def _find_token_position(token_ids_list, target_token, is_zero_digit=False):
    """Find the position of a target token in the token ID list."""
    matching_indices = [i for i, tid in enumerate(token_ids_list) if tid == target_token]
    if not matching_indices:
        return None
    # For digit 0, select second occurrence (after the dot in "0.X")
    if is_zero_digit and len(matching_indices) > 1:
        return matching_indices[1]
    return matching_indices[0]


def calculate_smoothed_score_torch(
    output_text: str,
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    rate2token: dict,
) -> tuple:
    """
    Calculate smoothed FLEUR score using torch logits (softmax probabilities).
    Used by torch-based backends (AF3, Qwen3-single).

    Returns:
        (raw_score, smoothed_score) — either may be None on failure
    """
    score_check = parse_raw_score(output_text)
    if score_check is None:
        return None, None
    if score_check < 0 or score_check > 1:
        return score_check, None

    token_ids_list = generated_ids.tolist()
    probs = torch.softmax(logits, dim=-1)

    if score_check < 1.0:
        score_str = str(score_check)
        find_num = int(score_str[score_str.index('.') + 1])

        pos = _find_token_position(token_ids_list, rate2token[find_num], is_zero_digit=(find_num == 0))
        if pos is None or pos >= len(probs):
            return score_check, score_check

        probs_at_pos = probs[pos]
        score = 0.0
        for rate, token in rate2token.items():
            score += probs_at_pos[token].item() * rate * 0.1

        # Second decimal place
        if len(score_str) > 3:
            idx2 = score_str.index('.') + 2
            if idx2 < len(score_str):
                find_num2 = int(score_str[idx2])
                target_token2 = rate2token[find_num2]
                matching_indices2 = [i for i, tid in enumerate(token_ids_list) if tid == target_token2]
                if matching_indices2:
                    pos2 = matching_indices2[1] if len(matching_indices2) > 1 else matching_indices2[0]
                    if pos2 < len(probs):
                        probs_at_pos2 = probs[pos2]
                        for rate, token in rate2token.items():
                            score += probs_at_pos2[token].item() * rate * 0.01
    else:
        # score == 1.0
        pos = _find_token_position(token_ids_list, rate2token[1])
        if pos is None or pos >= len(probs):
            return score_check, score_check

        probs_at_pos = probs[pos]
        score = 0.9 * probs_at_pos[rate2token[0]].item() + probs_at_pos[rate2token[1]].item()

    return score_check, score


# ---------------------------------------------------------------------------
# Score smoothing — vLLM backend (Qwen3-vLLM, Qwen2.5)
# ---------------------------------------------------------------------------

def calculate_smoothed_score_vllm(
    output_text: str,
    logprobs_list: list,
    token_ids: list,
    rate2token: dict,
) -> tuple:
    """
    Calculate smoothed FLEUR score using vLLM logprobs.
    Used by vLLM-based backends (Qwen3, Qwen2.5).

    Returns:
        (raw_score, smoothed_score) — either may be None on failure
    """
    score_check = parse_raw_score(output_text)
    if score_check is None:
        return None, None
    if score_check < 0 or score_check > 1:
        return score_check, None

    token_ids_list = list(token_ids)

    if score_check < 1.0:
        score_str = str(score_check)
        find_num = int(score_str[score_str.index('.') + 1])

        pos = _find_token_position(token_ids_list, rate2token[find_num], is_zero_digit=(find_num == 0))
        if pos is None or pos >= len(logprobs_list):
            return score_check, score_check

        logprobs_at_pos = logprobs_list[pos]
        score = 0.0
        for rate, token in rate2token.items():
            if token in logprobs_at_pos:
                prob = math.exp(logprobs_at_pos[token].logprob)
                score += prob * rate * 0.1

        # Second decimal place
        if len(score_str) > 3:
            idx2 = score_str.index('.') + 2
            if idx2 < len(score_str):
                find_num2 = int(score_str[idx2])
                matching_indices2 = [i for i, tid in enumerate(token_ids_list) if tid == rate2token[find_num2]]
                if matching_indices2:
                    pos2 = matching_indices2[1] if len(matching_indices2) > 1 else matching_indices2[0]
                    if pos2 < len(logprobs_list):
                        logprobs_at_pos2 = logprobs_list[pos2]
                        for rate, token in rate2token.items():
                            if token in logprobs_at_pos2:
                                prob = math.exp(logprobs_at_pos2[token].logprob)
                                score += prob * rate * 0.01
    else:
        # score == 1.0
        pos = _find_token_position(token_ids_list, rate2token[1])
        if pos is None or pos >= len(logprobs_list):
            return score_check, score_check

        logprobs_at_pos = logprobs_list[pos]
        score = 0.0
        if rate2token[0] in logprobs_at_pos:
            score += 0.9 * math.exp(logprobs_at_pos[rate2token[0]].logprob)
        if rate2token[1] in logprobs_at_pos:
            score += math.exp(logprobs_at_pos[rate2token[1]].logprob)

    return score_check, score


