"""
FLEUR backend for Audio-Flamingo-3 (NVIDIA)

Uses transformers with optional LoRA thinking-mode adapter.
"""

import os
import re

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

from .base import (
    FleurModel,
    make_fleur_prompt,
    parse_raw_score,
    build_rate2token,
    calculate_smoothed_score_torch,
)

MODEL_ID = "nvidia/audio-flamingo-3-hf"
THINK_ADAPTER_SUBDIR = "stage35"


def load_model(args) -> FleurModel:
    """Load Audio-Flamingo-3 model with optional thinking-mode LoRA adapter."""
    model_id = MODEL_ID
    print(f"Loading model from: {model_id}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
    )

    if getattr(args, 'use_think_mode', False):
        try:
            local_id = snapshot_download(model_id)
            non_lora_path = os.path.join(local_id, "think", "non_lora_trainables.bin")
            non_lora_trainables = torch.load(non_lora_path)
            model.load_state_dict(non_lora_trainables, strict=False)
            model = PeftModel.from_pretrained(model, local_id, subfolder="think")
            print("Loaded thinking mode LoRA adapter.")
        except Exception as e:
            print(f"Error loading thinking mode adapter: {e}")
            print("Proceeding without thinking mode.")

    rate2token = build_rate2token(processor)

    return FleurModel(
        model=model,
        processor=processor,
        rate2token=rate2token,
        backend="torch",
        _get_fleur_fn=get_fleur,
    )


def get_fleur(fm: FleurModel, caption: str, audio: str, **kwargs) -> tuple:
    """
    Compute FLEUR score for a single audio-caption pair using Audio-Flamingo-3.

    Returns:
        (raw_score, smoothed_score)
    """
    model = fm.model
    processor = fm.processor
    rate2token = fm.rate2token

    messages = make_fleur_prompt(audio, caption, audio_key="path")

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs.sequences[0, input_len:]

        output_text = processor.batch_decode(
            outputs.sequences[:, input_len:],
            skip_special_tokens=True
        )[0].strip()

        if outputs.scores:
            logits = torch.stack(outputs.scores, dim=0).squeeze(1)
            raw_score, score = calculate_smoothed_score_torch(
                output_text, logits, generated_ids, rate2token
            )
        else:
            raw_score = parse_raw_score(output_text)
            score = raw_score

    except Exception as e:
        print(f"Error in generation: {e}")
        raw_score = None
        score = None

    return raw_score, score
