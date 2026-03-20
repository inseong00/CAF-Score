"""
FLEUR backend for Qwen-Omni models (Qwen3-Omni, Qwen2.5-Omni)

Supports both vLLM (batch) and transformers (single) inference modes.
Model-specific classes are resolved by model_name at load time.
"""

import os

import torch

from .base import (
    FleurModel,
    make_fleur_prompt,
    parse_raw_score,
    build_rate2token,
    calculate_smoothed_score_torch,
    calculate_smoothed_score_vllm,
)

# ── Model configs ─────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    'qwen3omni': {
        'model_path': os.environ.get("QWEN3_OMNI_MODEL_PATH", "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
        'thinking_model_path': os.environ.get("QWEN3_OMNI_THINKING_MODEL_PATH", "Qwen/Qwen3-Omni-30B-A3B-Thinking"),
        'processor_cls': 'Qwen3OmniMoeProcessor',
        'model_cls': 'Qwen3OmniMoeForConditionalGeneration',
    },
    'qwen25omni-3b': {
        'model_path': os.environ.get("QWEN25_OMNI_3B_MODEL_PATH", "Qwen/Qwen2.5-Omni-3B"),
        'processor_cls': 'Qwen2_5OmniProcessor',
        'model_cls': 'Qwen2_5OmniForConditionalGeneration',
    },
    'qwen25omni-7b': {
        'model_path': os.environ.get("QWEN25_OMNI_7B_MODEL_PATH", "Qwen/Qwen2.5-Omni-7B"),
        'processor_cls': 'Qwen2_5OmniProcessor',
        'model_cls': 'Qwen2_5OmniForConditionalGeneration',
    },
}


def _resolve_config(args) -> dict:
    """Resolve the model config dict from args.lalm_model."""
    model_name = getattr(args, 'lalm_model', 'qwen3omni')
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown Qwen model: {model_name}. Supported: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]


def _resolve_model_path(args, cfg: dict) -> str:
    """Determine the model path, respecting think-mode overrides."""
    model_path = getattr(args, 'model_path', None) or cfg['model_path']
    if getattr(args, 'use_think_mode', False) and 'thinking_model_path' in cfg:
        model_path = getattr(args, 'thinking_model_path', None) or cfg['thinking_model_path']
        print("Using thinking model variant")
    return model_path


# ── Load ──────────────────────────────────────────────────────────────────

def load_model(args, use_vllm: bool = True) -> FleurModel:
    """
    Load a Qwen-Omni model.

    Args:
        args: Must have lalm_model ('qwen3omni', 'qwen25omni-3b', 'qwen25omni-7b').
              Optional: use_think_mode, tensor_parallel_size, gpu_memory_utilization.
        use_vllm: True for batch/vLLM, False for single-file transformers inference.
    """
    cfg = _resolve_config(args)
    model_path = _resolve_model_path(args, cfg)
    print(f"Loading model from: {model_path}")

    if use_vllm:
        return _load_vllm(args, model_path, cfg)
    else:
        return _load_torch(args, model_path, cfg)


def _load_vllm(args, model_path: str, cfg: dict) -> FleurModel:
    os.environ['VLLM_USE_V1'] = '0'

    from vllm import LLM, SamplingParams
    import transformers
    from transformers import AutoTokenizer

    processor_cls = getattr(transformers, cfg['processor_cls'])

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=getattr(args, 'gpu_memory_utilization', 0.95),
        tensor_parallel_size=getattr(args, 'tensor_parallel_size', 2),
        limit_mm_per_prompt={'image': 0, 'video': 0, 'audio': 1},
        max_num_seqs=8,
        max_model_len=16384,
        seed=1234,
    )

    processor = processor_cls.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        logprobs=20,
    )

    return FleurModel(
        model=llm,
        processor=processor,
        rate2token=build_rate2token(tokenizer),
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        backend="vllm",
        _get_fleur_fn=get_fleur,
    )


def _load_torch(args, model_path: str, cfg: dict) -> FleurModel:
    import transformers

    processor_cls = getattr(transformers, cfg['processor_cls'])
    model_cls = getattr(transformers, cfg['model_cls'])

    processor = processor_cls.from_pretrained(model_path)
    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    return FleurModel(
        model=model,
        processor=processor,
        rate2token=build_rate2token(processor),
        backend="torch",
        _get_fleur_fn=get_fleur,
    )


# ── Inference ─────────────────────────────────────────────────────────────

def get_fleur(fm: FleurModel, caption: str, audio: str, **kwargs) -> tuple:
    """
    Compute FLEUR score for a single audio-caption pair.

    Returns:
        (raw_score, smoothed_score)
    """
    if fm.backend == "vllm":
        return _get_fleur_vllm(fm, caption, audio)
    else:
        return _get_fleur_torch(fm, caption, audio)


def _get_fleur_vllm(fm: FleurModel, caption: str, audio: str) -> tuple:
    from qwen_omni_utils import process_mm_info

    messages = make_fleur_prompt(audio, caption)

    text = fm.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    audios, _, _ = process_mm_info(messages, use_audio_in_video=False)

    inputs = {
        'prompt': text,
        'multi_modal_data': {},
        'mm_processor_kwargs': {'use_audio_in_video': False},
    }
    if audios is not None:
        inputs['multi_modal_data']['audio'] = audios

    try:
        outputs = fm.model.generate([inputs], sampling_params=fm.sampling_params)
        output_obj = outputs[0].outputs[0]
        output_text = output_obj.text.strip()

        if "</think>" in output_text:
            output_text = output_text.split("</think>")[-1].strip()

        raw_score, score = calculate_smoothed_score_vllm(
            output_text, output_obj.logprobs, output_obj.token_ids, fm.rate2token
        )
    except Exception as e:
        print(f"Error in generation: {e}")
        raw_score = None
        score = None

    return raw_score, score


def _get_fleur_torch(fm: FleurModel, caption: str, audio: str) -> tuple:
    from qwen_omni_utils import process_mm_info

    messages = make_fleur_prompt(audio, caption)

    text = fm.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    audio_data, _, _ = process_mm_info(messages, use_audio_in_video=False)
    inputs = fm.processor(
        text=text, audio=audio_data, return_tensors="pt",
        padding=True, use_audio_in_video=False,
    )
    inputs = inputs.to(fm.model.device).to(fm.model.dtype)

    try:
        with torch.no_grad():
            outputs, _ = fm.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs.sequences[0, input_len:]

        output_text = fm.processor.batch_decode(
            outputs.sequences[:, input_len:],
            skip_special_tokens=True
        )[0].strip()

        if "</think>" in output_text:
            output_text = output_text.split("</think>")[-1].strip()

        if outputs.scores:
            logits = torch.stack(outputs.scores, dim=0).squeeze(1)
            raw_score, score = calculate_smoothed_score_torch(
                output_text, logits, generated_ids, fm.rate2token
            )
        else:
            raw_score = parse_raw_score(output_text)
            score = raw_score

    except Exception as e:
        print(f"Error in generation: {e}")
        raw_score = None
        score = None

    return raw_score, score
