"""
Evaluate CAF-Score directly from audio files on BRACE dataset.

This script computes CAF-Score by directly processing audio files using both
CLAP models and LALMs, without requiring pre-computed results.
CAF-Score = alpha * CLAP_similarity + (1 - alpha) * FLEUR_score
"""

import os
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from src.clap import load_clap


def eval_caf_on_dataset(args):
    """
    Evaluate CAF-Score on BRACE dataset by directly computing CLAP and FLEUR scores.

    Args:
        args: Command line arguments

    Returns:
        dict: Accuracy results for HH, HM, MM categories
    """
    # Setup dataset paths
    if args.dataset == 'clotho_main':
        json_path = f"{args.data_dir}/meta/BRACE_Clotho_Main_Processed.json"
        audio_dir = f"{args.data_dir}/audio/clotho"
        subset = 'main'
    elif args.dataset == 'clotho_hallu':
        json_path = f"{args.data_dir}/meta/BRACE_Clotho_Hallu_Processed.json"
        audio_dir = f"{args.data_dir}/audio/clotho"
        subset = 'hallu'
    elif args.dataset == 'audiocaps_main':
        json_path = f"{args.data_dir}/meta/BRACE_AudioCaps_Main_Processed.json"
        audio_dir = f"{args.data_dir}/audio/audiocaps"
        subset = 'main'
    elif args.dataset == 'audiocaps_hallu':
        json_path = f"{args.data_dir}/meta/BRACE_AudioCaps_Hallu_Processed.json"
        audio_dir = f"{args.data_dir}/audio/audiocaps"
        subset = 'hallu'

    with open(json_path, 'r') as f:
        dataset = json.load(f)

    # Setup output path
    output_json_path = f"{args.data_dir}/results/caf_direct/{args.dataset}/{args.lalm_model}_{args.clap_model}/alpha_{int(args.alpha*10)}.json"
    if args.use_think_mode:
        output_json_path = output_json_path.replace(args.lalm_model, f"{args.lalm_model}_think")
    if args.use_slide_window:
        output_json_path = output_json_path.replace(args.clap_model, f"{args.clap_model}_slide_{args.pooling}")

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # Load CLAP model
    print(f"Loading CLAP model: {args.clap_model}")
    clap_model = load_clap(args.clap_model)

    # Load LALM model
    print(f"Loading LALM model: {args.lalm_model}")
    if args.lalm_model == 'qwen3omni':
        from src.qwen3_fleur import load_model as load_lalm, get_fleur
        llm, processor, tokenizer, sampling_params, rate2token = load_lalm(args)
        lalm_components = (llm, processor, tokenizer, sampling_params, rate2token)
    elif args.lalm_model == 'audioflamingo3':
        from src.af3_fleur import load_model as load_lalm, get_fleur
        model, processor, rate2token = load_lalm(args)
        lalm_components = (model, processor, rate2token)

    # Counters for accuracy
    hh_total = 0
    caf_hh_correct = 0
    hm_total = 0
    caf_hm_correct = 0
    mm_total = 0
    caf_mm_correct = 0
    total = 0
    caf_correct = 0
    results = []
    caf_scores = []

    for item in tqdm(dataset, desc=f"Evaluating CAF-Score on {args.dataset}"):
        file_name = item['file_name']
        audio_path = os.path.join(audio_dir, file_name)

        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}, skipping...")
            continue

        new_item = {'file_name': file_name}

        for key, value in item.items():
            if key in ['file_name', 'references']:
                continue

            caption0 = value[0]
            caption1 = value[1]
            answer = value[-1]

            # Get CLAP similarities for both captions
            clap_similarities = clap_model.get_similarity(
                audio_path,
                [caption0, caption1],
                use_sliding_window=args.use_slide_window,
                pooling=args.pooling
            )
            clap_score0 = max(float(clap_similarities[0, 0].cpu()), 0.0)
            clap_score1 = max(float(clap_similarities[0, 1].cpu()), 0.0)

            # Get FLEUR scores for both captions
            if args.lalm_model == 'qwen3omni':
                llm, processor, tokenizer, sampling_params, rate2token = lalm_components
                raw_score0, fleur_score0 = get_fleur(llm, processor, tokenizer, sampling_params, rate2token, caption0, audio_path)
                raw_score1, fleur_score1 = get_fleur(llm, processor, tokenizer, sampling_params, rate2token, caption1, audio_path)
            elif args.lalm_model == 'audioflamingo3':
                model, processor, rate2token = lalm_components
                raw_score0, fleur_score0 = get_fleur(model, processor, rate2token, caption0, audio_path)
                raw_score1, fleur_score1 = get_fleur(model, processor, rate2token, caption1, audio_path)

            fleur_score0 = fleur_score0 if fleur_score0 is not None else 0
            fleur_score1 = fleur_score1 if fleur_score1 is not None else 0

            # Compute CAF-Score: alpha * CLAP + (1 - alpha) * FLEUR
            caf_score0 = args.alpha * clap_score0 + (1 - args.alpha) * fleur_score0
            caf_score1 = args.alpha * clap_score1 + (1 - args.alpha) * fleur_score1

            caf_scores.extend([caf_score0, caf_score1])

            # Make prediction based on CAF-Score
            if subset == 'main':
                caf_pred = 0 if caf_score0 > caf_score1 else 1
                if 'Human-Human' in key:
                    hh_total += 1
                    if caf_pred == answer:
                        caf_hh_correct += 1
                elif 'Human-Machine' in key:
                    hm_total += 1
                    if caf_pred == answer:
                        caf_hm_correct += 1
                elif 'Machine-Machine' in key:
                    mm_total += 1
                    if caf_pred == answer:
                        caf_mm_correct += 1
            else:
                caf_pred = 1 if caf_score0 > caf_score1 else 0
                if caf_pred == answer:
                    caf_correct += 1
                total += 1

            new_item[key] = {
                'caption0': caption0,
                'caption1': caption1,
                'answer': answer,
                'caption0_clap_score': clap_score0,
                'caption1_clap_score': clap_score1,
                'caption0_fleur_score': fleur_score0,
                'caption1_fleur_score': fleur_score1,
                'caption0_caf_score': caf_score0,
                'caption1_caf_score': caf_score1,
                'caf_prediction': caf_pred
            }

        results.append(new_item)

        # Save intermediate results
        if len(results) % 10 == 0:
            with open(output_json_path, 'w') as f:
                json.dump(results, f, indent=4)

    # Compute final metrics
    if subset == 'main':
        result_metric = {
            'Total Pairs Evaluated': hh_total + hm_total + mm_total,
            'Total HH Pairs': hh_total,
            'Total HM Pairs': hm_total,
            'Total MM Pairs': mm_total,
            'CAF HH Accuracy': caf_hh_correct / hh_total if hh_total > 0 else 0,
            'CAF HM Accuracy': caf_hm_correct / hm_total if hm_total > 0 else 0,
            'CAF MM Accuracy': caf_mm_correct / mm_total if mm_total > 0 else 0,
            'CAF Overall Accuracy': (caf_hh_correct + caf_hm_correct + caf_mm_correct) / (hh_total + hm_total + mm_total) if (hh_total + hm_total + mm_total) > 0 else 0
        }
    else:
        result_metric = {
            'Total Pairs Evaluated': total,
            'CAF Overall Accuracy': caf_correct / total if total > 0 else 0
        }

    final_results = [{'Result_Metric': result_metric}, {'Results': results}]

    print(f"\nCAF Scores - Mean: {np.mean(caf_scores):.4f}, Std: {np.std(caf_scores):.4f}, Max: {np.max(caf_scores):.4f}, Min: {np.min(caf_scores):.4f}")

    # Save final results
    with open(output_json_path, 'w') as f:
        json.dump(final_results, f, indent=4)

    return final_results


def main():
    parser = ArgumentParser(description="Evaluate CAF-Score directly from audio on BRACE dataset")
    parser.add_argument(
        '--lalm_model',
        type=str,
        required=True,
        choices=['audioflamingo3', 'qwen3omni'],
        help='LALM model to use for FLEUR scoring'
    )
    parser.add_argument(
        '--clap_model',
        type=str,
        required=True,
        choices=['msclap', 'laionclap', 'mgaclap', 'm2dclap'],
        help='CLAP model to use for similarity scoring'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['clotho_main', 'clotho_hallu', 'audiocaps_main', 'audiocaps_hallu'],
        help='Dataset to evaluate on'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Path to the dataset directory'
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
        help='Pooling method for sliding window'
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

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Evaluating CAF-Score on BRACE {args.dataset}")
    print(f"CLAP Model: {args.clap_model}")
    print(f"LALM Model: {args.lalm_model}")
    print(f"{'='*60}")

    results = eval_caf_on_dataset(args)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for key, value in results[0]['Result_Metric'].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
