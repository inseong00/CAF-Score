"""
Evaluate Large Audio Language Models (AudioFlamingo3, Qwen3-Omni) on BRACE dataset
using FLEUR metric for audio-caption alignment.
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser


def eval_lalm_on_dataset(args):
    """
    Evaluate the LALM model on BRACE dataset using FLEUR scores.

    For each audio-caption pair, compute FLEUR score and determine which caption
    better matches the audio based on the scores.

    Args:
        args: Command line arguments

    Returns:
        dict: Accuracy results for HH, HM, MM categories
    """
    with open(args.json_path, 'r') as f:
        dataset = json.load(f)

    if args.output_json is not None:
        output_json_path = args.output_json
    else:
        if 'Clotho' in args.json_path:
            dataset_name = 'clotho'
            
        elif 'AudioCaps' in args.json_path:
            dataset_name = 'audiocaps'
        
        else:
            raise ValueError("Please provide output_json path or use a recognized json_path containing 'Clotho' or 'AudioCaps'.")
        audio_dir = f"{args.audio_dir}/{dataset_name}"
        
        if 'Main' in args.json_path:
            subset = 'main'
        elif 'Hallu' in args.json_path:
            subset = 'hallu'
        else:
            raise ValueError("Please provide output_json path or use a recognized json_path containing 'Main' or 'Hallu'.")

        output_json_path = f'data/results/lalm/{args.lalm_model}_{dataset_name}_{subset}.json'
        if args.use_think_mode:
            output_json_path = output_json_path.replace('.json', '_think.json')

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # Load LALM model
    print(f"Loading {args.lalm_model} model...")
    print(f"Think mode: {args.use_think_mode}")
    if args.lalm_model == 'qwen3omni':
        from src.qwen3_fleur import load_model, get_fleur
        llm, processor, tokenizer, sampling_params, rate2token = load_model(args)
    elif args.lalm_model == 'audioflamingo3':
        from src.af3_fleur import load_model, get_fleur
        model, processor, rate2token = load_model(args)

    hh_total = 0
    fleur_hh_correct = 0
    raw_hh_correct = 0
    hh_tie_cases = 0
    hm_total = 0
    fleur_hm_correct = 0
    raw_hm_correct = 0
    hm_tie_cases = 0
    mm_total = 0
    fleur_mm_correct = 0
    raw_mm_correct = 0
    mm_tie_cases = 0
    total = 0
    total_tie_cases = 0
    fleur_correct = 0
    raw_correct = 0
    results = []
    fleur_scores = []
    raw_scores = []

    for item in tqdm(dataset, desc=f"Evaluating {args.lalm_model} on BRACE"):
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

            # Get FLEUR scores for both captions
            if args.lalm_model == 'qwen3omni':
                raw_score0, fleur_score0 = get_fleur(llm, processor, tokenizer, sampling_params, rate2token, caption0, audio_path)
                raw_score1, fleur_score1 = get_fleur(llm, processor, tokenizer, sampling_params, rate2token, caption1, audio_path)
            elif args.lalm_model == 'audioflamingo3':
                raw_score0, fleur_score0 = get_fleur(model, processor, rate2token, caption0, audio_path)
                raw_score1, fleur_score1 = get_fleur(model, processor, rate2token, caption1, audio_path)

            score0 = fleur_score0 if fleur_score0 is not None else 0
            score1 = fleur_score1 if fleur_score1 is not None else 0
            raw_score0 = raw_score0 if raw_score0 is not None else 0
            raw_score1 = raw_score1 if raw_score1 is not None else 0
            
            fleur_scores.append(score0)
            fleur_scores.append(score1)
            raw_scores.append(raw_score0)
            raw_scores.append(raw_score1)

            # Prediction: higher FLEUR score means better match (label 0)
            if subset == 'main':
                fleur_pred = 0 if score0 > score1 else 1
                if raw_score0 > raw_score1:
                    raw_pred = 0
                elif raw_score1 > raw_score0:
                    raw_pred = 1
                else:
                    raw_pred = -1  # Tie case
                # Update accuracy counters
                if 'Human-Human' in key:
                    hh_total += 1
                    if fleur_pred == answer:
                        fleur_hh_correct += 1
                    if raw_pred == answer:
                        raw_hh_correct += 1
                    if raw_pred == -1:
                        hh_tie_cases += 1
                elif 'Human-Machine' in key:
                    hm_total += 1
                    if fleur_pred == answer:
                        fleur_hm_correct += 1
                    if raw_pred == answer:
                        raw_hm_correct += 1
                    if raw_pred == -1:
                        hm_tie_cases += 1
                elif 'Machine-Machine' in key:
                    mm_total += 1
                    if fleur_pred == answer:
                        fleur_mm_correct += 1
                    if raw_pred == answer:
                        raw_mm_correct += 1
                    if raw_pred == -1:
                        mm_tie_cases += 1
            else:
                fleur_pred = 1 if score0 > score1 else 0
                if raw_score0 > raw_score1:
                    raw_pred = 1
                elif raw_score1 > raw_score0:
                    raw_pred = 0
                else:
                    raw_pred = -1  # Tie case
                    total_tie_cases += 1
                if fleur_pred == answer:
                    fleur_correct += 1
                if raw_pred == answer:
                    raw_correct += 1
                total += 1

            new_item[key] = {
                'caption0': caption0,
                'caption1': caption1,
                'answer': answer,
                'caption0_fleur_score': score0,
                'caption1_fleur_score': score1,
                'caption0_raw_score': raw_score0,
                'caption1_raw_score': raw_score1,
                'fleur_pred': fleur_pred,
                'raw_pred': raw_pred
            }

        results.append(new_item)

        # Save intermediate results
        if len(results) % 10 == 0:
            with open(output_json_path, 'w') as f:
                json.dump(results, f, indent=4)

    if subset == 'main':
        result_metric = {
            'FLEUR HH Accuracy': fleur_hh_correct / hh_total if hh_total > 0 else 0,
            'FLEUR HM Accuracy': fleur_hm_correct / hm_total if hm_total > 0 else 0,
            'FLEUR MM Accuracy': fleur_mm_correct / mm_total if mm_total > 0 else 0,
            'FLEUR Overall Accuracy': (fleur_hh_correct + fleur_hm_correct + fleur_mm_correct) / (hh_total + hm_total + mm_total) if (hh_total + hm_total + mm_total) > 0 else 0,
            'Raw HH Accuracy': raw_hh_correct / hh_total if hh_total > 0 else 0,
            'Raw HM Accuracy': raw_hm_correct / hm_total if hm_total > 0 else 0,
            'Raw MM Accuracy': raw_mm_correct / mm_total if mm_total > 0 else 0,
            'Raw Overall Accuracy': (raw_hh_correct + raw_hm_correct + raw_mm_correct) / (hh_total + hm_total + mm_total) if (hh_total + hm_total + mm_total) > 0 else 0,
            'Raw HH Tie Rate': hh_tie_cases / hh_total if hh_total > 0 else 0,
            'Raw HM Tie Rate': hm_tie_cases / hm_total if hm_total > 0 else 0,
            'Raw MM Tie Rate': mm_tie_cases / mm_total if mm_total > 0 else 0,
            'Raw Overall Tie Rate': (hh_tie_cases + hm_tie_cases + mm_tie_cases) / (hh_total + hm_total + mm_total) if (hh_total + hm_total + mm_total) > 0 else 0
        }
    else:
        result_metric = {
            'FLEUR Overall Accuracy': fleur_correct / total if total > 0 else 0,
            'Raw Overall Accuracy': raw_correct / total if total > 0 else 0,
            'Raw Overall Tie Rate': total_tie_cases / total if total > 0 else 0
        }
    final_results = [{'Result_Metric': result_metric}, {'Results': results}]
    
    print(f"FLEUR Scores Mean: {np.mean(fleur_scores):.4f}, Std: {np.std(fleur_scores):.4f}, Max: {np.max(fleur_scores):.4f}, Min: {np.min(fleur_scores):.4f}")
    print(f"Raw Scores Mean: {np.mean(raw_scores):.4f}, Std: {np.std(raw_scores):.4f}, Max: {np.max(raw_scores):.4f}, Min: {np.min(raw_scores):.4f}")
    
        # Save final results
    with open(output_json_path, 'w') as f:
        json.dump(final_results, f, indent=4)

    return final_results


def main():
    parser = ArgumentParser(description="Evaluate LALM models on BRACE dataset")
    parser.add_argument(
        '--lalm_model',
        type=str,
        required=True,
        choices=['audioflamingo3', 'qwen3omni'],
        help='LALM model to use'
    )
    parser.add_argument(
        '--json_path',
        type=str,
        default='data/meta/BRACE_Clotho_Hallu_Processed.json',
        help='Path to the JSON file containing dataset information'
    )
    parser.add_argument(
        '--audio_dir',
        type=str,
        default='data/audio',
        help='Directory containing audio files'
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default=None,
        help='Path to save the evaluation results'
    )
    parser.add_argument(
        '--use_think_mode',
        action='store_true',
        default=False,
        help='Whether to use think mode for the model'
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
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (default: all)'
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Evaluating {args.lalm_model} on BRACE dataset")
    print(f"{'='*60}")

    results = eval_lalm_on_dataset(args)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for key, value in results[0]['Result_Metric'].items():
        print(f"{key}: {value:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
