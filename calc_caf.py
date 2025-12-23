import os
import numpy as np
import json
from argparse import ArgumentParser

"""
Evaluate CAF-Score which combining CLAP Similarity from CLAP and FLEUR from LALM
on BRACE dataset for audio-caption alignment.
"""


def eval_caf_on_dataset(args):
    """
    Evaluate the CAF on BRACE dataset using FLEUR scores.

    For each audio-caption pair, compute CAF score and determine which caption
    better matches the audio based on the scores.

    Args:
        args: Command line arguments

    Returns:
        dict: Accuracy results for HH, HM, MM categories
    """
    lalm_result_json = f"{args.data_dir}/results/lalm/{args.lalm_model}_{args.dataset}.json"
    if args.use_think_mode:
        lalm_result_json = f"{args.data_dir}/results/lalm/{args.lalm_model}_{args.dataset}_think.json"
    with open(lalm_result_json, 'r') as f:
        lalm_results = json.load(f)
        
    clap_result_json = f"{args.data_dir}/results/clap/{args.clap_model}_{args.dataset}.json"
    if args.use_slide_window:
        clap_result_json = f"{args.data_dir}/results/clap/{args.clap_model}_slide_{args.dataset}.json"
        if args.pooling == 'max':
            clap_result_json = f"{args.data_dir}/results/clap/{args.clap_model}_slide_max_{args.dataset}.json"
    with open(clap_result_json, 'r') as f:
        clap_results = json.load(f)
        
    if args.avg_method == 'weighted':
        output_json_path = f"{args.data_dir}/results/caf/{args.dataset}/{args.lalm_model}_{args.clap_model}/{args.avg_method}_{int(args.alpha*10)}.json"
    elif args.avg_method == 'harmonic':
        output_json_path = f"{args.data_dir}/results/caf/{args.dataset}/{args.lalm_model}_{args.clap_model}/{args.avg_method}_{int(args.beta*10)}.json"
    if args.use_think_mode:
        output_json_path = output_json_path.replace(args.lalm_model, f"{args.lalm_model}_think")
    if args.use_slide_window:
        output_json_path = output_json_path.replace(f"{args.clap_model}", f"{args.clap_model}_slide")
        if args.pooling == 'max':
            output_json_path = output_json_path.replace(f"{args.clap_model}_slide", f"{args.clap_model}_slide_max")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # Load datasets

    hh_total = 0
    caf_hh_correct = 0
    raw_caf_hh_correct = 0
    hm_total = 0
    caf_hm_correct = 0
    raw_caf_hm_correct = 0
    mm_total = 0
    caf_mm_correct = 0
    raw_caf_mm_correct = 0
    total = 0
    caf_correct = 0
    raw_caf_correct = 0
    results = []
    caf_scores = []
    raw_caf_scores = []

    results_pair = zip(lalm_results[1]['Results'], clap_results[1]['Results'])
    
    for idx, (lalm_item, clap_item) in enumerate(results_pair):
        if args.max_samples is not None and idx >= args.max_samples:
            break
        new_item = {}
        subset = 'main' if 'main' in args.dataset else 'hallu'
        for lalm_key, clap_key in zip(lalm_item.keys(), clap_item.keys()):
            assert lalm_key == clap_key, "Mismatched keys between LALM and CLAP results"
            key = lalm_key
            if key == 'file_name':
                new_item[key] = lalm_item[key]
                continue
            fleur_score0 = lalm_item[key]['caption0_fleur_score']
            fleur_score1 = lalm_item[key]['caption1_fleur_score']
            raw_score0 = lalm_item[key]['caption0_raw_score']
            raw_score1 = lalm_item[key]['caption1_raw_score']
            clap_score0 = clap_item[key]['caption0_score']
            clap_score1 = clap_item[key]['caption1_score']
            caption0 = lalm_item[key]['caption0']
            caption1 = lalm_item[key]['caption1']
            answer = lalm_item[key]['answer']  # 0 or 1

            fleur_score0 = fleur_score0 if fleur_score0 is not None else 0
            fleur_score1 = fleur_score1 if fleur_score1 is not None else 0
            raw_score0 = raw_score0 if raw_score0 is not None else 0
            raw_score1 = raw_score1 if raw_score1 is not None else 0
            
            # Combine scores to get CAF score
            if args.avg_method == 'weighted':
                caf_score0 = args.alpha * clap_score0 + (1 - args.alpha) * fleur_score0
                caf_score1 = args.alpha * clap_score1 + (1 - args.alpha) * fleur_score1
                raw_caf_score0 = args.alpha * clap_score0 + (1 - args.alpha) * raw_score0
                raw_caf_score1 = args.alpha * clap_score1 + (1 - args.alpha) * raw_score1
            elif args.avg_method == 'harmonic':
                caf_score0 = (1 + args.beta**2) * (clap_score0 * fleur_score0) / ((args.beta**2) * clap_score0 + fleur_score0 + 1e-8)
                caf_score1 = (1 + args.beta**2) * (clap_score1 * fleur_score1) / ((args.beta**2) * clap_score1 + fleur_score1 + 1e-8)
                raw_caf_score0 = (1 + args.beta**2) * (clap_score0 * raw_score0) / ((args.beta**2) * clap_score0 + raw_score0 + 1e-8)
                raw_caf_score1 = (1 + args.beta**2) * (clap_score1 * raw_score1) / ((args.beta**2) * clap_score1 + raw_score1 + 1e-8)

            # Prediction: higher caf score means better match (label 0)
            if subset == 'main':
                caf_pred = 0 if caf_score0 > caf_score1 else 1
                if raw_caf_score0 > raw_caf_score1:
                    raw_caf_pred = 0
                elif raw_caf_score1 > raw_caf_score0:
                    raw_caf_pred = 1
                else:
                    raw_caf_pred = -1  # Tie case
                # Update accuracy counters
                if 'Human-Human' in key:
                    hh_total += 1
                    if caf_pred == answer:
                        caf_hh_correct += 1
                    if raw_caf_pred == answer:
                        raw_caf_hh_correct += 1
                elif 'Human-Machine' in key:
                    hm_total += 1
                    if caf_pred == answer:
                        caf_hm_correct += 1
                    if raw_caf_pred == answer:
                        raw_caf_hm_correct += 1
                elif 'Machine-Machine' in key:
                    mm_total += 1
                    if caf_pred == answer:
                        caf_mm_correct += 1
                    if raw_caf_pred == answer:
                        raw_caf_mm_correct += 1
            else:
                caf_pred = 1 if caf_score0 > caf_score1 else 0
                if raw_caf_score0 > raw_caf_score1:
                    raw_caf_pred = 1
                elif raw_score1 > raw_score0:
                    raw_caf_pred = 0
                else:
                    raw_caf_pred = -1  # Tie case
                if caf_pred == answer:
                    caf_correct += 1
                if raw_caf_pred == answer:
                    raw_caf_correct += 1
                total += 1
            caf_scores.extend([caf_score0, caf_score1])
            raw_caf_scores.extend([raw_caf_score0, raw_caf_score1])

            new_item[key] = {
                'caption0': caption0,
                'caption1': caption1,
                'answer': answer,
                'caption0_caf_score': caf_score0,
                'caption1_caf_score': caf_score1,
                'caption0_raw_csf_score': raw_caf_score0,
                'caption1_raw_csf_score': raw_caf_score1,
                'caf_prediction': caf_pred,
                'raw_caf_prediction': raw_caf_pred
            }

        results.append(new_item)

        # Save intermediate results
        if len(results) % 10 == 0:
            with open(output_json_path, 'w') as f:
                json.dump(results, f, indent=4)

    if subset == 'main':
        result_metric = {
            'Total Pairs Evaluated': hh_total + hm_total + mm_total,
            'Total HH Pairs': hh_total,
            'Total HM Pairs': hm_total,
            'Total MM Pairs': mm_total,
            'CAF HH Accuracy': caf_hh_correct / hh_total if hh_total > 0 else 0,
            'CAF HM Accuracy': caf_hm_correct / hm_total if hm_total > 0 else 0,
            'CAF MM Accuracy': caf_mm_correct / mm_total if mm_total > 0 else 0,
            'CAF Overall Accuracy': (caf_hh_correct + caf_hm_correct + caf_mm_correct) / (hh_total + hm_total + mm_total) if (hh_total + hm_total + mm_total) > 0 else 0,
            'Raw CAF HH Accuracy': raw_caf_hh_correct / hh_total if hh_total > 0 else 0,
            'Raw CAF HM Accuracy': raw_caf_hm_correct / hm_total if hm_total > 0 else 0,
            'Raw CAF MM Accuracy': raw_caf_mm_correct / mm_total if mm_total > 0 else 0,
            'Raw CAF Overall Accuracy': (raw_caf_hh_correct + raw_caf_hm_correct + raw_caf_mm_correct) / (hh_total + hm_total + mm_total) if (hh_total + hm_total + mm_total) > 0 else 0
        }
    else:
        result_metric = {
            'Total Pairs Evaluated': total,
            'CAF Overall Accuracy': caf_correct / total if total > 0 else 0,
            'Raw CAF Overall Accuracy': raw_caf_correct / total if total > 0 else 0
        }
    final_results = [{'Result_Metric': result_metric}, {'Results': results}]
    
    print(f"CAF Scores Mean: {np.mean(caf_scores):.4f}, Std: {np.std(caf_scores):.4f}, Max: {np.max(caf_scores):.4f}, Min: {np.min(caf_scores):.4f}")
    print(f"Raw CAF Scores Mean: {np.mean(raw_caf_scores):.4f}, Std: {np.std(raw_caf_scores):.4f}, Max: {np.max(raw_caf_scores):.4f}, Min: {np.min(raw_caf_scores):.4f}")
    
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
        '--clap_model',
        type=str,
        required=True,
        choices=['msclap', 'laionclap', 'mgaclap', 'm2dclap'], 
        help='Path to the CLAP model name.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['clotho_main', 'clotho_hallu', 'audiocaps_main', 'audiocaps_hallu'],
        help='Dataset to evaluate on'
    )
    parser.add_argument(
        '--use_slide_window',
        action='store_true',
        default=False,
        help='Whether to use sliding window for long audio inputs'
    )
    parser.add_argument(
        '--use_think_mode',
        action='store_true',
        default=False,
        help='Whether to use thinking model variant'
    )
    parser.add_argument(
        '--avg_method',
        type=str,
        default='weighted',
        choices=['weighted', 'harmonic'],
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Weighting factor for combining CLAP and FLEUR scores'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=1.0,
        help='Exponent factor for CLAP scores when combining'
    )
    parser.add_argument(
        '--pooling',
        type=str,
        default='max',
        choices=['mean', 'max'],
        help='Pooling method for SLIDE-CLAP.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Path to the dataset directory'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (default: all)'
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Evaluating {args.lalm_model} on BRACE {args.dataset} dataset")
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
