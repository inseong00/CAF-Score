"""
Evaluate CLAP models on BRACE dataset for audio-caption alignment.

This script evaluates various CLAP models (MS-CLAP, LAION-CLAP, MGA-CLAP, M2D-CLAP)
on the BRACE benchmark dataset.
"""

import os
import json
import numpy as np
from src.clap import load_clap
from tqdm import tqdm
from argparse import ArgumentParser

def eval_clap_on_dataset(args):
    """
    Evaluate the CLAP model on a dataset specified by a JSON file.

    Args:
        clap_model: The loaded CLAP model.
        dataset_json_path (str): Path to the JSON file containing dataset information.

    Returns:
        dict: A dictionary with audio file names as keys and their corresponding embeddings as values.
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
        if args.use_slide_window:
            output_json_path = f'data/results/clap/{args.clap_model}_slide_{args.pooling}_{dataset_name}_{subset}.json'
        else:
            output_json_path = f'data/results/clap/{args.clap_model}_{dataset_name}_{subset}.json'
    clap_model = load_clap(args.clap_model)
    hh_total = 0
    hh_correct = 0
    hm_total = 0
    hm_correct = 0
    mm_total = 0
    mm_correct = 0
    total = 0
    correct = 0
    results = []
    scores = []
    for item in tqdm(dataset, desc="Evaluating CLAP on dataset"):
        file_name = item['file_name']
        audio_path = os.path.join(audio_dir, file_name)
        texts = []
        answers = []
        new_item = {'file_name': file_name}
        for key, value in item.items():
            if key in ['file_name', 'references']:
                continue
            texts.append(value[0])
            texts.append(value[1])
            answers.append({key: value[-1]})
            new_item[key] = {'caption0': value[0], 'caption1': value[1], 'answer': value[-1]}
        
        
        similarities = clap_model.get_similarity(audio_path, texts, use_sliding_window=args.use_slide_window, pooling=args.pooling)
        similarities = similarities.squeeze(0).cpu().numpy()
        similarities_sub1 = similarities[::2]
        similarities_sub2 = similarities[1::2]
        
        for s1, s2, ans in zip(similarities_sub1, similarities_sub2, answers):
            scores.append(max(float(s1), 0.0))
            scores.append(max(float(s2), 0.0))
            if subset == 'main':
                pred = 0 if s1 > s2 else 1
                for key, value in ans.items():
                    new_item[key]['caption0_score'] = max(float(s1), 0.0)
                    new_item[key]['caption1_score'] = max(float(s2), 0.0)
                    new_item[key]['pred'] = pred
                    if 'Human-Human' in key:
                        hh_total += 1
                        if pred == value:
                            hh_correct += 1
                    elif 'Human-Machine' in key:
                        hm_total += 1
                        if pred == value:
                            hm_correct += 1
                    elif 'Machine-Machine' in key:
                        mm_total += 1
                        if pred == value:
                            mm_correct += 1
            else:
                pred = 1 if s1 > s2 else 0
                for key, value in ans.items():
                    new_item[key]['caption0_score'] = max(float(s1), 0.0)
                    new_item[key]['caption1_score'] = max(float(s2), 0.0)
                    new_item[key]['pred'] = pred
                    if pred == value:
                        correct += 1
                    total += 1
        results.append(new_item)
    if subset == 'main':
        hh_acc = hh_correct / hh_total if hh_total > 0 else 0.0
        hm_acc = hm_correct / hm_total if hm_total > 0 else 0.0
        mm_acc = mm_correct / mm_total if mm_total > 0 else 0.0
        overall_acc = (hh_correct + hm_correct + mm_correct) / (hh_total + hm_total + mm_total)
        result_metric = {
            'Human-Human Accuracy': hh_acc,
            'Human-Machine Accuracy': hm_acc,
            'Machine-Machine Accuracy': mm_acc,
            'Overall Accuracy': overall_acc
        }
        print(f"Total {hh_total + hm_total + mm_total} pairs evaluated.")
    else:
        overall_acc = correct / total if total > 0 else 0.0
        result_metric = {
            'Overall Accuracy': overall_acc
        }
        print(f"Total {total} pairs evaluated.")
    final_results = [{'Result_Metric': result_metric}, {'Results': results}]
    with open(output_json_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    
                        

                    
    return final_results

def main():
    parser = ArgumentParser()
    parser.add_argument('--clap_model', type=str, required=True, help='Path to the CLAP model name.')
    parser.add_argument('--json_path', type=str, default='data/meta/BRACE_AudioCaps_Hallu_Processed.json', help='Path to the JSON file containing dataset information.')
    parser.add_argument('--audio_dir', type=str, default='data/audio', help='Directory containing audio files.')
    parser.add_argument('--output_json', type=str, default= None, help='Path to save the evaluation results.')
    parser.add_argument('--use_slide_window', action='store_true', default=False, help='Whether to use sliding window for long audio.')
    parser.add_argument('--pooling', type=str, default='max', choices=['mean', 'max'],help='Pooling method for SLIDE-CLAP.')
    args = parser.parse_args()

    results = eval_clap_on_dataset(args)
    for key, value in results[0]['Result_Metric'].items():
        print(f"{key}: {value:.4f}")
        
if __name__ == "__main__":
    main()