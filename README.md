# CAF-Score: CAF-Score: Calibrating CLAP with LALMs for Reference-Free Audio Caption Evaluation

CAF-Score is a comprehensive reference-free audio-caption alignment evaluation metric that combines **CLAP** (Contrastive Language-Audio Pretraining) similarity scores with **[FLEUR](https://github.com/Yebin46/FLEUR)** scores from Large Audio Language Models (LALMs).

## Overview

This repository provides:
- **CLAP Evaluation**: Unified interface for multiple CLAP models (MS-CLAP, LAION-CLAP, MGA-CLAP, M2D-CLAP)
- **LALM Evaluation**: FLEUR metric implementation for Audio-Flamingo-3 and Qwen3-Omni
- **CAF-Score Computation**: Combined metric for robust audio-caption alignment assessment
- **BRACE Benchmark Evaluation**: Evaluation scripts for the BRACE dataset

## Installation

### Using Conda

```bash
# Create environment from yaml file
conda env create -f environment.yaml

# Activate the environment
conda activate caf_score
```

## Data Preparation

To run evaluations on the BRACE dataset, you need to place the audio files in the following directory structure:

```bash
data/
└── audio/
    ├── clotho/        # Place Clotho audio files here (.wav)
    └── audiocaps/     # Place AudioCaps audio files here (.wav)
```

## Project Structure

```bash
CAF_Score/
├── run_caf.py            # Single audio-caption CAF-Score computation
├── eval_caf.py           # Direct CAF-Score evaluation on BRACE dataset
├── eval_clap.py          # CLAP model evaluation script
├── eval_lalm.py          # LALM (FLEUR) evaluation script
├── calc_caf.py           # CAF-Score calculation from pre-computed results
├── src/
│   ├── clap.py           # Unified CLAP model wrapper
│   ├── af3_fleur.py      # Audio-Flamingo-3 FLEUR implementation
│   ├── qwen3_fleur.py    # Qwen3-Omni FLEUR implementation
│   └── models/           # Model implementations (MGA-CLAP, M2D-CLAP)
├── configs/
│   └── mgaclap_config.yaml
├── data/
│   ├── audio/            # Audio files
│   │   ├── clotho/       # Clotho .wav files
│   │   └── audiocaps/    # AudioCaps .wav files
│   ├── meta/             # BRACE dataset metadata
│   └── results/          # Evaluation results
├── pretrained_models/    # Pre-trained model weights (not included)
├── environment.yaml      # Conda environment specification
└── requirements.txt      # Pip requirements
```

## Quick Start

### Single Audio-Caption CAF-Score

Compute CAF-Score for a single audio file and caption:

```bash
# Basic usage
python run_caf.py --audio_path /path/to/audio.wav --caption "A dog barking loudly" \
    --clap_model m2dclap --lalm_model qwen3omni

# With sliding window for long audio
python run_caf.py --audio_path /path/to/long_audio.wav --caption "Music playing" \
    --clap_model laionclap --lalm_model audioflamingo3 --use_slide_window

# Quiet mode (suppress progress messages)
python run_caf.py --audio_path audio.wav --caption "A caption" \
    --clap_model mgaclap --lalm_model qwen3omni --use_slide_window --quiet
```

**Output example:**
```
============================================================
CAF-Score Results
============================================================
Audio: /path/to/audio.wav
Caption: A dog barking loudly
------------------------------------------------------------
CLAP Model: m2dclap
LALM Model: qwen3omni
------------------------------------------------------------
CLAP Score: 0.3245
FLEUR Score: 0.7812
------------------------------------------------------------
CAF-Score: 0.4158
============================================================
```

### Direct CAF-Score Evaluation on BRACE Dataset

Evaluate CAF-Score directly from audio files (computes both CLAP and FLEUR scores on-the-fly):

```bash
# Basic evaluation
python eval_caf.py --lalm_model audioflamingo3 --clap_model laionclap \
    --dataset audiocaps_main

# With Qwen3-Omni and sliding window
python eval_caf.py --lalm_model qwen3omni --clap_model msclap \
    --dataset clotho_main \
    --use_slide_window --pooling max \
    --tensor_parallel_size 2

# With thinking mode for LALM
python eval_caf.py --lalm_model qwen3omni --clap_model laionclap \
    --dataset audiocaps_hallu --use_think_mode
```

## Usage

### 1. CLAP Evaluation

Evaluate audio-caption alignment using CLAP models:

```bash
# Using MS-CLAP
python eval_clap.py --clap_model msclap --dataset audiocaps_main

# Using LAION-CLAP
python eval_clap.py --clap_model laionclap --dataset clotho_main

# With sliding window for long audio
python eval_clap.py --clap_model mgaclap --dataset audiocaps_hallu \
    --use_slide_window --pooling max
```

**Supported CLAP Models:**
- `msclap`: Microsoft CLAP
- `laionclap`: LAION-CLAP (htsat-base)
- `mgaclap`: MGA-CLAP (requires pre-trained weights)
- `m2dclap`: M2D-CLAP (requires pre-trained weights)

### 2. LALM Evaluation (FLEUR)

Evaluate using Large Audio Language Models:

```bash
# Using Audio-Flamingo-3
python eval_lalm.py --lalm_model audioflamingo3 --dataset audiocaps_main

# Using Qwen3-Omni
python eval_lalm.py --lalm_model qwen3omni --dataset clotho_main \
    --tensor_parallel_size 2

# With thinking mode
python eval_lalm.py --lalm_model qwen3omni --dataset audiocaps_hallu \
    --use_think_mode
```

### 3. CAF-Score Calculation from Pre-computed Results

Calculate CAF-Score from pre-computed CLAP and LALM results:

```bash
python calc_caf.py --lalm_model audioflamingo3 --clap_model laionclap \
    --dataset audiocaps_main
```

## CAF-Score Formula

CAF-Score combines CLAP similarity and FLEUR score using a weighted average:

```
CAF-Score = α × CLAP_Score + (1 - α) × FLEUR_Score
```

Where:
- `α` (alpha): Weight parameter (default: 0.8)
- `CLAP_Score`: Audio-text similarity from CLAP model
- `FLEUR_Score`: Smoothed evaluation score from LALM

## Pre-trained Models

### CLAP Models

| Model | Download | Notes |
|-------|----------|-------|
| MS-CLAP | Automatic (via msclap package) | Version 2023 |
| LAION-CLAP | Automatic (via HuggingFace) | Multiple variants available |
| MGA-CLAP | [Download](https://github.com/Ming-er/mga-clap) | Place in `pretrained_models/mga-clap.pt` |
| M2D-CLAP | [Download](https://github.com/nttcslab/m2d-clap) | Place in `pretrained_models/m2d_clap_*/` |

### LALM Models

| Model | Access |
|-------|--------|
| Audio-Flamingo-3 | [HuggingFace](https://huggingface.co/nvidia/audio-flamingo-3-hf) |
| Qwen3-Omni-Instruct| [HuggingFace](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) |
| Qwen3-Omni-Thinking| [HuggingFace](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking) |

## BRACE Dataset

The BRACE (Benchmark for Rating Audio Caption Evaluation) dataset provides standardized evaluation for audio captioning metrics. Download the dataset from the [official repository](https://github.com/HychTus/BRACE_Evaluation).

### Audio File Setup
For evaluation, place your audio files according to the following paths:
* **Clotho**: `data/audio/clotho/`
* **AudioCaps**: `data/audio/audiocaps/`

Supported subsets:
- `audiocaps_main`: AudioCaps main evaluation set
- `audiocaps_hallu`: AudioCaps hallucination set
- `clotho_main`: Clotho main evaluation set
- `clotho_hallu`: Clotho hallucination set

## Configuration

### Environment Variables

For Qwen3-Omni models, you can set custom model paths:

```bash
export QWEN3_OMNI_MODEL_PATH="/path/to/Qwen3-Omni-30B-A3B-Instruct"
export QWEN3_OMNI_THINKING_MODEL_PATH="/path/to/Qwen3-Omni-30B-A3B-Thinking"
```

### GPU Configuration

Set CUDA devices before running:

```bash
export CUDA_VISIBLE_DEVICES=0,1  # Use GPU 0 and 1
python eval_lalm.py --lalm_model qwen3omni --tensor_parallel_size 2
```

## Python API

You can also use CAF-Score programmatically:

```python
from run_caf import compute_caf_score

# Compute CAF-Score for a single audio-caption pair
result = compute_caf_score(
    audio_path="/path/to/audio.wav",
    caption="A dog barking loudly",
    clap_model_name="laionclap",
    lalm_model_name="audioflamingo3",
    verbose=True
)

print(f"CLAP Score: {result['clap_score']:.4f}")
print(f"FLEUR Score: {result['fleur_score']:.4f}")
print(f"CAF-Score: {result['caf_score']:.4f}")
```

## Citation

If you use CAF-Score in your research, please cite:

```
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FLEUR](https://github.com/Yebin46/FLEUR) - Reference-free evaluation metric
- [MS-CLAP](https://github.com/microsoft/CLAP) - Microsoft-CLAP implementation
- [LAION-CLAP](https://github.com/LAION-AI/CLAP) - LAION-CLAP implementation
- [MGA-CLAP](https://github.com/Ming-er/MGA-CLAP) - MGA-CLAP implementation
- [M2D-CLAP](https://github.com/nttcslab/m2d) - M2D-CLAP implementation
- [Audio-Flamingo-3](https://huggingface.co/nvidia/audio-flamingo-3-hf) - NVIDIA Audio-Flamingo3 model
- [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) - Qwen3 Omni model
