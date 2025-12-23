# CAF-Score: CLAP-LALM Audio Faithfulness Score

CAF-Score is a comprehensive audio-caption alignment evaluation framework that combines **CLAP** (Contrastive Language-Audio Pretraining) similarity scores with **FLEUR** (Flexible Evaluation Using Language Models) scores from Large Audio Language Models (LALMs).

## Overview

This repository provides:
- **CLAP Evaluation**: Unified interface for multiple CLAP models (MS-CLAP, LAION-CLAP, MGA-CLAP, M2D-CLAP)
- **LALM Evaluation**: FLEUR metric implementation for Audio-Flamingo-3 and Qwen3-Omni
- **CAF-Score Computation**: Combined metric for robust audio-caption alignment assessment
- **BRACE Benchmark Evaluation**: Evaluation scripts for the BRACE dataset

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create environment from yaml file
conda env create -f environment.yaml

# Activate the environment
conda activate caf_score
```

### Option 2: Using pip

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support first
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

## Project Structure

```
CAF_Score/
├── eval_clap.py          # CLAP model evaluation script
├── eval_lalm.py          # LALM (FLEUR) evaluation script
├── eval_caf.py           # CAF-Score evaluation (combining CLAP + LALM)
├── calc_caf.py           # CAF-Score calculation from pre-computed results
├── src/
│   ├── clap.py           # Unified CLAP model wrapper
│   ├── af3_fleur.py      # Audio-Flamingo-3 FLEUR implementation
│   ├── qwen3_fleur.py    # Qwen3-Omni FLEUR implementation
│   └── models/           # Model implementations (MGA-CLAP, M2D-CLAP)
├── configs/
│   └── mgaclap_config.yaml
├── data/
│   ├── audio/            # Audio files (not included)
│   ├── meta/             # BRACE dataset metadata
│   └── results/          # Evaluation results
├── pretrained_models/    # Pre-trained model weights (not included)
├── environment.yaml      # Conda environment specification
└── requirements.txt      # Pip requirements
```

## Usage

### 1. CLAP Evaluation

Evaluate audio-caption alignment using CLAP models:

```bash
# Using MS-CLAP
python eval_clap.py --clap_model msclap \
    --json_path data/meta/BRACE_AudioCaps_Main_Processed.json \
    --audio_dir data/audio

# Using LAION-CLAP
python eval_clap.py --clap_model laionclap \
    --json_path data/meta/BRACE_Clotho_Main_Processed.json \
    --audio_dir data/audio

# With sliding window for long audio
python eval_clap.py --clap_model mgaclap \
    --json_path data/meta/BRACE_AudioCaps_Hallu_Processed.json \
    --use_slide_window --pooling max
```

**Supported CLAP Models:**
- `msclap`: Microsoft CLAP
- `laionclap`: LAION-CLAP (htsat-base, htsat-large, general, music, music-speech)
- `mgaclap`: MGA-CLAP (requires pre-trained weights)
- `m2dclap`: M2D-CLAP (requires pre-trained weights)

### 2. LALM Evaluation (FLEUR)

Evaluate using Large Audio Language Models:

```bash
# Using Audio-Flamingo-3
python eval_lalm.py --lalm_model audioflamingo3 \
    --json_path data/meta/BRACE_AudioCaps_Main_Processed.json \
    --audio_dir data/audio

# Using Qwen3-Omni
python eval_lalm.py --lalm_model qwen3omni \
    --json_path data/meta/BRACE_Clotho_Main_Processed.json \
    --tensor_parallel_size 2

# With thinking mode
python eval_lalm.py --lalm_model qwen3omni \
    --json_path data/meta/BRACE_AudioCaps_Hallu_Processed.json \
    --use_think_mode
```

### 3. CAF-Score Calculation

Calculate CAF-Score from pre-computed CLAP and LALM results:

```bash
# Weighted average method
python calc_caf.py --lalm_model audioflamingo3 \
    --clap_model laionclap \
    --dataset audiocaps_main \
    --avg_method weighted --alpha 0.5

# Harmonic mean method
python calc_caf.py --lalm_model qwen3omni \
    --clap_model mgaclap \
    --dataset clotho_main \
    --avg_method harmonic --beta 1.0
```

## Pre-trained Models

### CLAP Models

| Model | Download | Notes |
|-------|----------|-------|
| MS-CLAP | Automatic (via msclap package) | Version 2023 |
| LAION-CLAP | Automatic (via HuggingFace) | Multiple variants available |
| MGA-CLAP | [Download](https://github.com/your-repo/mga-clap) | Place in `pretrained_models/mga-clap.pt` |
| M2D-CLAP | [Download](https://github.com/your-repo/m2d-clap) | Place in `pretrained_models/m2d_clap_*/` |

### LALM Models

| Model | Access |
|-------|--------|
| Audio-Flamingo-3 | [HuggingFace](https://huggingface.co/nvidia/audio-flamingo-3-hf) |
| Qwen3-Omni | [HuggingFace](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) |

## BRACE Dataset

The BRACE (Benchmark for Rating Audio Caption Evaluation) dataset provides standardized evaluation for audio captioning metrics. Download the dataset from the [official repository](https://github.com/your-repo/brace).

Supported subsets:
- `audiocaps_main`: AudioCaps main evaluation set
- `audiocaps_hallu`: AudioCaps hallucination detection set
- `clotho_main`: Clotho main evaluation set
- `clotho_hallu`: Clotho hallucination detection set

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

## Citation

If you use CAF-Score in your research, please cite:

```bibtex
@article{cafscore2024,
  title={CAF-Score: A Comprehensive Audio Faithfulness Score for Audio Captioning},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FLEUR](https://github.com/Yebin46/FLEUR) - Reference-free evaluation metric
- [MS-CLAP](https://github.com/microsoft/CLAP) - Microsoft CLAP implementation
- [LAION-CLAP](https://github.com/LAION-AI/CLAP) - LAION CLAP implementation
- [Audio-Flamingo-3](https://huggingface.co/nvidia/audio-flamingo-3-hf) - NVIDIA Audio-Flamingo model
- [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) - Qwen audio-language model
