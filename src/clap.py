"""
Unified CLAP Model Loader and Similarity Scorer

Supports:
- MS-CLAP (Microsoft CLAP)
- LAION-CLAP (LAION CLAP)
- MGA-CLAP (Modality-shared Guided Attention CLAP)
- M2D-CLAP (Masked Modeling Duo CLAP)

Returns logits (audio-text similarity scores) with optional sliding window for long audio.
"""

import os
import sys
from typing import Union, List, Optional, Literal
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import librosa


@dataclass
class CLAPConfig:
    """Configuration for CLAP models"""
    model_type: str
    sample_rate: int
    audio_duration: float
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    backbone_name: Optional[str] = None
    pretrained: Optional[str] = None
    version: Optional[str] = None


# Default configurations for each model type
DEFAULT_CONFIGS = {
    'msclap': CLAPConfig(
        model_type='msclap',
        sample_rate=44100,
        audio_duration=7.0,
        version='2023'
    ),
    'laionclap': CLAPConfig(
        model_type='laionclap',
        sample_rate=48000,
        audio_duration=10.0,
        backbone_name='htsat-base',
        pretrained='laion/clap-htsat-unfused'
    ),
    'mgaclap': CLAPConfig(
        model_type='mgaclap',
        sample_rate=32000,
        audio_duration=10.0,
        config_path='configs/mgaclap_config.yaml',
        checkpoint_path='pretrained_models/mga-clap.pt'
    ),
    'm2dclap': CLAPConfig(
        model_type='m2dclap',
        sample_rate=16000,
        audio_duration=10.0,
        checkpoint_path='pretrained_models/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth'
    )
}


class CLAPWrapper:
    """
    Unified wrapper for all CLAP model variants.

    Provides a consistent interface for:
    - Loading different CLAP models
    - Computing audio-text similarity (logits)
    - Sliding window processing for long audio
    - Batch processing
    """

    def __init__(
        self,
        model_type: Literal['msclap', 'laionclap', 'mgaclap', 'm2dclap'],
        device: Optional[str] = None,
        config: Optional[CLAPConfig] = None,
        **kwargs
    ):
        """
        Initialize CLAP wrapper.

        Args:
            model_type: Type of CLAP model ('msclap', 'laionclap', 'mgaclap', 'm2dclap')
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
            config: Optional CLAPConfig to override defaults
            **kwargs: Additional arguments to override config values
        """
        self.model_type = model_type
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Get config
        if config is None:
            config = DEFAULT_CONFIGS.get(model_type)
            if config is None:
                raise ValueError(f"Unknown model type: {model_type}")

        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)

        self.config = config
        self.sample_rate = config.sample_rate
        self.audio_duration = config.audio_duration
        self.window_length = int(self.sample_rate * self.audio_duration)

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the appropriate CLAP model based on model_type."""
        if self.model_type == 'msclap':
            self._load_msclap()
        elif self.model_type == 'laionclap':
            self._load_laionclap()
        elif self.model_type == 'mgaclap':
            self._load_mgaclap()
        elif self.model_type == 'm2dclap':
            self._load_m2dclap()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _load_msclap(self):
        """Load MS-CLAP model."""
        from msclap import CLAP as MSCLAP

        version = self.config.version or '2023'
        use_cuda = self.device == 'cuda'

        print(f"Loading MS-CLAP model version: {version}")
        self.clap_wrapper = MSCLAP(model_fp=None, version=version, use_cuda=use_cuda)
        self.model = self.clap_wrapper.clap.to(self.device)
        self.model.eval()
        self.tokenizer = self.clap_wrapper.tokenizer
        self.args = self.clap_wrapper.args
        self.token_keys = self.clap_wrapper.token_keys

    def _load_laionclap(self):
        """Load LAION-CLAP model."""
        from transformers import ClapModel, ClapProcessor, AutoTokenizer

        backbone_name = self.config.backbone_name or 'htsat-base'
        pretrained = self.config.pretrained

        # Map backbone name to pretrained model
        backbone2pretrained = {
            "htsat-base": "laion/clap-htsat-unfused",
            "htsat-large": "laion/clap-htsat-fused",
            "general": "laion/larger_clap_general",
            "music": "laion/larger_clap_music",
            "music-speech": "laion/larger_clap_music_and_speech",
        }

        if pretrained is None:
            pretrained = backbone2pretrained.get(backbone_name, "laion/clap-htsat-unfused")

        print(f"Loading LAION-CLAP model: {pretrained}")
        self.model = ClapModel.from_pretrained(pretrained).to(self.device)
        self.processor = ClapProcessor.from_pretrained(pretrained)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.model.eval()
        self.backbone_name = backbone_name

    def _load_mgaclap(self):
        """Load MGA-CLAP model."""
        from ruamel.yaml import YAML
        from src.models.mga_models.ase_model import ASE

        config_path = self.config.config_path
        checkpoint_path = self.config.checkpoint_path

        print(f"Loading MGA-CLAP config from: {config_path}")
        yaml = YAML(typ='safe', pure=True)
        with open(config_path, "r") as f:
            mga_config = yaml.load(f)

        print(f"Loading MGA-CLAP model from: {checkpoint_path}")
        self.model = ASE(mga_config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = self.model.text_encoder.tokenizer
        self.mga_config = mga_config

    def _load_m2dclap(self):
        """Load M2D-CLAP model."""
        from src.models.m2d_models.portable_m2d import PortableM2D

        checkpoint_path = self.config.checkpoint_path

        print(f"Loading M2D-CLAP model from: {checkpoint_path}")
        self.model = PortableM2D(weight_file=checkpoint_path, flat_features=True)
        self.model.get_clap_text_encoder()
        self.text_encoder = self.model.text_encoder
        self.tokenizer = self.text_encoder.tokenizer
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio waveform as numpy array
        """
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return audio

    def preprocess_audio(self, audio: np.ndarray, truncate: bool = True) -> np.ndarray:
        """
        Preprocess audio (pad or truncate to target duration).

        Args:
            audio: Audio waveform
            truncate: Whether to truncate long audio (if False, returns original)

        Returns:
            Preprocessed audio waveform
        """
        target_length = self.window_length

        if len(audio) > target_length:
            if truncate:
                audio = audio[:target_length]
        elif len(audio) < target_length:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')

        return audio

    def apply_sliding_window(
        self,
        audio: np.ndarray,
        hop_size_seconds: float = 1.0
    ) -> List[np.ndarray]:
        """
        Apply sliding window to audio signal.

        Args:
            audio: Input audio signal
            hop_size_seconds: Hop size in seconds (default: 1.0)

        Returns:
            List of audio windows
        """
        hop_samples = int(hop_size_seconds * self.sample_rate)
        windows = []

        # If audio is shorter than window, return as is (with padding)
        if len(audio) <= self.window_length:
            return [self.preprocess_audio(audio, truncate=True)]

        # Generate windows
        for start in range(0, len(audio) - self.window_length + 1, hop_samples):
            end = start + self.window_length
            windows.append(audio[start:end])

        return windows

    def encode_audio(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Encode audio to embeddings.

        Args:
            audio: Audio waveform (numpy array or tensor)

        Returns:
            Audio embeddings tensor
        """
        if isinstance(audio, np.ndarray):
            audio = torch.FloatTensor(audio)

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add batch dimension

        audio = audio.to(self.device)

        with torch.no_grad():
            if self.model_type == 'msclap':
                audio_features, _ = self.model.audio_encoder(audio)
                audio_features = F.normalize(audio_features, dim=-1)

            elif self.model_type == 'laionclap':
                # Check if audio is longer than expected
                is_longer = audio.shape[-1] > self.window_length
                if self.backbone_name != "htsat-large":
                    is_longer = False

                inputs = self.processor(
                    audios=audio.cpu().numpy(),
                    sampling_rate=self.sample_rate,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # Remove is_longer if processor added it, then set our own
                inputs.pop('is_longer', None)
                audio_features = self.model.get_audio_features(
                    **inputs,
                    is_longer=torch.tensor([is_longer], dtype=torch.bool, device=self.device)
                )
                audio_features = F.normalize(audio_features, dim=-1)

            elif self.model_type == 'mgaclap':
                _, frame_embeds = self.model.encode_audio(audio)
                audio_features = self.model.msc(frame_embeds, self.model.codebook)
                audio_features = F.normalize(audio_features, dim=-1)

            elif self.model_type == 'm2dclap':
                audio_features = self.model.encode(audio, average_per_time_frame=True)
                if hasattr(self.model.backbone, 'audio_proj'):
                    if not hasattr(self.model.backbone.audio_proj, 'dont_average'):
                        audio_features = audio_features.mean(dim=1)
                    audio_features = self.model.backbone.audio_proj(audio_features)
                audio_features = F.normalize(audio_features, dim=-1)

        return audio_features

    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text to embeddings.

        Args:
            texts: Text or list of texts

        Returns:
            Text embeddings tensor
        """
        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            if self.model_type == 'msclap':
                text_inputs = self.tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=self.args.text_len,
                    return_tensors="pt"
                )
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items() if k in self.token_keys}

                text_encoder_base = self.model.caption_encoder.base
                text_outputs = text_encoder_base(**text_inputs, return_dict=True)

                if 'gpt' in self.args.text_model:
                    batch_size = text_inputs['input_ids'].shape[0]
                    hidden_states = text_outputs.last_hidden_state
                    sequence_lengths = torch.ne(text_inputs['input_ids'], 0).sum(-1) - 1
                    text_base_features = hidden_states[
                        torch.arange(batch_size, device=hidden_states.device),
                        sequence_lengths
                    ]
                else:
                    text_base_features = text_outputs.last_hidden_state[:, 0, :]

                text_features = self.model.caption_encoder.projection(text_base_features)
                text_features = F.normalize(text_features, dim=-1)

            elif self.model_type == 'laionclap':
                text_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                text_features = self.model.get_text_features(**text_inputs)
                text_features = F.normalize(text_features, dim=-1)

            elif self.model_type == 'mgaclap':
                text_input = self.tokenizer(
                    texts,
                    padding='longest',
                    truncation=True,
                    max_length=30,
                    return_tensors="pt",
                    return_special_tokens_mask=True
                ).to(self.device)

                text_encoder_base = self.model.text_encoder.text_encoder
                text_outputs = text_encoder_base(
                    input_ids=text_input.input_ids,
                    attention_mask=text_input.attention_mask,
                    return_dict=True
                )

                text_feats = text_outputs.last_hidden_state
                attn_mask = (1 - text_input["special_tokens_mask"][:, 1:]).contiguous()
                word_embeds = self.model.word_proj(text_feats[:, 1:, :])
                text_features = self.model.msc(word_embeds, self.model.codebook, attn_mask)
                text_features = F.normalize(text_features, dim=-1)

            elif self.model_type == 'm2dclap':
                text_input = self.tokenizer(
                    texts,
                    padding='longest',
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                if hasattr(self.text_encoder, 'text_encoder'):
                    text_encoder_base = self.text_encoder.text_encoder
                elif hasattr(self.text_encoder, 'model'):
                    text_encoder_base = self.text_encoder.model
                    if hasattr(text_encoder_base, '_modules') and 'auto_model' in text_encoder_base._modules:
                        text_encoder_base = text_encoder_base.auto_model

                text_outputs = text_encoder_base(
                    input_ids=text_input.input_ids,
                    attention_mask=text_input.attention_mask,
                    return_dict=True
                )

                text_features = text_outputs.last_hidden_state[:, 0, :]
                if hasattr(self.model.backbone, 'text_proj'):
                    text_features = self.model.backbone.text_proj(text_features)
                text_features = F.normalize(text_features, dim=-1)

        return text_features

    def get_logit_scale(self) -> float:
        """Get the logit scale factor for the model."""
        with torch.no_grad():
            if self.model_type == 'msclap':
                return self.model.logit_scale.exp().item()
            elif self.model_type == 'laionclap':
                return self.model.logit_scale_t.exp().item()
            elif self.model_type == 'mgaclap':
                return 1.0 / self.model.temp  # temp is like inverse logit_scale
            elif self.model_type == 'm2dclap':
                return 1.0  # M2D-CLAP doesn't have explicit logit scale
        return 1.0

    def get_similarity(
        self,
        audio: Union[str, np.ndarray, torch.Tensor, List[str], List[np.ndarray]],
        text: Union[str, List[str]],
        use_sliding_window: bool = False,
        hop_size_seconds: float = 1.0,
        pooling: Literal['max', 'mean'] = 'mean'
    ) -> torch.Tensor:
        """
        Compute audio-text similarity (logits).

        Args:
            audio: Audio file path(s), waveform(s), or tensor(s)
            text: Text caption(s)
            use_sliding_window: Whether to use sliding window for long audio
            hop_size_seconds: Hop size for sliding window in seconds (default: 1.0)
            pooling: Pooling method for sliding window ('max' or 'mean')

        Returns:
            Similarity logits tensor of shape (batch_audio, batch_text)
        """

        # Handle single text input
        if isinstance(text, str):
            text = [text]

        # Encode text (batch)
        text_features = self.encode_text(text)  # (num_texts, embed_dim)

        # Process each audio
        all_audio_features = []

        # Load audio if path
        if isinstance(audio, str):
            audio = self.load_audio(audio)
        elif isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        if use_sliding_window and len(audio) > self.window_length:
            # Apply sliding window
            windows = self.apply_sliding_window(audio, hop_size_seconds)

            # Encode each window
            window_features = []
            for window in windows:
                window_feat = self.encode_audio(window)  # (1, embed_dim)
                window_features.append(window_feat)

            # Stack and pool
            window_features = torch.cat(window_features, dim=0)  # (num_windows, embed_dim)

            if pooling == 'max':
                audio_feat = window_features.max(dim=0, keepdim=True)[0]
            else:  # mean
                audio_feat = window_features.mean(dim=0, keepdim=True)
        else:
            # Preprocess and encode single audio
            audio = self.preprocess_audio(audio, truncate=True)
            audio_feat = self.encode_audio(audio)  # (1, embed_dim)

        all_audio_features.append(audio_feat)

        # Stack all audio features
        audio_features = torch.cat(all_audio_features, dim=0)  # (1, embed_dim)

        # Compute similarity
        logit_scale = self.get_logit_scale()
        logits = audio_features @ text_features.t()  # (1, num_texts)

        return logits

    def __call__(
        self,
        audio: Union[str, np.ndarray, torch.Tensor, List[str], List[np.ndarray]],
        text: Union[str, List[str]],
        use_sliding_window: bool = False,
        hop_size_seconds: float = 1.0,
        pooling: Literal['max', 'mean'] = 'mean'
    ) -> torch.Tensor:
        """Alias for get_similarity method."""
        return self.get_similarity(
            audio=audio,
            text=text,
            use_sliding_window=use_sliding_window,
            hop_size_seconds=hop_size_seconds,
            pooling=pooling
        )


def load_clap(
    model_type: Literal['msclap', 'laionclap', 'mgaclap', 'm2dclap'],
    device: Optional[str] = None,
    **kwargs
) -> CLAPWrapper:
    """
    Convenience function to load a CLAP model.

    Args:
        model_type: Type of CLAP model
        device: Device to use
        **kwargs: Additional config overrides

    Returns:
        CLAPWrapper instance
    """
    return CLAPWrapper(model_type=model_type, device=device, **kwargs)


# Example usage
if __name__ == "__main__":
    # Example: Load LAION-CLAP and compute similarity
    clap = load_clap('laionclap', device='cuda')

    audio_path = "/path/to/audio.wav"
    texts = ["A dog barking", "A cat meowing", "Water flowing"]

    # Get similarity scores
    logits = clap.get_similarity(audio_path, texts)
    print(f"Similarity scores: {logits}")

    # With sliding window for long audio
    logits_sw = clap.get_similarity(
        audio_path,
        texts,
        use_sliding_window=True,
        hop_size_seconds=1.0,
        pooling='max'
    )
    print(f"Similarity scores (sliding window): {logits_sw}")
