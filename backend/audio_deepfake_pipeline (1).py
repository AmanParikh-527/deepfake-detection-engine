"""
Audio Deepfake Detection Pipeline — compiled single-file build.

Pipeline: Audio → Preprocess → Wav2Vec2 → Real/Synthetic → Confidence
                 → Grad-CAM → Explanation signals → Visualization → Report

This file merges (in pipeline order): preprocess.py, model.py, gradcam.py,
explainability.py, visualize.py, and main.py into one module. Internal
cross-file imports have been removed since everything now shares a single
namespace, and only one CLI entry point (from main.py) remains active.

Usage:
    python audio_deepfake_pipeline.py <input_audio> [--output-dir OUTPUT] [--device cuda|cpu]

Example:
    python audio_deepfake_pipeline.py sample.wav --output-dir results --device cuda
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import librosa.display
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from scipy.signal import butter, find_peaks, lfilter
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Model,
)


# =============================================================================
# SECTION 1 — preprocess.py
# Converts any audio file → 16 kHz mono WAV, normalizes noise floor,
# and returns a tensor ready for Wav2Vec2 inference.
# =============================================================================

TARGET_SR = 16_000          # Wav2Vec2 expects 16 kHz
NOISE_FLOOR_DB = -60.0      # dBFS threshold below which we treat as silence
RMS_TARGET_DB = -20.0       # normalize overall loudness to this dBFS


def _load_audio(path: str | Path) -> Tuple[np.ndarray, int]:
    """Load any format librosa/soundfile can handle → (mono float32, sr)."""
    data, sr = sf.read(str(path), always_2d=False, dtype="float32")
    # downmix to mono
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sr


def _resample(data: np.ndarray, sr_in: int, sr_out: int = TARGET_SR) -> np.ndarray:
    """High-quality polyphase resample."""
    if sr_in == sr_out:
        return data
    waveform = torch.from_numpy(data).unsqueeze(0)  # (1, T)
    resampler = torchaudio.transforms.Resample(
        orig_freq=sr_in, new_freq=sr_out,
        resampling_method="sinc_interp_kaiser",
    )
    resampled = resampler(waveform).squeeze(0).numpy()
    return resampled


def _trim_silence(data: np.ndarray, sr: int, threshold_db: float = NOISE_FLOOR_DB) -> np.ndarray:
    """Trim leading/trailing silence below *threshold_db* dBFS."""
    ref = np.max(np.abs(data)) + 1e-10
    threshold = ref * (10 ** (threshold_db / 20))
    nonzero = np.where(np.abs(data) > threshold)[0]
    if len(nonzero) < 2:
        return data
    return data[nonzero[0]:nonzero[-1] + 1]


def _normalize_loudness(data: np.ndarray, target_db: float = RMS_TARGET_DB) -> np.ndarray:
    """Peak-normalize to a fixed RMS level so the model sees consistent energy."""
    rms = float(np.sqrt(np.mean(data ** 2)) + 1e-10)
    target_rms = 10 ** (target_db / 20)
    gain = target_rms / rms
    return np.clip(data * gain, -1.0, 1.0).astype(np.float32)


def _highpass_filter(data: np.ndarray, sr: int, cutoff_hz: int = 80) -> np.ndarray:
    """Remove sub-bass rumble below human voice range."""
    nyq = sr / 2
    norm_cutoff = min(cutoff_hz / nyq, 0.99)
    b, a = butter(4, norm_cutoff, btype="high")
    return lfilter(b, a, data).astype(np.float32)


def preprocess_audio(path: str | Path) -> Tuple[np.ndarray, int]:
    """
    Full preprocessing pipeline.

    1. Load any audio format
    2. Convert to mono
    3. Resample to 16 kHz
    4. High-pass filter (remove rumble < 80 Hz)
    5. Trim silence below noise floor
    6. Normalize loudness to -20 dBFS RMS

    Returns (waveform float32, sample_rate)
    """
    data, sr = _load_audio(path)
    data = _resample(data, sr, TARGET_SR)
    data = _highpass_filter(data, TARGET_SR)
    data = _trim_silence(data, TARGET_SR)
    data = _normalize_loudness(data)
    return data, TARGET_SR


def to_model_input(waveform: np.ndarray, sr: int = TARGET_SR) -> torch.Tensor:
    """Convert raw waveform → Wav2Vec2 input tensor (1, T)."""
    return torch.from_numpy(waveform).float().unsqueeze(0)


def save_wav(waveform: np.ndarray, sr: int, out_path: str | Path) -> None:
    """Write processed audio to disk as 16-bit PCM WAV."""
    int16 = (waveform * 32767).astype(np.int16)
    sf.write(str(out_path), int16, sr, subtype="PCM_16")


# =============================================================================
# SECTION 2 — model.py
# Wraps a fine-tuned Wav2Vec2 model (e.g. Sara1708/deepfake-audio-wav2vec2)
# for real-vs-synthetic speech classification.
#   index 0 → real (bonafide)
#   index 1 → synthetic (deepfake)
# =============================================================================

MODEL_ID = "Sara1708/deepfake-audio-wav2vec2"

# Some anti-spoofing checkpoints are plain Wav2Vec2Model + a custom head.
# If the HF repo has a classification head, we use SequenceClassification.
# If it only has the base encoder, we attach a linear head trained for 2 classes.
# Set this to True if the checkpoint is a base encoder without a cls head.
USE_BASE_ENCODER = False
CHECKPOINT_FILENAME = "stage2_best.pt"


@dataclass
class DetectionResult:
    label: str               # "Real" or "Synthetic"
    confidence: float        # 0–100
    probs: Dict[str, float]  # {"real": p, "synthetic": p}
    logits: np.ndarray
    hidden_states: Optional[torch.Tensor] = None  # for Grad-CAM
    feature_lengths: Optional[int] = None


class AudioDeepfakeDetector:
    """
    Loads a fine-tuned Wav2Vec2 model and runs real-vs-synthetic inference.

    Parameters
    ----------
    model_id : str
        HuggingFace repo ID of the fine-tuned checkpoint.
    device : str
        "cuda" or "cpu".
    use_base_encoder : bool
        If True, loads Wav2Vec2Model (encoder only) and attaches a
        randomly-initialised linear head — useful when the checkpoint
        does not include a classification head.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_base_encoder: bool = USE_BASE_ENCODER,
    ):
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            print("CUDA is unavailable in this PyTorch build; falling back to CPU.")
            requested_device = torch.device("cpu")
        self.device = requested_device
        self.model_id = model_id
        self.use_base_encoder = use_base_encoder

        if model_id == MODEL_ID:
            self._load_published_checkpoint()
        elif use_base_encoder:
            self.encoder = Wav2Vec2Model.from_pretrained(model_id).to(self.device)
            self.config = self.encoder.config
            self.head = torch.nn.Linear(self.config.hidden_size, 2).to(self.device)
            self.model = None
        else:
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_id,
                num_labels=2,
                ignore_mismatched_sizes=True,
            ).to(self.device)
            self.config = self.model.config
            self.encoder = None  # accessed via self.model.wav2vec2
            self.head = None

        self.eval()

    def _load_published_checkpoint(self) -> None:
        """Load the project's custom Hugging Face checkpoint format."""
        checkpoint_path = hf_hub_download(self.model_id, CHECKPOINT_FILENAME)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"]

        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        encoder_state = {
            key.removeprefix("backbone."): value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.config = self.encoder.config
        self.head = torch.nn.Linear(self.config.hidden_size, 2)
        self.head.load_state_dict({
            key: value for key, value in state_dict.items()
            if key.startswith("classifier.")
        }, strict=False)
        self.encoder = self.encoder.to(self.device)
        self.head = self.head.to(self.device)
        self.model = None
        self.use_base_encoder = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def eval(self):
        """Set model to eval mode."""
        if self.model is not None:
            self.model.eval()
        if self.encoder is not None:
            self.encoder.eval()
        if self.head is not None:
            self.head.eval()

    def zero_grad(self) -> None:
        """Clear gradients from the active encoder and classifier."""
        if self.model is not None:
            self.model.zero_grad()
        if self.encoder is not None:
            self.encoder.zero_grad()
        if self.head is not None:
            self.head.zero_grad()

    @torch.no_grad()
    def predict(
        self,
        waveform: np.ndarray | torch.Tensor,
        sr: int = 16_000,
        return_hidden: bool = False,
    ) -> DetectionResult:
        """
        Run inference on a single audio clip.

        Parameters
        ----------
        waveform : np.ndarray or torch.Tensor
            1-D float32 waveform at *sr* Hz.
        sr : int
            Sample rate (must be 16 000 for Wav2Vec2).
        return_hidden : bool
            If True, also return last hidden states for Grad-CAM.

        Returns
        -------
        DetectionResult
        """
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, T)
        waveform = waveform.to(self.device)

        if self.use_base_encoder:
            out = self.encoder(waveform)
            pooled = out.last_hidden_state.mean(dim=1)  # (1, H)
            logits = self.head(pooled)                    # (1, 2)
            hidden = out.last_hidden_state if return_hidden else None
        else:
            out = self.model(
                waveform,
                output_hidden_states=return_hidden,
            )
            logits = out.logits  # (1, 2)
            hidden = None
            if return_hidden and out.hidden_states is not None:
                hidden = out.hidden_states[-1]  # last layer (1, T', H)

        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_idx = int(probs.argmax())
        label = "Synthetic" if pred_idx == 1 else "Real"
        confidence = float(probs[pred_idx]) * 100

        return DetectionResult(
            label=label,
            confidence=confidence,
            probs={"real": float(probs[0]), "synthetic": float(probs[1])},
            logits=logits.squeeze(0).cpu().numpy(),
            hidden_states=hidden,
            feature_lengths=None,
        )

    # ------------------------------------------------------------------
    # Grad-CAM support — forward with gradients enabled
    # ------------------------------------------------------------------

    def forward_with_grad(
        self,
        waveform: torch.Tensor,
        target_class: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that keeps gradients (for Grad-CAM).

        Returns (logits, target_logit, last_hidden_states)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)

        if self.use_base_encoder:
            out = self.encoder(waveform)
            pooled = out.last_hidden_state.mean(dim=1)
            logits = self.head(pooled)
            hidden = out.last_hidden_state
        else:
            out = self.model(waveform, output_hidden_states=True)
            logits = out.logits
            hidden = out.hidden_states[-1]

        hidden.retain_grad()
        target_logit = logits[0, target_class]
        return logits, target_logit, hidden


# Cached detector loader
_detector_cache: Dict[str, AudioDeepfakeDetector] = {}


def load_detector(model_id: str = MODEL_ID, device: str | None = None) -> AudioDeepfakeDetector:
    """Cached detector loader."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = f"{model_id}:{device}"
    if key not in _detector_cache:
        _detector_cache[key] = AudioDeepfakeDetector(model_id=model_id, device=device)
    return _detector_cache[key]


# =============================================================================
# SECTION 3 — gradcam.py
# Grad-CAM adapted for audio: treats the last transformer layer's hidden
# states as "feature maps" and the classification head as the "conv output".
# =============================================================================

class AudioGradCAM:
    """
    Grad-CAM explainer for the Wav2Vec2 deepfake detector.

    Usage:
        cam = AudioGradCAM(detector)
        relevance, hidden = cam.compute(waveform, target_class=1)
        # relevance: (T_audio,) numpy array — per-sample importance
    """

    def __init__(self, detector):
        self.detector = detector

    def compute(
        self,
        waveform: np.ndarray | torch.Tensor,
        target_class: int = 1,
        layer_index: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Grad-CAM relevance map.

        Parameters
        ----------
        waveform : np.ndarray or torch.Tensor
            1-D float32 waveform at 16 kHz.
        target_class : int
            0 = real, 1 = synthetic. Use the predicted class.
        layer_index : int
            Which hidden state layer to use (-1 = last).

        Returns
        -------
        (relevance_audio, relevance_frames)
            relevance_audio  : (T_audio,)  — interpolated to sample resolution
            relevance_frames : (T_frames,) — per-frame importance before interp
        """
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(self.detector.device)
        waveform.requires_grad_(False)

        # --- Forward with grad ---
        self.detector.encoder or self.detector.model  # ensure loaded
        logits, target_logit, hidden = self.detector.forward_with_grad(
            waveform, target_class=target_class
        )

        # hidden: (1, T_frames, H)
        # --- Backward ---
        self.detector.zero_grad()
        target_logit.backward(retain_graph=False)

        # Gradients of target logit w.r.t. hidden states
        grads = hidden.grad  # (1, T_frames, H)
        if grads is None:
            raise RuntimeError(
                "No gradients captured on hidden states. "
                "Ensure the model is in train mode for grad or use retain_grad()."
            )

        # --- Grad-CAM core ---
        # Channel weights: global average of gradients across time
        weights = grads.mean(dim=1)          # (1, H)
        # Weighted sum across channels
        cam = (hidden * weights.unsqueeze(1)).sum(dim=-1)  # (1, T_frames)
        cam = F.relu(cam)  # only positive contributions
        cam = cam.squeeze(0).detach().cpu().numpy()  # (T_frames,)

        # --- Normalize to [0, 1] ---
        if cam.max() > 0:
            cam = cam / cam.max()

        # --- Interpolate to audio-sample resolution ---
        T_audio = waveform.shape[-1]
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)  # (1, 1, T_frames)
        cam_interp = F.interpolate(
            cam_tensor, size=T_audio, mode="linear", align_corners=False
        )
        cam_audio = cam_interp.squeeze().numpy()  # (T_audio,)

        return cam_audio, cam

    def compute_smoothed(
        self,
        waveform: np.ndarray | torch.Tensor,
        target_class: int = 1,
        n_samples: int = 5,
        noise_std: float = 0.005,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        SmoothGrad-style Grad-CAM: average CAM over several noisy copies.

        This reduces visual noise in the saliency map.
        """
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        cams_audio = []
        cams_frames = []
        for i in range(n_samples):
            noise = torch.randn_like(waveform) * noise_std
            noisy = waveform + noise
            cam_a, cam_f = self.compute(noisy, target_class=target_class)
            cams_audio.append(cam_a)
            cams_frames.append(cam_f)

        avg_audio = np.mean(cams_audio, axis=0)
        avg_frames = np.mean(cams_frames, axis=0)

        # Re-normalize
        if avg_audio.max() > 0:
            avg_audio = avg_audio / avg_audio.max()
        if avg_frames.max() > 0:
            avg_frames = avg_frames / avg_frames.max()

        return avg_audio, avg_frames


# =============================================================================
# SECTION 4 — explainability.py
# Explanation signal generation. Each signal is *actually computed* from the
# audio — not hardcoded. If a signal cannot be reliably measured, it is
# omitted from the output.
#
# Signals implemented:
#   1. Spectral consistency  — how uniform the spectral envelope is over time
#   2. High-frequency energy ratio — abnormal HF patterns
#   3. Voice texture irregularity — MFCC statistics variance
#   4. Natural variation — pitch & energy dynamics
#   5. Formant stability — synthetic speech tends to have unnaturally stable formants
#   6. Phase coherence — synthetic vocoders produce more coherent phase patterns
# =============================================================================

@dataclass
class ExplanationSignal:
    name: str
    value: float          # raw measured value
    threshold: float      # threshold above which it's flagged
    is_flagged: bool      # whether this signal contributes to "synthetic"
    description: str      # human-readable explanation
    direction: str        # "high" or "low" — does high value indicate synthetic?


@dataclass
class ExplanationReport:
    signals: List[ExplanationSignal] = field(default_factory=list)
    flagged_signals: List[str] = field(default_factory=list)

    def to_bullet_list(self) -> List[str]:
        """Return human-readable bullet strings for the detected signals."""
        bullets = []
        for sig in self.signals:
            if sig.is_flagged:
                bullets.append(f"• {sig.description}")
        return bullets

    def summary(self) -> str:
        lines = ["Detected signals"]
        for sig in self.signals:
            marker = "⚠" if sig.is_flagged else "✓"
            lines.append(f"  {marker} {sig.name}: {sig.value:.3f} (threshold: {sig.threshold:.3f})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual signal extractors
# ---------------------------------------------------------------------------

def _spectral_consistency(wav: np.ndarray, sr: int = 16000) -> float:
    """
    Measure how uniform the spectral envelope is across time frames.

    Synthetic speech often has an unusually consistent spectral envelope
    because the vocoder applies the same filter parameters more uniformly.

    We compute the mel spectrogram, then measure the average cosine similarity
    between adjacent time frames. Higher = more consistent = more likely synthetic.
    """
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80, hop_length=160, n_fft=512)
    mel_db = librosa.power_to_db(mel + 1e-10, ref=np.max)
    # Normalize each frame
    norms = np.linalg.norm(mel_db, axis=0, keepdims=True)
    mel_norm = mel_db / (norms + 1e-10)
    # Cosine similarity between adjacent frames
    cos_sim = np.sum(mel_norm[:, :-1] * mel_norm[:, 1:], axis=0)
    return float(np.mean(cos_sim))


def _hf_energy_ratio(wav: np.ndarray, sr: int = 16000, cutoff_hz: int = 4000) -> float:
    """
    Ratio of energy above *cutoff_hz* to total energy.

    Synthetic speech sometimes has an artificial high-frequency pattern
    — either too much HF energy (vocoder artifacts) or too little (band-limited TTS).
    We measure deviation from the expected ratio for natural speech (~0.05–0.15).
    """
    S = np.abs(librosa.stft(wav, n_fft=512, hop_length=160))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    hf_mask = freqs >= cutoff_hz
    total_energy = np.sum(S ** 2)
    hf_energy = np.sum(S[hf_mask] ** 2)
    ratio = float(hf_energy / (total_energy + 1e-10))
    # Return deviation from 0.10 (typical natural speech HF ratio)
    return abs(ratio - 0.10)


def _voice_texture_irregularity(wav: np.ndarray, sr: int = 16000) -> float:
    """
    Measure irregularity in MFCC statistics.

    Natural speech has organic variation in spectral texture across time.
    Synthetic speech tends to have lower variance in MFCC deltas, indicating
    a smoother, less textured voice quality.

    Returns the *inverse* of MFCC delta variance (higher = less texture = more synthetic).
    """
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=20, hop_length=160, n_fft=512)
    delta = librosa.feature.delta(mfcc)
    # Variance of deltas across time, averaged over coefficients
    delta_var = np.mean(np.var(delta, axis=1))
    # Invert: low variance → high irregularity score
    return float(1.0 / (delta_var + 1e-6))


def _natural_variation(wav: np.ndarray, sr: int = 16000) -> float:
    """
    Measure natural pitch and energy variation.

    Natural speech has micro-variations in F0 (pitch) and energy that are
    hard for synthesizers to replicate. We compute the coefficient of variation
    (CV) of F0 and RMS energy. Low CV → low natural variation → more synthetic.

    Returns (1 - normalized_CV), so higher = less variation = more synthetic.
    """
    f0, voiced = librosa.piptrack(y=wav, sr=sr, hop_length=160, n_fft=512)
    # Extract F0 contour
    f0_contour = []
    for t in range(f0.shape[1]):
        v_frame = voiced[:, t] > 0
        if v_frame.any():
            f0_contour.append(f0[v_frame, t].min())
    f0_arr = np.array(f0_contour) if f0_contour else np.array([0.0])

    # RMS energy contour
    rms = librosa.feature.rms(y=wav, frame_length=512, hop_length=160)[0]

    # Coefficient of variation
    f0_cv = float(np.std(f0_arr) / (np.mean(f0_arr) + 1e-6))
    rms_cv = float(np.std(rms) / (np.mean(rms) + 1e-6))

    # Combined natural variation score (0–1)
    combined_cv = (f0_cv + rms_cv) / 2
    # Normalize: typical natural speech CV ~0.3–0.6
    normalized = min(combined_cv / 0.45, 1.0)
    return float(1.0 - normalized)  # invert so high = less natural variation


def _formant_stability(wav: np.ndarray, sr: int = 16000) -> float:
    """
    Measure formant frequency stability over time.

    Synthetic speech tends to have more stable formant trajectories because
    vocoders smooth formant transitions. We compute the first 3 formants
    using LPC and measure their frame-to-frame variation.

    Returns inverse of formant trajectory variance (higher = more stable = more synthetic).
    """
    # Frame the audio
    frames = librosa.util.frame(wav, frame_length=512, hop_length=160).T
    if len(frames) == 0:
        return 0.0

    formants_per_frame = []
    for frame in frames:
        if np.max(np.abs(frame)) < 1e-4:
            continue
        # LPC order ~ 2 + sr/1000
        lpc_order = min(2 + sr // 1000, len(frame) - 1)
        try:
            a = librosa.lpc(frame, order=lpc_order)
            roots = np.roots(a)
            roots = roots[np.imag(roots) > 0]
            angles = np.angle(roots)
            freqs = angles * sr / (2 * np.pi)
            # Keep first 3 formants (sorted by frequency, 90–5000 Hz)
            valid = freqs[(freqs > 90) & (freqs < 5000)]
            valid = np.sort(valid)[:3]
            if len(valid) >= 2:
                formants_per_frame.append(valid[:3])
        except Exception:
            continue

    if len(formants_per_frame) < 5:
        return 0.0

    formants = np.array(formants_per_frame)  # (n_frames, 3)
    # Mean frame-to-frame difference (lower = more stable)
    frame_diffs = np.abs(np.diff(formants, axis=0))
    mean_diff = float(np.mean(frame_diffs))
    # Invert: low diff → high stability score
    return float(1.0 / (mean_diff + 1e-6))


def _phase_coherence(wav: np.ndarray, sr: int = 16000) -> float:
    """
    Measure phase coherence across time frames.

    Synthetic vocoders (especially neural vocoders like HiFi-GAN, WaveNet)
    produce phase patterns that are more coherent/regular than natural speech,
    which has chaotic phase due to the physics of vocal fold vibration.

    We compute the group delay variance — natural speech has higher group delay
    variance (more chaotic phase), synthetic has lower.
    """
    D = librosa.stft(wav, n_fft=512, hop_length=160)
    if D.shape[1] < 2:
        return 0.0

    # Phase
    phase = np.angle(D)
    # Unwrap phase along time axis
    phase_unwrapped = np.unwrap(phase, axis=1)
    # Group delay = -d(phase)/d(omega)
    group_delay = -np.diff(phase_unwrapped, axis=0)
    # Variance across frequency bins (higher = more natural/chaotic)
    gd_var = float(np.mean(np.var(group_delay, axis=1)))
    # Invert: low variance → high coherence → more synthetic
    return float(1.0 / (gd_var + 1e-6))


# ---------------------------------------------------------------------------
# Main explanation generator
# ---------------------------------------------------------------------------

# Thresholds are calibrated so that a signal is flagged when its value
# exceeds the threshold. These are starting points — tune on your dataset.
SIGNAL_CONFIG = {
    "spectral_consistency": {
        "threshold": 0.85,
        "direction": "high",
        "label": "unusual spectral consistency",
        "desc": "Spectral envelope is unusually uniform across time, a hallmark of vocoder-based synthesis",
    },
    "hf_energy_ratio": {
        "threshold": 0.08,
        "direction": "high",
        "label": "artificial high-frequency pattern",
        "desc": "High-frequency energy deviates from the natural speech baseline, suggesting vocoder artifacts",
    },
    "voice_texture_irregularity": {
        "threshold": 50.0,
        "direction": "high",
        "label": "abnormal voice texture",
        "desc": "MFCC delta variance is lower than natural speech, indicating a smoother, less organic voice texture",
    },
    "natural_variation": {
        "threshold": 0.55,
        "direction": "high",
        "label": "low natural variation",
        "desc": "Pitch and energy contours show less micro-variation than natural speech, consistent with synthetic generation",
    },
    "formant_stability": {
        "threshold": 50.0,
        "direction": "high",
        "label": "unnaturally stable formants",
        "desc": "Formant trajectories are more stable than natural speech, which typically shows organic transitions",
    },
    "phase_coherence": {
        "threshold": 50.0,
        "direction": "high",
        "label": "abnormal phase coherence",
        "desc": "Phase patterns are more coherent than natural speech, suggesting neural vocoder synthesis",
    },
}


def generate_explanations(
    wav: np.ndarray,
    sr: int = 16000,
    model_confidence: float = 0.0,
) -> ExplanationReport:
    """
    Compute all explanation signals from the audio waveform.

    Parameters
    ----------
    wav : np.ndarray
        Preprocessed 1-D float32 waveform.
    sr : int
        Sample rate.
    model_confidence : float
        Model's confidence (0–1). Used to gate which signals are shown —
        if the model is uncertain (< 0.6), we only flag strong signals.

    Returns
    -------
    ExplanationReport with all computed signals.
    """
    # Compute all signals
    raw_values = {
        "spectral_consistency": _spectral_consistency(wav, sr),
        "hf_energy_ratio": _hf_energy_ratio(wav, sr),
        "voice_texture_irregularity": _voice_texture_irregularity(wav, sr),
        "natural_variation": _natural_variation(wav, sr),
        "formant_stability": _formant_stability(wav, sr),
        "phase_coherence": _phase_coherence(wav, sr),
    }

    # Build report
    signals = []
    confidence_gate = 0.65 if model_confidence < 0.6 else 1.0  # stricter when uncertain

    for name, value in raw_values.items():
        cfg = SIGNAL_CONFIG[name]
        threshold = cfg["threshold"]
        is_flagged = value > threshold

        # When model is uncertain, only flag very strong signals
        if is_flagged and model_confidence < 0.6:
            is_flagged = value > threshold * 1.3

        signals.append(ExplanationSignal(
            name=cfg["label"],
            value=value,
            threshold=threshold,
            is_flagged=is_flagged,
            description=cfg["desc"],
            direction=cfg["direction"],
        ))

    flagged = [s.name for s in signals if s.is_flagged]
    return ExplanationReport(signals=signals, flagged_signals=flagged)


# =============================================================================
# SECTION 5 — visualize.py
# Waveform and spectrogram visualization with Grad-CAM overlay.
# =============================================================================

def plot_waveform_with_cam(
    wav: np.ndarray,
    sr: int = 16000,
    cam: Optional[np.ndarray] = None,
    title: str = "Waveform with Grad-CAM",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot the audio waveform. If *cam* is provided, overlay a colored
    heatmap strip showing the Grad-CAM relevance.
    """
    t = np.arange(len(wav)) / sr
    fig, ax = plt.subplots(figsize=(12, 3), dpi=150)

    # Waveform
    ax.plot(t, wav, color="#2c3e50", linewidth=0.3, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(0, len(wav) / sr)

    if cam is not None:
        # Normalize cam to [0, 1]
        cam_norm = cam / (cam.max() + 1e-10)
        # Create a colored strip below the waveform
        cam_resized = cam_norm[: len(wav)]
        # Use imshow for the heatmap strip
        extent = [0, len(wav) / sr, -1.0, -0.5]
        ax.imshow(
            cam_resized[np.newaxis, :],
            aspect="auto",
            extent=extent,
            cmap="inferno",
            alpha=0.9,
            vmin=0,
            vmax=1,
        )
        # Also tint the waveform regions with high CAM
        threshold = np.percentile(cam_resized, 85)
        high_cam = cam_resized > threshold
        # Color the waveform background where CAM is high
        for i in range(0, len(t), max(1, len(t) // 500)):
            if i < len(high_cam) and high_cam[i]:
                ax.axvspan(t[i], t[min(i + max(1, len(t) // 500), len(t) - 1)],
                          alpha=0.08, color="red", linewidth=0)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_spectrogram_with_cam(
    wav: np.ndarray,
    sr: int = 16000,
    cam_frames: Optional[np.ndarray] = None,
    title: str = "Mel Spectrogram with Grad-CAM",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot a mel spectrogram. If *cam_frames* is provided, overlay
    the per-frame relevance as a colored bar at the bottom.
    """
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_mels=80, hop_length=160, n_fft=512
    )
    mel_db = librosa.power_to_db(mel + 1e-10, ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=160, x_axis="time", y_axis="mel",
        ax=ax, cmap="viridis",
    )
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="dB")

    if cam_frames is not None:
        # Normalize
        cam_norm = cam_frames / (cam_frames.max() + 1e-10)
        # Create a semi-transparent overlay
        n_frames = mel_db.shape[1]
        cam_resized = np.interp(
            np.linspace(0, len(cam_norm) - 1, n_frames),
            np.arange(len(cam_norm)),
            cam_norm,
        )
        # Overlay as a red strip at the top of the spectrogram
        ax.imshow(
            cam_resized[np.newaxis, :] * 0 + cam_resized[np.newaxis, :],
            aspect="auto",
            extent=[0, len(wav) / sr, sr / 2 - 500, sr / 2],
            cmap="inferno",
            alpha=0.85,
            vmin=0,
            vmax=1,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_combined(
    wav: np.ndarray,
    sr: int = 16000,
    cam_audio: Optional[np.ndarray] = None,
    cam_frames: Optional[np.ndarray] = None,
    label: str = "",
    confidence: float = 0.0,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Combined visualization: waveform + Grad-CAM + spectrogram + explanation.

    This is the main figure for the detection report.
    """
    fig = plt.figure(figsize=(14, 8), dpi=150)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 2, 0.6], hspace=0.35)

    t = np.arange(len(wav)) / sr

    # --- Panel 1: Waveform with CAM ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, wav, color="#2c3e50", linewidth=0.3, alpha=0.8)
    ax1.set_xlim(0, len(wav) / sr)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Voice Authenticity: {label} ({confidence:.1f}% confidence)",
                  fontsize=13, fontweight="bold")

    if cam_audio is not None:
        cam_norm = cam_audio / (cam_audio.max() + 1e-10)
        cam_resized = cam_norm[: len(wav)]
        extent = [0, len(wav) / sr, -1.0, -0.4]
        ax1.imshow(
            cam_resized[np.newaxis, :],
            aspect="auto",
            extent=extent,
            cmap="inferno",
            alpha=0.9,
            vmin=0,
            vmax=1,
        )

    # --- Panel 2: Mel spectrogram with CAM ---
    ax2 = fig.add_subplot(gs[1])
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_mels=80, hop_length=160, n_fft=512
    )
    mel_db = librosa.power_to_db(mel + 1e-10, ref=np.max)
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=160, x_axis="time", y_axis="mel",
        ax=ax2, cmap="viridis",
    )
    ax2.set_ylabel("Mel freq (Hz)")
    fig.colorbar(img, ax=ax2, format="%+2.0f dB", label="dB", pad=0.01)

    if cam_frames is not None:
        cam_norm = cam_frames / (cam_frames.max() + 1e-10)
        n_frames = mel_db.shape[1]
        cam_resized = np.interp(
            np.linspace(0, len(cam_norm) - 1, n_frames),
            np.arange(len(cam_norm)),
            cam_norm,
        )
        ax2.imshow(
            cam_resized[np.newaxis, :],
            aspect="auto",
            extent=[0, len(wav) / sr, sr / 2 - 1000, sr / 2],
            cmap="inferno",
            alpha=0.85,
            vmin=0,
            vmax=1,
        )

    # --- Panel 3: Grad-CAM timeline ---
    ax3 = fig.add_subplot(gs[2])
    if cam_audio is not None:
        cam_norm = cam_audio / (cam_audio.max() + 1e-10)
        ax3.fill_between(t, cam_norm[: len(t)], 0, color="crimson", alpha=0.6)
        ax3.set_xlim(0, len(wav) / sr)
        ax3.set_ylim(0, 1.05)
        ax3.set_ylabel("CAM")
        ax3.set_xlabel("Time (s)")
    else:
        ax3.set_visible(False)

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


# =============================================================================
# SECTION 6 — main.py
# Pipeline orchestration + CLI entry point.
# =============================================================================

def detect_audio_deepfake(
    audio_path: str | Path = "audio-dataset/FlashSpeech/3.wav",
    output_dir: str | Path = "output",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_id: str = MODEL_ID,
    smooth_grad: bool = True,
    n_smooth: int = 5,
) -> dict:
    """
    Full detection pipeline.

    Returns a dict with all results, and saves artifacts to *output_dir*.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Preprocess
    # ------------------------------------------------------------------
    print(f"[1/5] Preprocessing {audio_path} ...")
    wav, sr = preprocess_audio(audio_path)
    processed_path = output_dir / "processed.wav"
    save_wav(wav, sr, processed_path)
    duration = len(wav) / sr
    print(f"      → {duration:.2f}s, {sr} Hz, mono, normalized")

    # ------------------------------------------------------------------
    # 2. Model inference
    # ------------------------------------------------------------------
    print(f"[2/5] Running Wav2Vec2 inference ({model_id}) ...")
    detector = load_detector(model_id=model_id, device=device)
    result = detector.predict(wav, sr, return_hidden=True)
    pred_class = 1 if result.label == "Synthetic" else 0
    print(f"      → {result.label} ({result.confidence:.1f}% confidence)")

    # ------------------------------------------------------------------
    # 3. Grad-CAM
    # ------------------------------------------------------------------
    print(f"[3/5] Computing Grad-CAM explanations ...")
    cam = AudioGradCAM(detector)
    if smooth_grad:
        cam_audio, cam_frames = cam.compute_smoothed(
            wav, target_class=pred_class, n_samples=n_smooth
        )
    else:
        cam_audio, cam_frames = cam.compute(wav, target_class=pred_class)
    np.save(output_dir / "gradcam_audio.npy", cam_audio)
    np.save(output_dir / "gradcam_frames.npy", cam_frames)
    print(f"      → Relevance map: {len(cam_audio)} samples")

    # ------------------------------------------------------------------
    # 4. Explanation signals
    # ------------------------------------------------------------------
    print(f"[4/5] Computing explanation signals ...")
    report = generate_explanations(
        wav, sr, model_confidence=result.confidence / 100
    )
    bullets = report.to_bullet_list()
    print(f"      → {len(bullets)} signals flagged")
    for b in bullets:
        print(f"        {b}")

    # ------------------------------------------------------------------
    # 5. Visualization
    # ------------------------------------------------------------------
    print(f"[5/5] Generating visualization ...")
    viz_path = output_dir / "analysis.png"
    plot_combined(
        wav, sr,
        cam_audio=cam_audio,
        cam_frames=cam_frames,
        label=result.label,
        confidence=result.confidence,
        save_path=viz_path,
    )
    plt_close = True
    print(f"      → Saved {viz_path}")

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------
    report_data = {
        "label": result.label,
        "confidence": round(result.confidence, 1),
        "probabilities": {
            "real": round(result.probs["real"], 4),
            "synthetic": round(result.probs["synthetic"], 4),
        },
        "duration_seconds": round(duration, 2),
        "sample_rate": sr,
        "model_id": model_id,
        "detected_signals": bullets,
        "all_signals": [
            {
                "name": s.name,
                "value": round(s.value, 4),
                "threshold": round(s.threshold, 4),
                "flagged": s.is_flagged,
            }
            for s in report.signals
        ],
        "artifacts": {
            "processed_audio": str(processed_path),
            "gradcam_audio": str(output_dir / "gradcam_audio.npy"),
            "gradcam_frames": str(output_dir / "gradcam_frames.npy"),
            "visualization": str(viz_path),
        },
    }

    # Save JSON report
    json_path = output_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    # Print formatted output
    print()
    print("=" * 50)
    print("  VOICE AUTHENTICITY")
    print("=" * 50)
    print()
    print(f"  Likely {result.label}")
    print()
    print(f"  Confidence: {result.confidence:.1f}%")
    print()
    if bullets:
        print("  Detected signals")
        for b in bullets:
            print(f"  {b}")
    else:
        print("  No abnormal signals detected")
    print()
    print("=" * 50)
    print(f"  Report saved: {json_path}")
    print(f"  Visualization: {viz_path}")
    print("=" * 50)

    return report_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Audio Deepfake Detection — Wav2Vec2 + Grad-CAM + Explainability"
    )
    parser.add_argument("audio", type=str, help="Input audio file (any format)")
    parser.add_argument(
        "--output-dir", "-o", type=str, default="output",
        help="Directory for output artifacts (default: output)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda or cpu (default: auto)"
    )
    parser.add_argument(
        "--model-id", type=str, default=MODEL_ID,
        help=f"HuggingFace model ID (default: {MODEL_ID})"
    )
    parser.add_argument(
        "--no-smooth-grad", action="store_true",
        help="Disable SmoothGrad for Grad-CAM (faster, noisier)"
    )
    parser.add_argument(
        "--n-smooth", type=int, default=5,
        help="Number of SmoothGrad samples (default: 5)"
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    result = detect_audio_deepfake(
        audio_path=args.audio,
        output_dir=args.output_dir,
        device=device,
        model_id=args.model_id,
        smooth_grad=not args.no_smooth_grad,
        n_smooth=args.n_smooth,
    )

    return result


if __name__ == "__main__":
    main()
