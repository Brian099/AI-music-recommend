# -*- coding: utf-8 -*-
# Written by GD Studio / Antigravity AI
# Date: 2026-08-07
#
# True Lossless Audio & Fake Lossless (Upsampled Audio) Spectral Analyzer
# Performs STFT frequency cutoff spectral analysis to detect fake FLAC/WAV files.

import os
import math
import numpy as np
try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False
from mutagen import File as MutagenFile
from typing import Dict, Any, Optional


def analyze_audio_quality(file_path: str, duration_sec: float = 30.0) -> Dict[str, Any]:
    """
    Performs STFT spectral analysis on an audio file to determine cutoff frequency
    and detect true vs fake lossless audio.
    """
    if not os.path.exists(file_path):
        return {
            "is_true_lossless": False,
            "cutoff_freq": 0,
            "quality_rating": "File Not Found",
            "error": "File does not exist"
        }

    ext = os.path.splitext(file_path)[1].lower()
    is_lossless_container = ext in [".flac", ".wav", ".alac", ".ape"]

    try:
        # Read basic mutagen info
        audio_info = MutagenFile(file_path)
        bitrate = getattr(audio_info.info, "bitrate", 0) if audio_info and hasattr(audio_info, "info") else 0
        sample_rate = getattr(audio_info.info, "sample_rate", 44100) if audio_info and hasattr(audio_info, "info") else 44100
        
        # Target nyquist frequency
        nyquist = sample_rate / 2.0

        if not _LIBROSA_AVAILABLE:
            return {
                "is_true_lossless": is_lossless_container,
                "cutoff_freq": int(nyquist),
                "sample_rate": sample_rate,
                "bitrate": bitrate,
                "quality_rating": "True Lossless" if is_lossless_container else "Standard Audio",
                "note": "librosa not installed in runtime, using container metadata"
            }

        # Load 30 seconds of audio signal for spectral STFT analysis
        y, sr = librosa.load(file_path, sr=sample_rate, mono=True, duration=duration_sec, offset=10.0)
        if len(y) == 0:
            # Fallback to offset 0 if audio is short
            y, sr = librosa.load(file_path, sr=sample_rate, mono=True, duration=duration_sec, offset=0.0)

        if len(y) == 0:
            return {
                "is_true_lossless": is_lossless_container,
                "cutoff_freq": int(nyquist),
                "quality_rating": "Lossless Container" if is_lossless_container else "Standard Audio",
                "error": "Empty audio buffer"
            }

        # STFT Spectrogram calculation
        n_fft = 2048
        hop_length = 512
        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        
        # Calculate mean magnitude spectrum across time frames
        mean_spectrum = np.mean(stft, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Convert to dB relative to max magnitude
        max_mag = np.max(mean_spectrum) + 1e-10
        db_spectrum = 20 * np.log10(mean_spectrum / max_mag)

        # Find brickwall cutoff frequency (where energy drops below threshold e.g. -45 dB)
        cutoff_freq = int(nyquist)
        # Search backwards from nyquist frequency down to 10kHz
        for i in range(len(db_spectrum) - 1, 0, -1):
            if db_spectrum[i] > -42.0:  # Threshold for active musical harmonics
                cutoff_freq = int(freqs[i])
                break

        # Rating logic
        is_true_lossless = False
        quality_rating = "Lossy MP3/AAC"

        if is_lossless_container:
            if cutoff_freq >= 20500:
                is_true_lossless = True
                quality_rating = f"🟢 True Lossless (真无损 {cutoff_freq/1000:.1f}kHz)"
            elif 18500 <= cutoff_freq < 20500:
                is_true_lossless = False
                quality_rating = f"🟡 Fake FLAC 20kHz Cutoff (320k假无损)"
            elif cutoff_freq < 18500:
                is_true_lossless = False
                quality_rating = f"🔴 Fake FLAC {cutoff_freq/1000:.1f}kHz Cutoff (128k严重假无损)"
        else:
            if bitrate >= 320000 or cutoff_freq >= 19500:
                quality_rating = f"🟢 High Quality 320k ({cutoff_freq/1000:.1f}kHz)"
            elif bitrate >= 192000:
                quality_rating = f"🟡 Medium Quality MP3 ({cutoff_freq/1000:.1f}kHz)"
            else:
                quality_rating = f"🔴 Low Bitrate MP3 ({cutoff_freq/1000:.1f}kHz)"

        return {
            "is_true_lossless": is_true_lossless,
            "cutoff_freq": cutoff_freq,
            "sample_rate": sample_rate,
            "bitrate": bitrate,
            "quality_rating": quality_rating
        }

    except Exception as e:
        return {
            "is_true_lossless": is_lossless_container,
            "cutoff_freq": 22050 if is_lossless_container else 18000,
            "quality_rating": "True Lossless" if is_lossless_container else "Standard Audio",
            "error": str(e)
        }
