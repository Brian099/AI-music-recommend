# -*- coding: utf-8 -*-
# Written by GD Studio / Antigravity AI
# Date: 2026-08-04
# Dual-backend audio feature extractor:
#   - Linux: Essentia (full-song analysis, higher accuracy)
#   - Windows/fallback: Librosa (30-second snippet, cross-platform)

import argparse
import os
import platform
import re
import sys
import uuid
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "infer"))

from infer.infer import load_model, build_features
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# ──────────────────────────────────────────────────────────────────────────────
# Backend detection
# ──────────────────────────────────────────────────────────────────────────────

_ESSENTIA_AVAILABLE = False
_LIBROSA_AVAILABLE = False

if platform.system() == "Linux":
    try:
        import essentia.standard as es
        _ESSENTIA_AVAILABLE = True
        print("[Extractor] Backend: Essentia (Linux, full-song analysis)")
    except ImportError:
        print("[Extractor] Essentia not available, falling back to Librosa.")

if not _ESSENTIA_AVAILABLE:
    try:
        import librosa
        _LIBROSA_AVAILABLE = True
        print("[Extractor] Backend: Librosa (30-second snippet analysis)")
    except ImportError:
        print("Error: Neither Essentia nor Librosa is installed.")
        print("  Linux:   pip install essentia")
        print("  Windows: pip install librosa soundfile")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Essentia backend (Linux) — full-song, higher accuracy
# ──────────────────────────────────────────────────────────────────────────────

def _extract_with_essentia(audio_path: str) -> dict | None:
    """
    Full-song feature extraction using Essentia.
    Covers the entire audio for more accurate key, mode, tempo, and energy.
    """
    try:
        # Load full mono audio at 44100 Hz (Essentia standard)
        loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
        audio = loader()
    except Exception as e:
        print(f"   [Essentia] Failed to load audio: {e}")
        return None

    try:
        # ── Rhythm / Tempo ────────────────────────────────────────────────────
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, _, _, _, _ = rhythm_extractor(audio)
        tempo = float(bpm)
    except Exception:
        tempo = 120.0

    try:
        # ── Key / Mode ────────────────────────────────────────────────────────
        key_extractor = es.KeyExtractor()
        key_str, scale_str, key_strength = key_extractor(audio)
        # Map note name to Spotify integer (C=0 … B=11)
        _NOTE_MAP = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
                     "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
                     "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
        best_key = _NOTE_MAP.get(key_str, 0)
        best_mode = 1 if scale_str == "major" else 0
    except Exception:
        best_key = 0
        best_mode = 1

    try:
        # ── Energy / RMS ─────────────────────────────────────────────────────
        rms_extractor = es.RMS()
        rms = float(rms_extractor(audio))

        # ── Spectral Centroid (brightness → valence proxy) ────────────────────
        w = es.Windowing(type="hann")
        spectrum = es.Spectrum()
        centroid_extractor = es.Centroid(range=22050)
        centroids = []
        for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=512):
            spec = spectrum(w(frame))
            centroids.append(centroid_extractor(spec))
        brightness = float(np.mean(centroids)) if centroids else 2000.0

        # ── Zero-Crossing Rate (speechiness proxy) ────────────────────────────
        zcr_extractor = es.ZeroCrossingRate()
        zcrs = [zcr_extractor(frame)
                for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=512)]
        zcr = float(np.mean(zcrs)) if zcrs else 0.05

        # ── Time signature (Beat histogram peak) ─────────────────────────────
        beat_tracker = es.BeatTrackerMultiFeature()
        ticks, _ = beat_tracker(audio)
        # Simple heuristic: 3/4 if tempo sits near waltz range
        time_sig = 3 if 100 < tempo < 160 and len(ticks) > 4 and \
            abs(np.median(np.diff(ticks)) - (60.0 / tempo) * 3) < 0.1 else 4

    except Exception:
        rms = 0.05
        brightness = 2000.0
        zcr = 0.05
        time_sig = 4

    # ── Normalise to [0, 1] using same heuristics as Librosa backend ──────────
    norm_energy = min(max(rms * 5.0, 0.0), 1.0)
    norm_acousticness = 1.0 - norm_energy
    norm_danceability = min(max(tempo / 180.0, 0.0), 1.0)
    norm_speechiness = min(max(zcr * 2.0, 0.0), 0.5)
    norm_instrumentalness = 0.5 if norm_speechiness < 0.05 else 0.0
    norm_valence = min(max(brightness / 5000.0, 0.0), 1.0)
    norm_liveness = 0.1

    return {
        "key": best_key,
        "mode": best_mode,
        "tempo": int(tempo),
        "time_signature": time_sig,
        "danceability": round(norm_danceability, 3),
        "energy": round(norm_energy, 3),
        "speechiness": round(norm_speechiness, 3),
        "instrumentalness": round(norm_instrumentalness, 3),
        "valence": round(norm_valence, 3),
        "acousticness": round(norm_acousticness, 3),
        "liveness": round(norm_liveness, 3),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Librosa backend (Windows / fallback) — 30-second snippet
# ──────────────────────────────────────────────────────────────────────────────

def _extract_with_librosa(audio_path: str) -> dict | None:
    """
    30-second snippet feature extraction using Librosa.
    Cross-platform fallback for Windows or when Essentia is unavailable.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, offset=60.0, duration=30.0)
    except Exception:
        try:
            y, sr = librosa.load(audio_path, sr=None, duration=30.0)
        except Exception as e:
            print(f"   [Librosa] Failed to load audio: {e}")
            return None

    # Tempo
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0])
    except Exception:
        tempo = 120.0

    # Key & Mode — Krumhansl-Schmuckler
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        major_t = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_t = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        best_key, best_mode, max_sim = 0, 1, -1.0
        for i in range(12):
            sm = np.dot(chroma_mean, np.roll(major_t, i)) / \
                 (np.linalg.norm(chroma_mean) * np.linalg.norm(major_t))
            sn = np.dot(chroma_mean, np.roll(minor_t, i)) / \
                 (np.linalg.norm(chroma_mean) * np.linalg.norm(minor_t))
            if sm > max_sim:
                max_sim, best_key, best_mode = sm, i, 1
            if sn > max_sim:
                max_sim, best_key, best_mode = sn, i, 0
    except Exception:
        best_key, best_mode = 0, 1

    # Energy, brightness, ZCR
    try:
        rms = float(np.mean(librosa.feature.rms(y=y)[0]))
        brightness = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)[0]))
    except Exception:
        rms, brightness, zcr = 0.05, 2000.0, 0.05

    norm_energy = min(max(rms * 5.0, 0.0), 1.0)
    norm_acousticness = 1.0 - norm_energy
    norm_danceability = min(max(tempo / 180.0, 0.0), 1.0)
    norm_speechiness = min(max(zcr * 2.0, 0.0), 0.5)
    norm_instrumentalness = 0.5 if norm_speechiness < 0.05 else 0.0
    norm_valence = min(max(brightness / 5000.0, 0.0), 1.0)

    return {
        "key": best_key,
        "mode": best_mode,
        "tempo": int(tempo),
        "time_signature": 4,
        "danceability": round(norm_danceability, 3),
        "energy": round(norm_energy, 3),
        "speechiness": round(norm_speechiness, 3),
        "instrumentalness": round(norm_instrumentalness, 3),
        "valence": round(norm_valence, 3),
        "acousticness": round(norm_acousticness, 3),
        "liveness": 0.1,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API — auto-selects backend
# ──────────────────────────────────────────────────────────────────────────────

def extract_audio_features(audio_path: str) -> dict | None:
    """
    Extract acoustic features from a local audio file.
    Automatically selects the best available backend:
      - Essentia on Linux  (full-song, higher accuracy)
      - Librosa on Windows (30-second snippet, cross-platform)
    Returns a dict compatible with the EmbeatMLP input format, or None on error.
    """
    print(f"-> Analyzing: {os.path.basename(audio_path)}")
    if _ESSENTIA_AVAILABLE:
        return _extract_with_essentia(audio_path)
    return _extract_with_librosa(audio_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract acoustic features offline and import to Qdrant."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Path to local audio file or folder.")
    parser.add_argument("-c", "--collection", default="spotify_tracks",
                        help="Qdrant collection name.")
    parser.add_argument("-q", "--qdrant-url", default="http://127.0.0.1:6333",
                        help="Qdrant database URL.")
    args = parser.parse_args()

    # 1. Scan files
    input_path = os.path.abspath(args.input)
    audio_files = []
    if os.path.isfile(input_path):
        audio_files.append(input_path)
    elif os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith((".mp3", ".wav", ".flac", ".m4a", ".ogg")):
                    audio_files.append(os.path.join(root, file))
    else:
        print(f"Error: Input path '{args.input}' does not exist.")
        sys.exit(1)

    if not audio_files:
        print("No audio files found.")
        sys.exit(0)

    print(f"Found {len(audio_files)} audio files to process.")

    # 2. Load EmbeatMLP model
    checkpoint_path = os.path.join(project_root, "checkpoints/EmbeatMLP/model.pt")
    if not os.path.isfile(checkpoint_path):
        print(f"Error: Pre-trained model not found at {checkpoint_path}")
        sys.exit(1)

    print("Loading pre-trained EmbeatMLP model...")
    model = load_model(checkpoint_path=checkpoint_path, device="cpu")

    # 3. Extract features
    songs_rows = []
    for file_path in audio_files:
        features = extract_audio_features(file_path)
        if features is None:
            continue

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        parts = base_name.split(" - ", 1)
        if len(parts) == 2:
            artist = re.sub(r"^\d+\.\s*", "", parts[0].strip())
            title = parts[1].strip()
        else:
            artist = "Unknown Artist"
            title = base_name.strip()

        row = {
            "track_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, file_path)),
            "track_name": title,
            "artist_name": artist,
            "artist_idx": abs(hash(artist)) % 1000 + 1,
            "artist_genres": "pop",
            "artist_genre_idx": 1,
            "related_artist_idxs": [],
            "album_name": "Local Audio",
            "isrc": f"LOCAL_{abs(hash(file_path)) % 1000000}",
            "popularity": 0.5,
            "local_path": file_path,
            **features,
        }
        songs_rows.append(row)

    if not songs_rows:
        print("No audio features could be extracted.")
        sys.exit(0)

    # 4. Generate embeddings
    print("Generating embeddings using EmbeatMLP...")
    device = next(model.parameters()).device
    computed_features = build_features(samples=songs_rows, torch_device=device)
    with torch.no_grad():
        embeddings = model(computed_features).cpu().numpy()

    # 5. Upload to Qdrant
    print(f"Connecting to Qdrant server at {args.qdrant_url}...")
    client = QdrantClient(url=args.qdrant_url)

    if not client.collection_exists(args.collection):
        client.create_collection(
            collection_name=args.collection,
            vectors_config=qdrant_models.VectorParams(
                size=64,
                distance=qdrant_models.Distance.COSINE,
                datatype=qdrant_models.Datatype.FLOAT32,
            ),
        )
        client.create_payload_index(
            args.collection, "artist_genre_idx", qdrant_models.PayloadSchemaType.INTEGER
        )
        client.create_payload_index(
            args.collection, "artist_idx", qdrant_models.PayloadSchemaType.INTEGER
        )
        client.create_payload_index(
            args.collection, "popularity", qdrant_models.PayloadSchemaType.FLOAT
        )

    points = []
    for row, emb in zip(songs_rows, embeddings):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, row["track_id"]))
        payload = {k: row[k] for k in (
            "track_id", "track_name", "popularity", "artist_name",
            "artist_idx", "artist_genres", "artist_genre_idx",
            "related_artist_idxs", "album_name", "isrc", "local_path",
        )}
        points.append(qdrant_models.PointStruct(
            id=point_id, vector=emb.tolist(), payload=payload
        ))

    print(f"Upserting {len(points)} songs into collection '{args.collection}'...")
    client.upsert(collection_name=args.collection, points=points)
    client.close()
    print("Done! All local audio files successfully processed and imported to Qdrant.")


if __name__ == "__main__":
    main()
