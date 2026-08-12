# -*- coding: utf-8 -*-
# Written by GD Studio / Antigravity AI
# Date: 2026-08-04
#
# Dual-backend audio feature extractor with genre detection:
#   Linux  : Essentia MusicExtractor — full-song, higher accuracy, genre inference
#   Windows: Librosa 30-second snippet + mutagen ID3 tag genre reading

import argparse
import os
import platform
import re
import sys
import uuid
import numpy as np
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "infer"))

try:
    from infer.model_infer import load_model, build_features
except ImportError:
    pass
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
except ImportError:
    pass

# ── Import EmbeatUtils built-in genre map ─────────────────────────────────────
try:
    from infer.EmbeatUtils import build_in_genre_index_dict as _GENRE_MAP
except ImportError:
    _GENRE_MAP = {"pop": 1, "rock": 3, "hip hop": 5, "r&b": 20, "edm": 29,
                  "jazz": 379, "classical": 151, "country": 42, "folk": 166,
                  "electronic": 224, "soul": 68, "metal": 99, "reggae": 337,
                  "disco": 169, "mandopop": 220, "k-pop": 19, "j-pop": 57, "": 0}


def _genre_to_idx(genre_str: str) -> int:
    """Map a genre string to EmbeatMLP artist_genre_idx. Case-insensitive lookup."""
    if not genre_str:
        return 1  # default: pop
    g = genre_str.strip().lower()
    if g in _GENRE_MAP:
        return _GENRE_MAP[g]
    # Fuzzy fallback: find first key that contains the query word
    for key, idx in _GENRE_MAP.items():
        if key and g in key:
            return idx
    return 1  # default: pop


# ──────────────────────────────────────────────────────────────────────────────
# Backend detection
# ──────────────────────────────────────────────────────────────────────────────

_ESSENTIA_AVAILABLE = False
_LIBROSA_AVAILABLE = False
_MUTAGEN_AVAILABLE = False

if platform.system() == "Linux":
    try:
        import essentia.standard as es
        _ESSENTIA_AVAILABLE = True
        print("[Extractor] Backend: Essentia (Linux, full-song + genre inference)")
    except ImportError:
        print("[Extractor] Essentia not found, falling back to Librosa.")

if not _ESSENTIA_AVAILABLE:
    try:
        import librosa
        _LIBROSA_AVAILABLE = True
        print("[Extractor] Backend: Librosa (30-second snippet)")
    except ImportError:
        print("[Extractor] Warning: Neither Essentia nor Librosa is installed in current Python environment.")

try:
    import mutagen
    from mutagen import File as MutagenFile
    _MUTAGEN_AVAILABLE = True
except ImportError:
    pass  # mutagen is optional; used for ID3 genre reading on Windows


# ──────────────────────────────────────────────────────────────────────────────
# Note-name → Spotify integer map (C=0 … B=11)
# ──────────────────────────────────────────────────────────────────────────────
_NOTE_MAP = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


# ──────────────────────────────────────────────────────────────────────────────
# Genre heuristic shared by both backends
# ──────────────────────────────────────────────────────────────────────────────

def _infer_genre(bpm: float, loudness_norm: float, zcr: float,
                 spectral_centroid: float, key_scale: str) -> str:
    """
    Rule-based genre inference from low-level audio features.
    Returns a genre string compatible with EmbeatMLP's built-in genre map.

    Parameters
    ----------
    bpm              : tempo in BPM
    loudness_norm    : normalised RMS energy in [0, 1]
    zcr              : zero-crossing rate mean
    spectral_centroid: spectral centroid mean in Hz
    key_scale        : 'major' or 'minor'
    """
    # Electronic / EDM: fast, loud, bright
    if bpm > 126 and loudness_norm > 0.65 and spectral_centroid > 3500:
        return "edm"

    # Dance pop: upbeat, bright, moderate-high energy
    if bpm > 112 and loudness_norm > 0.5 and spectral_centroid > 2800:
        return "dance pop"

    # Hard rock / metal: very loud + high ZCR
    if bpm > 110 and loudness_norm > 0.72 and zcr > 0.10:
        return "hard rock"

    # Rock: moderately fast + loud
    if bpm > 100 and loudness_norm > 0.58 and zcr > 0.07:
        return "rock"

    # Hip hop / rap: moderate tempo + high ZCR (speech rhythm)
    if 60 < bpm < 118 and zcr > 0.12:
        return "hip hop"

    # R&B: smooth, warm (low centroid), moderate tempo
    if 60 < bpm < 110 and spectral_centroid < 2200 and loudness_norm > 0.38:
        return "r&b"

    # Classical: slow, very quiet, low ZCR
    if bpm < 85 and loudness_norm < 0.28 and zcr < 0.04:
        return "classical"

    # Jazz: slow-moderate, minor key, low-medium energy
    if bpm < 108 and key_scale == "minor" and loudness_norm < 0.48:
        return "jazz"

    # Ambient / lo-fi: very slow and quiet
    if bpm < 75 and loudness_norm < 0.35:
        return "ambient"

    # Soul: moderate tempo, warm sound
    if 70 < bpm < 110 and spectral_centroid < 2500 and loudness_norm < 0.55:
        return "soul"

    return "pop"  # universal fallback


# ──────────────────────────────────────────────────────────────────────────────
# Essentia backend (Linux) — full-song via MusicExtractor
# ──────────────────────────────────────────────────────────────────────────────

def _extract_with_essentia(audio_path: str) -> dict | None:
    """
    Full-song feature extraction using Essentia MusicExtractor.
    Analyses the entire audio file for accurate key, tempo, energy and genre.
    """
    try:
        extractor = es.MusicExtractor(
            lowlevelStats=["mean", "stdev"],
            rhythmStats=["mean", "stdev"],
            tonalStats=["mean", "stdev"],
        )
        features, _ = extractor(audio_path)
    except Exception as e:
        print(f"   [Essentia] MusicExtractor failed: {e}")
        return None

    try:
        bpm = float(features["rhythm.bpm"])
    except Exception:
        bpm = 120.0

    try:
        key_str   = features["tonal.key_key"]
        scale_str = features["tonal.key_scale"]
        best_key  = _NOTE_MAP.get(key_str, 0)
        best_mode = 1 if scale_str == "major" else 0
    except Exception:
        best_key, best_mode, scale_str = 0, 1, "major"

    try:
        # average_loudness is in [0,1] in Essentia (1 = loudest)
        loudness_norm     = float(features["lowlevel.average_loudness"])
        spectral_centroid = float(features["lowlevel.spectral_centroid.mean"])
        zcr               = float(features["lowlevel.zerocrossingrate.mean"])
    except Exception:
        loudness_norm, spectral_centroid, zcr = 0.5, 2000.0, 0.06

    # Detect time signature from beat ticks
    time_sig = 4
    try:
        ticks = features["rhythm.beats_position"]
        if len(ticks) > 4:
            intervals = np.diff(ticks)
            median_interval = float(np.median(intervals))
            beat_at_bpm = 60.0 / bpm
            if abs(median_interval - beat_at_bpm * 3) < 0.08:
                time_sig = 3
    except Exception:
        pass

    # Genre inference
    genre     = _infer_genre(bpm, loudness_norm, zcr, spectral_centroid, scale_str)
    genre_idx = _genre_to_idx(genre)

    # Normalise to EmbeatMLP feature ranges
    norm_energy           = min(max(loudness_norm, 0.0), 1.0)
    norm_acousticness     = 1.0 - norm_energy
    norm_danceability     = min(max(bpm / 180.0, 0.0), 1.0)
    norm_speechiness      = min(max(zcr * 2.0, 0.0), 0.5)
    norm_instrumentalness = 0.5 if norm_speechiness < 0.05 else 0.0
    norm_valence          = min(max(spectral_centroid / 5000.0, 0.0), 1.0)

    return {
        "key": best_key,
        "mode": best_mode,
        "tempo": int(bpm),
        "time_signature": time_sig,
        "danceability": round(norm_danceability, 3),
        "energy": round(norm_energy, 3),
        "speechiness": round(norm_speechiness, 3),
        "instrumentalness": round(norm_instrumentalness, 3),
        "valence": round(norm_valence, 3),
        "acousticness": round(norm_acousticness, 3),
        "liveness": 0.1,
        "artist_genres": genre,
        "artist_genre_idx": genre_idx,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ID3 tag reader (mutagen) — used on Windows for genre
# ──────────────────────────────────────────────────────────────────────────────

def _read_id3_metadata(audio_path: str) -> dict:
    """
    Read embedded ID3/Vorbis tags from an audio file.
    Returns a dict with keys: title, artist, album, genre (all may be None).
    """
    if not _MUTAGEN_AVAILABLE:
        return {}
    try:
        audio = MutagenFile(audio_path, easy=True)
        if audio is None:
            return {}
        def _get(tag):
            v = audio.get(tag)
            return v[0].strip() if v else None
        return {
            "title":  _get("title"),
            "artist": _get("artist"),
            "album":  _get("album"),
            "genre":  _get("genre"),
        }
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Librosa backend (Windows / fallback) — 30-second snippet
# ──────────────────────────────────────────────────────────────────────────────

def _extract_with_librosa(audio_path: str) -> dict | None:
    """
    30-second snippet feature extraction using Librosa.
    Reads genre from ID3 tags first; falls back to heuristic inference.
    """
    # 1. Try loading 30s from middle (offset 60s)
    try:
        y, sr = librosa.load(audio_path, sr=None, offset=60.0, duration=30.0)
    except Exception:
        try:
            y, sr = librosa.load(audio_path, sr=None, duration=30.0)
        except Exception as e:
            print(f"   [Librosa] Failed to load audio: {e}")
            return None

    # 2. Tempo
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        bpm = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0])
    except Exception:
        bpm = 120.0

    # 3. Key & Mode (Krumhansl-Schmuckler)
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        major_t = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                   5.19, 2.39, 3.66, 2.29, 2.88]
        minor_t = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                   4.75, 3.98, 2.69, 3.34, 3.17]
        best_key, best_mode, max_sim, key_scale = 0, 1, -1.0, "major"
        for i in range(12):
            sm = np.dot(chroma_mean, np.roll(major_t, i)) / \
                 (np.linalg.norm(chroma_mean) * np.linalg.norm(major_t) + 1e-9)
            sn = np.dot(chroma_mean, np.roll(minor_t, i)) / \
                 (np.linalg.norm(chroma_mean) * np.linalg.norm(minor_t) + 1e-9)
            if sm > max_sim:
                max_sim, best_key, best_mode, key_scale = sm, i, 1, "major"
            if sn > max_sim:
                max_sim, best_key, best_mode, key_scale = sn, i, 0, "minor"
    except Exception:
        best_key, best_mode, key_scale = 0, 1, "major"

    # 4. Energy, brightness, ZCR
    try:
        rms               = float(np.mean(librosa.feature.rms(y=y)[0]))
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))
        zcr               = float(np.mean(librosa.feature.zero_crossing_rate(y=y)[0]))
    except Exception:
        rms, spectral_centroid, zcr = 0.05, 2000.0, 0.06

    loudness_norm = min(max(rms * 5.0, 0.0), 1.0)

    # 5. Genre: try ID3 first, then heuristic
    id3 = _read_id3_metadata(audio_path)
    raw_genre = id3.get("genre") or ""
    if raw_genre:
        genre     = raw_genre.lower()
        genre_idx = _genre_to_idx(genre)
    else:
        genre     = _infer_genre(bpm, loudness_norm, zcr, spectral_centroid, key_scale)
        genre_idx = _genre_to_idx(genre)

    # 6. Normalise
    norm_energy           = loudness_norm
    norm_acousticness     = 1.0 - norm_energy
    norm_danceability     = min(max(bpm / 180.0, 0.0), 1.0)
    norm_speechiness      = min(max(zcr * 2.0, 0.0), 0.5)
    norm_instrumentalness = 0.5 if norm_speechiness < 0.05 else 0.0
    norm_valence          = min(max(spectral_centroid / 5000.0, 0.0), 1.0)

    return {
        "key": best_key,
        "mode": best_mode,
        "tempo": int(bpm),
        "time_signature": 4,
        "danceability": round(norm_danceability, 3),
        "energy": round(norm_energy, 3),
        "speechiness": round(norm_speechiness, 3),
        "instrumentalness": round(norm_instrumentalness, 3),
        "valence": round(norm_valence, 3),
        "acousticness": round(norm_acousticness, 3),
        "liveness": 0.1,
        "artist_genres": genre,
        "artist_genre_idx": genre_idx,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API — auto-selects backend
# ──────────────────────────────────────────────────────────────────────────────

def extract_audio_features(audio_path: str) -> dict | None:
    """
    Extract acoustic features + genre from a local audio file.
    Auto-selects backend:
      - Essentia  on Linux  (full-song, MusicExtractor, genre heuristic)
      - Librosa   otherwise (30-second snippet, ID3 genre → heuristic fallback)
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
    parser.add_argument("-i", "--input",      required=True,
                        help="Path to local audio file or folder.")
    parser.add_argument("-c", "--collection", default="spotify_tracks",
                        help="Qdrant collection name.")
    parser.add_argument("-q", "--qdrant-url", default="http://127.0.0.1:6333",
                        help="Qdrant database URL.")
    args = parser.parse_args()

    # 1. Scan audio files
    input_path  = os.path.abspath(args.input)
    audio_files = []
    if os.path.isfile(input_path):
        audio_files.append(input_path)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith((".mp3", ".wav", ".flac", ".m4a", ".ogg")):
                    audio_files.append(os.path.join(root, f))
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

    # 3. Extract features from each file
    songs_rows = []
    for file_path in audio_files:
        features = extract_audio_features(file_path)
        if features is None:
            continue

        # Parse artist / title from ID3 tags first, fall back to filename
        id3  = _read_id3_metadata(file_path)
        base = os.path.splitext(os.path.basename(file_path))[0]

        if id3.get("artist") and id3.get("title"):
            artist = id3["artist"]
            title  = id3["title"]
            album  = id3.get("album") or "Local Audio"
        else:
            parts  = base.split(" - ", 1)
            artist = re.sub(r"^\d+\.\s*", "", parts[0].strip()) if len(parts) == 2 else "Unknown Artist"
            title  = parts[1].strip() if len(parts) == 2 else base.strip()
            album  = "Local Audio"

        artist_genres  = features.pop("artist_genres",  "pop")
        artist_genre_idx = features.pop("artist_genre_idx", 1)

        print(f"   Genre detected: {artist_genres} (idx={artist_genre_idx})")

        row = {
            "track_id":         str(uuid.uuid5(uuid.NAMESPACE_DNS, file_path)),
            "track_name":       title,
            "artist_name":      artist,
            "artist_idx":       abs(hash(artist)) % 1000 + 1,
            "artist_genres":    artist_genres,
            "artist_genre_idx": artist_genre_idx,
            "related_artist_idxs": [],
            "album_name":       album,
            "isrc":             f"LOCAL_{abs(hash(file_path)) % 1000000}",
            "popularity":       0.5,
            "local_path":       file_path,
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
    print(f"Connecting to Qdrant at {args.qdrant_url}...")
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
            args.collection, "artist_genre_idx", qdrant_models.PayloadSchemaType.INTEGER)
        client.create_payload_index(
            args.collection, "artist_idx", qdrant_models.PayloadSchemaType.INTEGER)
        client.create_payload_index(
            args.collection, "popularity", qdrant_models.PayloadSchemaType.FLOAT)

    points = []
    for row, emb in zip(songs_rows, embeddings):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, row["track_id"]))
        payload  = {k: row[k] for k in (
            "track_id", "track_name", "popularity", "artist_name",
            "artist_idx", "artist_genres", "artist_genre_idx",
            "related_artist_idxs", "album_name", "isrc", "local_path",
        )}
        points.append(qdrant_models.PointStruct(
            id=point_id, vector=emb.tolist(), payload=payload
        ))

    print(f"Upserting {len(points)} songs into '{args.collection}'...")
    client.upsert(collection_name=args.collection, points=points)
    client.close()
    print("Done! All local audio files processed and imported to Qdrant.")


if __name__ == "__main__":
    main()
