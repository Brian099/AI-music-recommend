# -*- coding: utf-8 -*-
# Written by GD Studio / Antigravity AI
# Date: 2026-08-03

import argparse
import os
import sys
import uuid
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "infer"))

try:
    import librosa
except ImportError:
    print("Error: `librosa` is not installed. Please run: pip install librosa soundfile")
    sys.exit(1)

from infer.infer import load_model, build_features
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models


def extract_audio_features(audio_path: str):
    """
    Extract key, mode, tempo, energy and estimate other features offline
    using a 30-second snippet of the audio to maximize performance.
    """
    print(f"-> Analyzing: {os.path.basename(audio_path)}")
    try:
        # Load 30 seconds from the middle of the song (offset 60s) to speed up loading
        y, sr = librosa.load(audio_path, sr=None, offset=60.0, duration=30.0)
    except Exception as e:
        # Fallback to load from start if song is shorter than 60s
        try:
            y, sr = librosa.load(audio_path, sr=None, duration=30.0)
        except Exception as err:
            print(f"   Failed to load audio: {err}")
            return None

    # 1. Extract Tempo (BPM)
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo_array = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        tempo = float(tempo_array[0])
    except Exception:
        tempo = 120.0

    # 2. Extract Key & Mode (Chroma-based Key Detection)
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Standard Krumhansl-Schmuckler profiles for Major and Minor
        major_template = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_template = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        best_key = 0
        best_mode = 1
        max_sim = -1.0
        for i in range(12):
            shifted_major = np.roll(major_template, i)
            shifted_minor = np.roll(minor_template, i)
            sim_maj = np.dot(chroma_mean, shifted_major) / (np.linalg.norm(chroma_mean) * np.linalg.norm(shifted_major))
            sim_min = np.dot(chroma_mean, shifted_minor) / (np.linalg.norm(chroma_mean) * np.linalg.norm(shifted_minor))

            if sim_maj > max_sim:
                max_sim = sim_maj
                best_key = i
                best_mode = 1
            if sim_min > max_sim:
                max_sim = sim_min
                best_key = i
                best_mode = 0
    except Exception:
        best_key = 0
        best_mode = 1

    # 3. Extract continuous acoustic features (Energy & estimated metrics)
    try:
        rms = librosa.feature.rms(y=y)[0]
        energy = float(np.mean(rms))
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        brightness = float(np.mean(spectral_centroids))
        zero_cross = librosa.feature.zero_crossing_rate(y=y)[0]
        zcr = float(np.mean(zero_cross))
    except Exception:
        energy = 0.1
        brightness = 2000.0
        zcr = 0.05

    # Scale/Normalize heuristics to align with Spotify API's feature distribution
    norm_energy = min(max(energy * 5.0, 0.0), 1.0)
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
        "time_signature": 4,
        "danceability": round(norm_danceability, 3),
        "energy": round(norm_energy, 3),
        "speechiness": round(norm_speechiness, 3),
        "instrumentalness": round(norm_instrumentalness, 3),
        "valence": round(norm_valence, 3),
        "acousticness": round(norm_acousticness, 3),
        "liveness": round(norm_liveness, 3)
    }


def main():
    parser = argparse.ArgumentParser(description="Extract acoustic features offline and import to Qdrant.")
    parser.add_argument("-i", "--input", required=True, help="Path to local audio file or folder.")
    parser.add_argument("-c", "--collection", default="spotify_tracks", help="Qdrant collection name.")
    parser.add_argument("-q", "--qdrant-url", default="http://127.0.0.1:6333", help="Qdrant database URL.")
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

    # 3. Process and extract features
    songs_rows = []
    for file_path in audio_files:
        features = extract_audio_features(file_path)
        if features is None:
            continue

        # Guess song name and artist from filename (assuming format: Artist - Title.mp3)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        parts = base_name.split(" - ", 1)
        if len(parts) == 2:
            artist = parts[0].strip()
            title = parts[1].strip()
        else:
            artist = "Unknown Artist"
            title = base_name.strip()

        row = {
            "track_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, file_path)),
            "track_name": title,
            "artist_name": artist,
            "artist_idx": abs(hash(artist)) % 1000 + 1,
            "artist_genres": "pop",  # default
            "artist_genre_idx": 1,   # default
            "related_artist_idxs": [],
            "album_name": "Local Audio",
            "isrc": f"LOCAL_{abs(hash(file_path)) % 1000000}",
            "popularity": 0.5,
            "local_path": file_path, # store filepath in Qdrant
            **features
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

    # 5. Connect and upload to Qdrant
    print(f"Connecting to Qdrant server at {args.qdrant_url}...")
    client = QdrantClient(url=args.qdrant_url)

    # Recreate or ensure collection
    if not client.collection_exists(args.collection):
        client.create_collection(
            collection_name=args.collection,
            vectors_config=qdrant_models.VectorParams(
                size=64,
                distance=qdrant_models.Distance.COSINE,
                datatype=qdrant_models.Datatype.FLOAT32
            )
        )
        # Create necessary indexes
        client.create_payload_index(args.collection, "artist_genre_idx", qdrant_models.PayloadSchemaType.INTEGER)
        client.create_payload_index(args.collection, "artist_idx", qdrant_models.PayloadSchemaType.INTEGER)
        client.create_payload_index(args.collection, "popularity", qdrant_models.PayloadSchemaType.FLOAT)

    points = []
    for row, emb in zip(songs_rows, embeddings):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, row['track_id']))
        payload = {
            "track_id": row["track_id"],
            "track_name": row["track_name"],
            "popularity": row["popularity"],
            "artist_name": row["artist_name"],
            "artist_idx": row["artist_idx"],
            "artist_genres": row["artist_genres"],
            "artist_genre_idx": row["artist_genre_idx"],
            "related_artist_idxs": row["related_artist_idxs"],
            "album_name": row["album_name"],
            "isrc": row["isrc"],
            "local_path": row["local_path"]
        }
        point = qdrant_models.PointStruct(id=point_id, vector=emb.tolist(), payload=payload)
        points.append(point)

    print(f"Upserting {len(points)} songs into collection '{args.collection}'...")
    client.upsert(collection_name=args.collection, points=points)
    client.close()
    print("Done! All local audio files successfully processed and imported to Qdrant.")


if __name__ == "__main__":
    main()
