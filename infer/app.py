# -*- coding: utf-8 -*-
# Written by GD Studio / Antigravity AI
# Date: 2026-08-03

import os
import sys
import uuid
import re
import torch
import numpy as np
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "infer"))

from infer.Embeat import EmbeatDatabase
from infer.offline_extractor import extract_audio_features
from infer.infer import load_model, build_features
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

app = FastAPI(title="Embeat Music Recommendation API Service")

# Load environment configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
MUSIC_DIR = os.getenv("MUSIC_DIR", "/music")
COLLECTION_NAME = "spotify_tracks"

# Initialize recommendation engine
print(f"Connecting to Qdrant at {QDRANT_URL}...")
db = EmbeatDatabase(qdrant_url=QDRANT_URL, collection_name=COLLECTION_NAME, verbose_log=True)


class RecommendRequest(BaseModel):
    track_id: str
    top_k: Optional[int] = 10


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the front-end testing UI."""
    index_path = os.path.join(project_root, "infer", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/songs")
async def list_songs():
    """Scrolls and lists all indexed songs from Qdrant database."""
    try:
        client = QdrantClient(url=QDRANT_URL)
        # Return empty list if collection hasn't been created yet (fresh deployment)
        if not client.collection_exists(COLLECTION_NAME):
            client.close()
            return []
        result, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        songs = []
        for point in result:
            payload = point.payload or {}
            songs.append({
                "track_id": payload.get("track_id"),
                "track_name": payload.get("track_name"),
                "artist_name": payload.get("artist_name"),
                "local_path": payload.get("local_path", "")
            })
        client.close()
        return songs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch songs from Qdrant: {str(e)}")


@app.post("/scan")
async def scan_music():
    """
    Scans the mounted /music folder, extracts acoustic features for new tracks,
    generates 64D embeddings, and upserts them to Qdrant database.
    """
    if not os.path.exists(MUSIC_DIR):
        raise HTTPException(status_code=400, detail=f"Music folder '{MUSIC_DIR}' does not exist inside container.")

    # 1. Scan local files
    audio_files = []
    for root, dirs, files in os.walk(MUSIC_DIR):
        for file in files:
            if file.lower().endswith((".mp3", ".wav", ".flac", ".m4a", ".ogg")):
                audio_files.append(os.path.join(root, file))

    if not audio_files:
        return {"scanned_files": 0, "message": "No audio files found in music directory."}

    # 2. Get existing local_paths in Qdrant to avoid duplicate extraction
    client = QdrantClient(url=QDRANT_URL)
    existing_paths = set()
    try:
        if client.collection_exists(COLLECTION_NAME):
            result, _ = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            for point in result:
                p = point.payload or {}
                if p.get("local_path"):
                    # Normalize path format
                    existing_paths.add(os.path.abspath(p["local_path"]))
    except Exception as e:
        print(f"Warning: Failed to scroll existing paths: {e}")

    # 3. Filter out already indexed files
    files_to_process = []
    for f in audio_files:
        abs_f = os.path.abspath(f)
        if abs_f not in existing_paths:
            files_to_process.append(f)

    if not files_to_process:
        client.close()
        return {"scanned_files": len(audio_files), "message": "All audio files are already indexed in Qdrant."}

    # 4. Extract features for new files
    checkpoint_path = os.path.join(project_root, "checkpoints/EmbeatMLP/model.pt")
    if not os.path.exists(checkpoint_path):
        client.close()
        raise HTTPException(status_code=500, detail="Pre-trained MLP model weights not found.")

    model = load_model(checkpoint_path=checkpoint_path, device="cpu")
    songs_rows = []
    for file_path in files_to_process:
        features = extract_audio_features(file_path)
        if features is None:
            continue

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        parts = base_name.split(" - ", 1)
        if len(parts) == 2:
            # Check for leading song number prefix like '01. '
            artist = parts[0].strip()
            title = parts[1].strip()
            # Clean up leading track number e.g. "01. 周杰伦" -> "周杰伦"
            artist = re.sub(r'^\d+\.\s*', '', artist)
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
            **features
        }
        songs_rows.append(row)

    if not songs_rows:
        client.close()
        return {"scanned_files": len(audio_files), "message": "No new audio files could be processed."}

    # 5. Generate embeddings
    device = next(model.parameters()).device
    computed_features = build_features(samples=songs_rows, torch_device=device)
    with torch.no_grad():
        embeddings = model(computed_features).cpu().numpy()

    # 6. Upload to Qdrant
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(
                size=64,
                distance=qdrant_models.Distance.COSINE,
                datatype=qdrant_models.Datatype.FLOAT32
            )
        )
        client.create_payload_index(COLLECTION_NAME, "artist_genre_idx", qdrant_models.PayloadSchemaType.INTEGER)
        client.create_payload_index(COLLECTION_NAME, "artist_idx", qdrant_models.PayloadSchemaType.INTEGER)
        client.create_payload_index(COLLECTION_NAME, "popularity", qdrant_models.PayloadSchemaType.FLOAT)

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

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    client.close()

    return {
        "scanned_files": len(audio_files),
        "newly_indexed_files": len(songs_rows),
        "message": "Scan completed and Qdrant database updated successfully."
    }


@app.post("/recommend")
async def recommend_songs(req: RecommendRequest):
    """Retrieves similar tracks for the requested seed track_id."""
    try:
        results = db.search_entry(track_id=req.track_id, top_k=req.top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing recommendation: {str(e)}")


@app.get("/audio/{track_id}")
async def stream_audio(track_id: str, range: Optional[str] = Header(None)):
    """
    Streams audio files directly using range requests for scrubbability in audio players.
    Applies adaptive path resolution to bridge Windows paths and Linux container directories.
    """
    # 1. Look up song metadata from Qdrant database
    client = QdrantClient(url=QDRANT_URL)
    target_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(track_id)))
    try:
        records = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[target_uuid],
            with_payload=True,
            with_vectors=False
        )
    except Exception as e:
        client.close()
        raise HTTPException(status_code=500, detail=f"Database lookup error: {str(e)}")
    finally:
        client.close()

    if not records:
        raise HTTPException(status_code=404, detail="Track ID not found in database.")

    payload = records[0].payload or {}
    local_path = payload.get("local_path", "")

    # 2. Path-agnostic file resolution
    actual_file_path = None
    if local_path and os.path.exists(local_path):
        actual_file_path = local_path
    else:
        # Resolve path mismatch (Windows host dev vs Linux container run)
        filename = os.path.basename(local_path) if local_path else f"{track_id}.mp3"
        # Search directly in the mounted /music folder
        possible_path = os.path.join(MUSIC_DIR, filename)
        if os.path.exists(possible_path):
            actual_file_path = possible_path
        else:
            # Perform depth-first folder scan
            for root, dirs, files in os.walk(MUSIC_DIR):
                if filename in files:
                    actual_file_path = os.path.join(root, filename)
                    break

    if not actual_file_path:
        raise HTTPException(
            status_code=404, 
            detail=f"Audio file '{filename if 'filename' in locals() else local_path}' not found in {MUSIC_DIR}."
        )

    # 3. Stream audio with byte-range headers
    file_size = os.path.getsize(actual_file_path)
    
    # Check range headers for scrubbable playbacks
    if range:
        match = re.match(r"bytes=(\d+)-(\d*)", range)
        if match:
            start = int(match.group(1))
            end = match.group(2)
            end = int(end) if end else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1
            
            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
            }
            
            def file_iterator():
                with open(actual_file_path, "rb") as f:
                    f.seek(start)
                    bytes_left = length
                    while bytes_left > 0:
                        chunk_size = min(bytes_left, 1024 * 1024)  # 1MB chunk sizes
                        data = f.read(chunk_size)
                        if not data:
                            break
                        bytes_left -= len(data)
                        yield data
                        
            # Return HTTP 206 Partial Content response
            return StreamingResponse(file_iterator(), status_code=206, headers=headers, media_type="audio/mpeg")

    # Regular stream fallback
    def full_file_iterator():
        with open(actual_file_path, "rb") as f:
            while True:
                data = f.read(1024 * 1024)
                if not data:
                    break
                yield data

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
    }
    return StreamingResponse(full_file_iterator(), headers=headers, media_type="audio/mpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
