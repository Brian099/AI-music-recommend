# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2026-04-16

import json
import os
import sys
import time
import torch
import uuid
from datasets import load_from_disk
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from typing import Any, Optional
from tqdm.auto import tqdm
from urllib.parse import urlparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from infer.model_infer import build_features, load_model

# Payload fields to keep in qdrant points
PAYLOAD_FIELD_WHITELIST = (
    "track_id",
    "track_name",
    "popularity",
    "artist_name",
    "artist_idx",
    "artist_genres",
    "artist_genre_idx",
    "related_artist_idxs",
    "album_name",
    "isrc"
)


# Map string distance name to qdrant enum
def get_qdrant_distance(distance_name: str):
    distance_name = str(distance_name).strip().lower()
    if distance_name == "cosine":
        return qdrant_models.Distance.COSINE
    if distance_name == "dot":
        return qdrant_models.Distance.DOT
    if distance_name == "euclid":
        return qdrant_models.Distance.EUCLID
    raise ValueError(f"Unsupported distance: {distance_name}")


# Map string datatype to qdrant enum
def get_qdrant_datatype(datatype_name: str):
    datatype_name = str(datatype_name).strip().lower()
    if datatype_name == "float32":
        return qdrant_models.Datatype.FLOAT32
    if datatype_name == "uint8":
        return qdrant_models.Datatype.UINT8
    raise ValueError(f"Unsupported embedding datatype: {datatype_name}")


# Build deterministic point_id from track_id
def build_point_id(track_id: Any):
    point_id = ""
    track_id_value = str(track_id or "").strip()
    if not track_id_value:
        raise ValueError("track_id is missing in row")
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, track_id_value))
    return point_id


# Build qdrant client
def build_qdrant_client(qdrant_url: str, qdrant_api_key: str = "", qdrant_timeout: float = 120.0):
    api_key_value = str(qdrant_api_key).strip()
    if api_key_value:
        client = QdrantClient(url=str(qdrant_url), api_key=api_key_value, timeout=float(qdrant_timeout))
    else:
        client = QdrantClient(url=str(qdrant_url), timeout=float(qdrant_timeout))
    return client


# Wait until qdrant is ready
def wait_for_qdrant_ready(client: QdrantClient, qdrant_url: str, wait_seconds: float = 30.0):
    wait_seconds = max(0.0, float(wait_seconds))
    if wait_seconds <= 0.0:
        return
    started = time.time()
    last_error: Optional[Exception] = None
    while True:
        try:
            _collections = client.get_collections()
            return
        except Exception as error:
            last_error = error
        elapsed = time.time() - started
        if elapsed >= wait_seconds:
            break
        time.sleep(3.0)
    raise ConnectionError(
        f"Qdrant is not reachable at {qdrant_url} after waiting {wait_seconds:.1f}s. "
        f"Last error: {repr(last_error)}"
    )


# Create or recreate collection
def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int, distance: str, embedding_datatype: str, recreate: bool, hnsw_indexing_threshold: int):
    try:
        exists = bool(client.collection_exists(collection_name=collection_name))
    except Exception:
        try:
            client.get_collection(collection_name=collection_name)
            exists = True
        except Exception:
            exists = False
    if exists and recreate:
        client.delete_collection(collection_name=collection_name)
        exists = False
    if hnsw_indexing_threshold is not None and hnsw_indexing_threshold < 0:
        hnsw_indexing_threshold = None
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=int(vector_size),
                distance=get_qdrant_distance(distance),
                datatype=get_qdrant_datatype(embedding_datatype)
            ),
            hnsw_config=qdrant_models.HnswConfigDiff(
                m=8,
                ef_construct=200
            ),
            optimizers_config=qdrant_models.OptimizersConfigDiff(
                indexing_threshold=hnsw_indexing_threshold,
            )
        )
    return


# Split list into fixed-size chunks
def chunk_list(items: list, chunk_size: int):
    chunks = []
    chunk_size = max(1, int(chunk_size))
    for begin in range(0, len(items), chunk_size):
        chunks.append(items[begin:begin + chunk_size])
    return chunks


# Convert dataset slice (List[dict]) to rows (Dict[list])
def slice_to_rows(dataset_slice: dict, row_count: int):
    rows = []
    columns = list(dataset_slice.keys())
    for i in range(int(row_count)):
        row = {col: dataset_slice[col][i] for col in columns}
        rows.append(row)
    return rows


# Load artist_genre_idx patch map from JSON
def load_artist_genre_idx_patch(patch_path: str):
    patch_file = os.path.abspath(str(patch_path).strip())
    if not os.path.isfile(patch_file):
        raise FileNotFoundError(f"artist_genre_idx patch file not found: {patch_file}")
    with open(patch_file, "r", encoding="utf-8") as f:
        patch_data = json.load(f)
    if not isinstance(patch_data, dict):
        raise ValueError(f"artist_genre_idx patch should be a JSON object: {patch_file}")
    normalized = {}
    for key, value in patch_data.items():
        patch_key = str(key).strip()
        if not patch_key:
            continue
        try:
            normalized[patch_key] = int(value)
        except Exception:
            continue
    return normalized


# Load related_artist_idx map from JSON
def load_related_artist_idx(related_path: str):
    related_file = os.path.abspath(str(related_path).strip())
    if not os.path.isfile(related_file):
        raise FileNotFoundError(f"related_artist_idx file not found: {related_file}")
    with open(related_file, "r", encoding="utf-8") as f:
        related_data = json.load(f)
    if not isinstance(related_data, dict):
        raise ValueError(f"related_artist_idx should be a JSON object: {related_file}")
    normalized = {}
    for key, value in related_data.items():
        related_key = str(key).strip()
        if not related_key:
            continue
        idx_list = []
        if isinstance(value, (list, tuple)):
            for item in value:
                try:
                    idx_list.append(int(item))
                except Exception:
                    continue
        normalized[related_key] = idx_list
    return normalized


# Patch artist_genre_idx when artist_id matches and current genre_idx == 0
def patch_artist_genre_idx(rows: list, patch_map: dict):
    if not rows or not patch_map:
        return 0
    patched = 0
    for row in rows:
        artist_id = str(row.get("artist_id") or "").strip()
        if not artist_id:
            continue
        current_idx = int(row.get("artist_genre_idx") or 0)
        if current_idx != 0:
            continue
        new_idx = patch_map.get(artist_id)
        if new_idx is not None:
            row['artist_genre_idx'] = int(new_idx)
            patched = patched + 1
    return patched


# Add related_artist_idxs field from artist_idx map
def patch_related_artist_idxs(rows: list, related_map: dict):
    if not rows:
        return 0
    patched = 0
    for row in rows:
        artist_idx = row.get("artist_idx")
        artist_idx_key = str(artist_idx).strip() if artist_idx is not None else ""
        related_values = related_map.get(artist_idx_key, []) if artist_idx_key else []
        row['related_artist_idxs'] = list(related_values)
        if related_values:
            patched = patched + 1
    return patched


# Filter rows by popularity and track_id
def filter_rows(rows: list, min_popularity: float):
    threshold = float(min_popularity)
    filtered = []
    for row in rows:
        track_id = str(row.get("track_id") or "").strip()
        if not track_id:
            continue
        popularity = float(row.get("popularity") or 0.0)
        if popularity >= threshold:
            filtered.append(row)
    return filtered


# Convert value to qdrant-safe payload value
def to_payload_value(value: Any):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if hasattr(value, "item") and callable(getattr(value, "item", None)):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [to_payload_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): to_payload_value(v) for k, v in value.items()}
    return str(value)


# Quantize embedding tensor from float32 to uint8, input range [-1.0, 1.0]
def quantize_to_uint8(embedding_tensor: torch.Tensor):
    embedding_list = []
    if embedding_tensor.ndim != 2:
        raise ValueError("embedding tensor should be 2D [batch, dim]")
    if embedding_tensor.numel() == 0:
        return []
    embedding_tensor = embedding_tensor.to(torch.float32)
    quantized = torch.round((embedding_tensor + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    embedding_list = quantized.tolist()
    return embedding_list


# Compute embeddings for a batch of rows
def compute_embeddings(model, rows: list, embedding_datatype: str = "float32"):
    if not rows:
        return []
    device = next(model.parameters()).device
    features = build_features(samples=rows, torch_device=device)
    with torch.no_grad():
        embedding_tensor = model(features)
    embedding_tensor = embedding_tensor.detach().cpu().to(torch.float32)
    if str(embedding_datatype).strip().lower() == "uint8":
        embedding_list = quantize_to_uint8(embedding_tensor)
    else:
        embedding_list = embedding_tensor.tolist()
    return embedding_list


# Build qdrant points from rows and embeddings
def build_points(rows: list, embedding_list: list, payload_include_embedding: bool = False):
    if len(rows) != len(embedding_list):
        raise RuntimeError("embedding row count mismatch")
    points = []
    for row, embedding in zip(rows, embedding_list):
        point_id = build_point_id(track_id=row.get("track_id"))
        payload = {}
        for field in PAYLOAD_FIELD_WHITELIST:
            if field in row:
                payload[field] = to_payload_value(row[field])
        if payload_include_embedding:
            payload['embedding'] = to_payload_value(embedding)
        point = qdrant_models.PointStruct(id=point_id, vector=embedding, payload=payload)
        points.append(point)
    return points


# Upsert points to qdrant in sub-batches
def upsert_points(client: QdrantClient, collection_name: str, points: list, upsert_batch_size: int = 512):
    if not points:
        return 0
    uploaded = 0
    for batch in chunk_list(points, upsert_batch_size):
        client.upsert(collection_name=collection_name, points=batch, wait=False)
        uploaded = uploaded + len(batch)
    return uploaded


# Main export entry
def main():
    dataset_path = f"{project_root}/data/datasets/spotify_45m_tracks_metadata"
    checkpoint_path = f"{project_root}/train/checkpoints/model.pt"
    collection_name = "spotify_tracks"
    qdrant_url = "http://localhost:6333"
    qdrant_storage_path = f"{project_root}/infer/qdrant_database"
    qdrant_api_key = ""
    qdrant_timeout = 120.0
    wait_qdrant_seconds = 30.0
    distance = "cosine"  # Choose between: cosine, dot, euclid
    embedding_datatype = "uint8"  # Choose between: uint8, float32
    recreate = True
    batch_size = 4096
    upsert_batch_size = 512
    hnsw_indexing_threshold = 0  # 20000 or None = Qdrant default value; 0 = no HNSW indexing
    start_index = 0
    max_rows = 0
    min_popularity = 0.01
    device = None
    strict = True
    payload_include_embedding = False
    artist_genre_idx_patch_enabled = True
    artist_genre_idx_patch_path = f"{project_root}/data/everynoise/unk_artist_genre/artist_genre_idx_patch.json"
    related_artist_idx_enabled = True
    related_artist_idx_path = f"{project_root}/data/everynoise/related_artist/related_artist_idx.json"

    print("Starting HF -> Qdrant export")
    print("dataset:", dataset_path)
    print("checkpoint:", checkpoint_path)
    print("collection:", collection_name)
    print("qdrant_url:", qdrant_url)
    print("distance:", distance)
    print("embedding_datatype:", embedding_datatype)
    print("batch_size:", batch_size)
    print("min_popularity:", min_popularity)
    print("hnsw_indexing_threshold:", hnsw_indexing_threshold)

    genre_patch_map = {}
    if artist_genre_idx_patch_enabled:
        genre_patch_map = load_artist_genre_idx_patch(artist_genre_idx_patch_path)
        print("artist_genre_idx_patch_size:", len(genre_patch_map))
    related_map = {}
    if related_artist_idx_enabled:
        related_map = load_related_artist_idx(related_artist_idx_path)
        print("related_artist_idx_size:", len(related_map))

    dataset = load_from_disk(str(dataset_path))
    total_rows = len(dataset)
    if start_index >= total_rows:
        raise ValueError(f"start_index out of range: {start_index} >= {total_rows}")
    end_index = min(total_rows, start_index + max_rows) if max_rows > 0 else total_rows
    target_rows = end_index - start_index
    print("dataset_rows:", total_rows)
    print("target_rows:", target_rows)

    model = load_model(checkpoint_path=checkpoint_path, device=device, strict=strict)
    vector_size = int(model.config.embedding_dim)
    model_device = str(next(model.parameters()).device)
    print("vector_size:", vector_size)
    print("model_device:", model_device)

    parsed_host = str(urlparse(qdrant_url).hostname or "").lower()
    if parsed_host in {"localhost", "127.0.0.1", "::1"}:
        local_path = os.path.abspath(qdrant_storage_path)
        os.makedirs(local_path, exist_ok=True)
        os.environ['QDRANT__STORAGE__STORAGE_PATH'] = local_path

    client = build_qdrant_client(qdrant_url, qdrant_api_key, qdrant_timeout)
    wait_for_qdrant_ready(client, qdrant_url, wait_qdrant_seconds)
    ensure_collection(client, collection_name, vector_size, distance, embedding_datatype, recreate, hnsw_indexing_threshold)
    print("qdrant_collection:", collection_name)

    stat_scanned = 0
    stat_filtered = 0
    stat_genre_patched = 0
    stat_related_patched = 0
    stat_kept = 0
    stat_uploaded = 0
    t1 = time.time()
    progress = tqdm(total=target_rows, desc="HF->Qdrant", unit="rows", dynamic_ncols=True)
    try:
        for begin in range(start_index, end_index, batch_size):
            batch_end = min(end_index, begin + batch_size)
            row_count = batch_end - begin
            if row_count <= 0:
                continue
            dataset_slice = dataset[begin:batch_end]
            rows = slice_to_rows(dataset_slice, row_count)
            stat_scanned = stat_scanned + len(rows)
            progress.update(row_count)
            rows = filter_rows(rows, min_popularity)
            stat_filtered = stat_filtered + row_count - len(rows)
            if not rows:
                progress.set_postfix(kept=stat_kept, filtered=stat_filtered, refresh=False)
                continue
            stat_genre_patched = stat_genre_patched + patch_artist_genre_idx(rows, genre_patch_map)
            if related_artist_idx_enabled:
                stat_related_patched = stat_related_patched + patch_related_artist_idxs(rows, related_map)
            embedding_list = compute_embeddings(model, rows, embedding_datatype)
            points = build_points(rows, embedding_list, payload_include_embedding)
            uploaded = upsert_points(client, collection_name, points, upsert_batch_size)
            stat_kept = stat_kept + len(rows)
            stat_uploaded = stat_uploaded + uploaded
            progress.set_postfix(kept=stat_kept, filtered=stat_filtered, refresh=False)
    finally:
        progress.close()
    t2 = time.time()
    print("Done")
    print("scanned_rows:", stat_scanned)
    print("kept_rows:", stat_kept)
    print("filtered_rows:", stat_filtered)
    print("genre_patched_rows:", stat_genre_patched)
    print("related_patched_rows:", stat_related_patched)
    print("uploaded_rows:", stat_uploaded)
    print("elapsed_sec:", round(float(t2 - t1), 2))


if __name__ == "__main__":
    main()
