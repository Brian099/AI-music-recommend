# -*- coding: utf-8 -*-
# Written by GD Studio / Antigravity AI
# Date: 2026-08-03

import os
import sys
import uuid
import re
import json
import time
import torch
import numpy as np
from mutagen import File as MutagenFile
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel


def _ndjson(obj: dict) -> str:
    """Serialize dict as a single NDJSON line (newline-terminated)."""
    return json.dumps(obj, ensure_ascii=False) + "\n"

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
    exclude_style: Optional[bool] = False


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
    流式扫描 /music 文件夹：清理失效条目 → 提取新文件特征 → 生成向量 → 入库。
    通过 NDJSON 流推送实时进度（百分比、ETA、逐文件状态）。
    """
    if not os.path.exists(MUSIC_DIR):
        return StreamingResponse(
            iter([_ndjson({"type": "error", "message": f"Music folder '{MUSIC_DIR}' does not exist inside container."})]),
            media_type="application/x-ndjson"
        )

    async def event_stream():
        # === 阶段 1: 扫描本地音频文件 ===
        audio_files = []
        for root, dirs, files in os.walk(MUSIC_DIR):
            for file in files:
                if file.lower().endswith((".mp3", ".wav", ".flac", ".m4a", ".ogg")):
                    audio_files.append(os.path.join(root, file))

        if not audio_files:
            yield _ndjson({
                "type": "done",
                "scanned_files": 0,
                "newly_indexed": 0,
                "cleaned": 0,
                "failed": 0,
                "message": "No audio files found in music directory."
            })
            return

        # === 阶段 2: 滚动 Qdrant 已有条目，构建 path -> point_id 映射 ===
        client = QdrantClient(url=QDRANT_URL)
        existing_paths = {}  # abs_path -> point_id
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
                        existing_paths[os.path.abspath(p["local_path"])] = point.id
        except Exception as e:
            print(f"Warning: Failed to scroll existing paths: {e}")

        # === 阶段 3: 识别并清理失效条目（磁盘文件已被删除的） ===
        paths_to_delete = [
            (abs_path, point_id)
            for abs_path, point_id in existing_paths.items()
            if not os.path.exists(abs_path)
        ]

        cleaned_count = 0
        if paths_to_delete:
            yield _ndjson({
                "type": "phase",
                "phase": "cleanup",
                "total": len(paths_to_delete)
            })
            for i, (abs_path, point_id) in enumerate(paths_to_delete, 1):
                try:
                    client.delete(collection_name=COLLECTION_NAME, points_selector=[point_id])
                    cleaned_count += 1
                except Exception as e:
                    print(f"Failed to delete point {point_id}: {e}")
                yield _ndjson({
                    "type": "cleanup_progress",
                    "current": i,
                    "total": len(paths_to_delete),
                    "path": os.path.basename(abs_path)
                })

        # === 阶段 4: 过滤出待处理的新文件 ===
        files_to_process = [
            f for f in audio_files
            if os.path.abspath(f) not in existing_paths
        ]

        if not files_to_process:
            client.close()
            yield _ndjson({
                "type": "done",
                "scanned_files": len(audio_files),
                "newly_indexed": 0,
                "cleaned": cleaned_count,
                "failed": 0,
                "message": "All audio files are already indexed in Qdrant."
            })
            return

        # === 阶段 5: 加载 MLP 模型并流式提取特征 ===
        checkpoint_path = os.path.join(project_root, "checkpoints/EmbeatMLP/model.pt")
        if not os.path.exists(checkpoint_path):
            client.close()
            yield _ndjson({"type": "error", "message": "Pre-trained MLP model weights not found."})
            return

        model = load_model(checkpoint_path=checkpoint_path, device="cpu")

        total_to_process = len(files_to_process)
        yield _ndjson({
            "type": "phase",
            "phase": "extract",
            "total": total_to_process,
            "cleaned": cleaned_count
        })

        songs_rows = []
        failed_files = []
        extract_start = time.time()

        for idx, file_path in enumerate(files_to_process, 1):
            # 推送"开始处理"进度（含 ETA 估算）
            elapsed = time.time() - extract_start
            eta_seconds = int((elapsed / (idx - 1)) * (total_to_process - idx + 1)) if idx > 1 else 0
            yield _ndjson({
                "type": "progress",
                "current": idx,
                "total": total_to_process,
                "percent": round((idx - 1) * 100 / total_to_process, 1),
                "eta_seconds": eta_seconds,
                "track_name": os.path.splitext(os.path.basename(file_path))[0],
                "status": "processing"
            })

            features = extract_audio_features(file_path)
            if features is None:
                failed_files.append(file_path)
                yield _ndjson({
                    "type": "progress",
                    "current": idx,
                    "total": total_to_process,
                    "percent": round(idx * 100 / total_to_process, 1),
                    "status": "failed",
                    "track_name": os.path.basename(file_path)
                })
                continue

            # --- 优先读取内嵌元数据标签，失败则回退到文件名解析 ---
            title, artist = None, None
            try:
                audio = MutagenFile(file_path, easy=True)
                if audio is not None:
                    # easy=True 统一了 MP3/FLAC/M4A/OGG 等格式的标签键名
                    title_tags = audio.get("title") or audio.get("TIT2") or []
                    artist_tags = audio.get("artist") or audio.get("TPE1") or []
                    if title_tags:
                        title = str(title_tags[0]).strip() or None
                    if artist_tags:
                        artist = str(artist_tags[0]).strip() or None
            except Exception:
                pass

            # 回退：从文件名解析（格式："歌手 - 歌名" 或 "01. 歌手 - 歌名"）
            if not title or not artist:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                parts = base_name.split(" - ", 1)
                if len(parts) == 2:
                    fb_artist = re.sub(r'^\d+\.\s*', '', parts[0].strip())
                    fb_title = parts[1].strip()
                else:
                    fb_artist = "Unknown Artist"
                    fb_title = base_name.strip()
                if not artist:
                    artist = fb_artist
                if not title:
                    title = fb_title

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

            yield _ndjson({
                "type": "progress",
                "current": idx,
                "total": total_to_process,
                "percent": round(idx * 100 / total_to_process, 1),
                "status": "ok",
                "track_name": title,
                "artist": artist
            })

        # === 阶段 6: 批量生成嵌入向量 ===
        if not songs_rows:
            client.close()
            yield _ndjson({
                "type": "done",
                "scanned_files": len(audio_files),
                "newly_indexed": 0,
                "cleaned": cleaned_count,
                "failed": len(failed_files),
                "message": "No new audio files could be processed."
            })
            return

        yield _ndjson({"type": "phase", "phase": "embedding", "count": len(songs_rows)})

        device = next(model.parameters()).device
        computed_features = build_features(samples=songs_rows, torch_device=device)
        with torch.no_grad():
            embeddings = model(computed_features).cpu().numpy()

        # === 阶段 7: 创建集合（如不存在）并 upsert ===
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

        yield _ndjson({
            "type": "done",
            "scanned_files": len(audio_files),
            "newly_indexed": len(songs_rows),
            "cleaned": cleaned_count,
            "failed": len(failed_files),
            "message": "Scan completed and Qdrant database updated successfully."
        })

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/recommend")
async def recommend_songs(req: RecommendRequest):
    """Retrieves similar tracks for the requested seed track_id."""
    try:
        if req.exclude_style:
            # 步骤 1: 随机抽取一首完全不同风格的歌曲作为新种子
            new_seed = db.get_random_track_exclude_style(track_id=req.track_id)
            if new_seed is None:
                raise HTTPException(status_code=400, detail="无法找到不同风格的歌曲，请确认曲库中有足够多不同风格的音乐。")

            # 步骤 2: 基于新种子做正常的相似推荐
            results = db.search_entry(track_id=new_seed['track_id'], top_k=req.top_k)
            return {
                "mode": "diff_style",
                "new_seed": {
                    "track_id": new_seed['track_id'],
                    "track_name": new_seed['track_name'],
                    "artist_name": new_seed['artist_name'],
                    "artist_genre_idx": new_seed['artist_genre_idx']
                },
                "recommendations": results
            }
        else:
            # 正常推荐：直接基于原种子做相似检索
            results = db.search_entry(track_id=req.track_id, top_k=req.top_k)
            return {
                "mode": "similar",
                "recommendations": results
            }
    except HTTPException:
        raise
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
