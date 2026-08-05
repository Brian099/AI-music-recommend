# -*- coding: utf-8 -*-
# Written by GD Studio / Antigravity AI
# Date: 2026-08-03

import os
import sys
import uuid
import re
import json
import time
import asyncio
import torch
import numpy as np
from mutagen import File as MutagenFile
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Header, Query
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


STATUS_FILE_PATH = os.path.join(project_root, "data", "scan_status.json")


class ScanTaskManager:
    def __init__(self):
        self.is_running = False
        self.cancel_requested = False
        self.task: Optional[asyncio.Task] = None
        self.listeners: list[asyncio.Queue] = []
        self.status = {
            "is_running": False,
            "phase": "idle",
            "current": 0,
            "total": 0,
            "percent": 0.0,
            "eta_seconds": 0,
            "cleaned": 0,
            "failed": 0,
            "scanned_files": 0,
            "newly_indexed": 0,
            "message": "",
            "last_updated": 0.0
        }
        self.logs: list[str] = []
        self.load_from_disk()

    def load_from_disk(self):
        try:
            if os.path.exists(STATUS_FILE_PATH):
                with open(STATUS_FILE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.status = data.get("status", self.status)
                    self.status["is_running"] = False
                    self.logs = data.get("logs", [])[-200:]
        except Exception as e:
            print(f"[ScanManager] Failed to load status from disk: {e}")

    def save_to_disk(self):
        try:
            os.makedirs(os.path.dirname(STATUS_FILE_PATH), exist_ok=True)
            with open(STATUS_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "status": self.status,
                    "logs": self.logs[-200:]
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ScanManager] Failed to save status to disk: {e}")

    def add_log(self, text: str):
        self.logs.append(text)
        if len(self.logs) > 300:
            self.logs = self.logs[-200:]

    async def broadcast(self, event: dict):
        event_type = event.get("type")
        if event_type == "phase":
            self.status["phase"] = event.get("phase", "")
            if event.get("phase") == "cleanup":
                self.add_log(f"-> 阶段: 清理失效条目 (共 {event.get('total', 0)} 个)")
            elif event.get("phase") == "extract":
                self.add_log(f"-> 阶段: 提取音频特征 (共 {event.get('total', 0)} 个新文件)")
            elif event.get("phase") == "embedding":
                self.add_log(f"-> 阶段: 批量生成向量并入库 ({event.get('count', 0)} 首)")
        elif event_type == "progress":
            self.status["current"] = event.get("current", 0)
            self.status["total"] = event.get("total", 0)
            self.status["percent"] = event.get("percent", 0.0)
            self.status["eta_seconds"] = event.get("eta_seconds", 0)
            track_name = event.get("track_name", "")
            status_str = event.get("status", "")
            if status_str == "processing":
                self.add_log(f"-> [{event.get('current')}/{event.get('total')}] 处理中: {track_name}")
            elif status_str == "ok":
                artist = event.get("artist", "")
                self.add_log(f"   ✓ {track_name} - {artist or '未知歌手'}")
            elif status_str == "failed":
                self.add_log(f"   ✗ 失败: {track_name}")
        elif event_type == "cleanup_progress":
            self.add_log(f"   🧹 清理失效条目: {event.get('path', '')}")
        elif event_type == "done":
            self.status["is_running"] = False
            self.status["phase"] = "done"
            self.status["scanned_files"] = event.get("scanned_files", 0)
            self.status["newly_indexed"] = event.get("newly_indexed", 0)
            self.status["cleaned"] = event.get("cleaned", 0)
            self.status["failed"] = event.get("failed", 0)
            self.status["message"] = event.get("message", "")
            self.add_log(f"\n-> 扫描完成: {event.get('message', '')}")
            self.add_log(f"   - 扫描文件: {event.get('scanned_files', 0)}, 新增入库: {event.get('newly_indexed', 0)}, 清理失效: {event.get('cleaned', 0)}, 失败: {event.get('failed', 0)}")
        elif event_type == "aborted":
            self.status["is_running"] = False
            self.status["phase"] = "aborted"
            self.status["message"] = "扫描已被用户手动中止。"
            self.add_log(f"\n🛑 扫描任务已被用户手动中止。")
        elif event_type == "error":
            self.status["is_running"] = False
            self.status["phase"] = "error"
            self.status["message"] = event.get("message", "")
            self.add_log(f"\n❌ 出错: {event.get('message', '')}")

        self.status["last_updated"] = time.time()
        self.save_to_disk()

        line = _ndjson(event)
        to_remove = []
        for q in self.listeners:
            try:
                q.put_nowait(line)
            except Exception:
                to_remove.append(q)
        for q in to_remove:
            if q in self.listeners:
                self.listeners.remove(q)


scan_manager = ScanTaskManager()


def get_cpu_info():
    """检测容器/系统可用的逻辑 CPU 线程数并计算最大安全并行线程数 (总线程 - 1)。"""
    total_cpus = 4
    try:
        if os.path.exists("/sys/fs/cgroup/cpu.max"):
            with open("/sys/fs/cgroup/cpu.max", "r") as f:
                quota, period = f.read().strip().split()
                if quota != "max":
                    total_cpus = max(1, int(int(quota) / int(period)))
                else:
                    total_cpus = len(os.sched_getaffinity(0))
        elif hasattr(os, "sched_getaffinity"):
            total_cpus = len(os.sched_getaffinity(0))
        else:
            total_cpus = os.cpu_count() or 4
    except Exception:
        total_cpus = os.cpu_count() or 4

    return {
        "total_cpus": total_cpus,
        "max_workers": max(1, total_cpus - 1)
    }


async def _run_scan_background(workers: int = 1):
    scan_manager.is_running = True
    scan_manager.cancel_requested = False
    scan_manager.status["is_running"] = True
    scan_manager.status["phase"] = "starting"
    scan_manager.add_log(f"-> 开始扫描挂载文件夹 /music (并行线程数: {workers}) ...")

    try:
        if not os.path.exists(MUSIC_DIR):
            await scan_manager.broadcast({"type": "error", "message": f"Music folder '{MUSIC_DIR}' does not exist inside container."})
            return

        audio_files = []
        for root, dirs, files in os.walk(MUSIC_DIR):
            for file in files:
                if file.lower().endswith((".mp3", ".wav", ".flac", ".m4a", ".ogg")):
                    audio_files.append(os.path.join(root, file))

        if not audio_files:
            await scan_manager.broadcast({
                "type": "done",
                "scanned_files": 0,
                "newly_indexed": 0,
                "cleaned": 0,
                "failed": 0,
                "message": "No audio files found in music directory."
            })
            return

        client = QdrantClient(url=QDRANT_URL)
        existing_paths = {}
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

        paths_to_delete = [
            (abs_path, point_id)
            for abs_path, point_id in existing_paths.items()
            if not os.path.exists(abs_path)
        ]

        cleaned_count = 0
        if paths_to_delete:
            await scan_manager.broadcast({
                "type": "phase",
                "phase": "cleanup",
                "total": len(paths_to_delete)
            })
            for i, (abs_path, point_id) in enumerate(paths_to_delete, 1):
                if scan_manager.cancel_requested:
                    client.close()
                    await scan_manager.broadcast({"type": "aborted"})
                    return
                try:
                    client.delete(collection_name=COLLECTION_NAME, points_selector=[point_id])
                    cleaned_count += 1
                except Exception as e:
                    print(f"Failed to delete point {point_id}: {e}")
                await scan_manager.broadcast({
                    "type": "cleanup_progress",
                    "current": i,
                    "total": len(paths_to_delete),
                    "path": os.path.basename(abs_path)
                })

        files_to_process = [
            f for f in audio_files
            if os.path.abspath(f) not in existing_paths
        ]

        if not files_to_process:
            client.close()
            await scan_manager.broadcast({
                "type": "done",
                "scanned_files": len(audio_files),
                "newly_indexed": 0,
                "cleaned": cleaned_count,
                "failed": 0,
                "message": "All audio files are already indexed in Qdrant."
            })
            return

        checkpoint_path = os.path.join(project_root, "checkpoints/EmbeatMLP/model.pt")
        if not os.path.exists(checkpoint_path):
            client.close()
            await scan_manager.broadcast({"type": "error", "message": "Pre-trained MLP model weights not found."})
            return

        model = load_model(checkpoint_path=checkpoint_path, device="cpu")
        total_to_process = len(files_to_process)
        await scan_manager.broadcast({
            "type": "phase",
            "phase": "extract",
            "total": total_to_process,
            "cleaned": cleaned_count
        })

        songs_rows = []
        failed_files = []
        extract_start = time.time()
        completed_count = 0
        sem = asyncio.Semaphore(workers)

        async def process_file(file_path):
            nonlocal completed_count
            async with sem:
                if scan_manager.cancel_requested:
                    return

                track_basename = os.path.splitext(os.path.basename(file_path))[0]
                await scan_manager.broadcast({
                    "type": "progress",
                    "current": completed_count + 1,
                    "total": total_to_process,
                    "percent": round(completed_count * 100 / total_to_process, 1),
                    "track_name": track_basename,
                    "status": "processing"
                })

                features = await asyncio.to_thread(extract_audio_features, file_path)
                if features is None or scan_manager.cancel_requested:
                    completed_count += 1
                    if features is None:
                        failed_files.append(file_path)
                        await scan_manager.broadcast({
                            "type": "progress",
                            "current": completed_count,
                            "total": total_to_process,
                            "percent": round(completed_count * 100 / total_to_process, 1),
                            "status": "failed",
                            "track_name": track_basename
                        })
                    return

                title, artist = None, None
                try:
                    audio = MutagenFile(file_path, easy=True)
                    if audio is not None:
                        title_tags = audio.get("title") or audio.get("TIT2") or []
                        artist_tags = audio.get("artist") or audio.get("TPE1") or []
                        if title_tags:
                            title = str(title_tags[0]).strip() or None
                        if artist_tags:
                            artist = str(artist_tags[0]).strip() or None
                except Exception:
                    pass

                if not title or not artist:
                    parts = track_basename.split(" - ", 1)
                    if len(parts) == 2:
                        fb_artist = re.sub(r'^\d+\.\s*', '', parts[0].strip())
                        fb_title = parts[1].strip()
                    else:
                        fb_artist = "Unknown Artist"
                        fb_title = track_basename.strip()
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
                completed_count += 1

                elapsed = time.time() - extract_start
                eta_seconds = int((elapsed / completed_count) * (total_to_process - completed_count)) if completed_count > 0 else 0
                await scan_manager.broadcast({
                    "type": "progress",
                    "current": completed_count,
                    "total": total_to_process,
                    "percent": round(completed_count * 100 / total_to_process, 1),
                    "eta_seconds": eta_seconds,
                    "status": "ok",
                    "track_name": title,
                    "artist": artist
                })

        # 多核心并发执行
        tasks = [asyncio.create_task(process_file(fp)) for fp in files_to_process]
        await asyncio.gather(*tasks, return_exceptions=True)

        if scan_manager.cancel_requested:
            client.close()
            await scan_manager.broadcast({"type": "aborted"})
            return

        if not songs_rows:
            client.close()
            await scan_manager.broadcast({
                "type": "done",
                "scanned_files": len(audio_files),
                "newly_indexed": 0,
                "cleaned": cleaned_count,
                "failed": len(failed_files),
                "message": "No new audio files could be processed."
            })
            return

        await scan_manager.broadcast({"type": "phase", "phase": "embedding", "count": len(songs_rows)})

        device = next(model.parameters()).device
        computed_features = build_features(samples=songs_rows, torch_device=device)
        with torch.no_grad():
            embeddings = model(computed_features).cpu().numpy()

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

        await scan_manager.broadcast({
            "type": "done",
            "scanned_files": len(audio_files),
            "newly_indexed": len(songs_rows),
            "cleaned": cleaned_count,
            "failed": len(failed_files),
            "message": "Scan completed and Qdrant database updated successfully."
        })
    except Exception as e:
        print(f"[ScanTask] Error in background scan: {e}")
        await scan_manager.broadcast({"type": "error", "message": f"Background scan error: {str(e)}"})
    finally:
        scan_manager.is_running = False
        scan_manager.status["is_running"] = False
        scan_manager.save_to_disk()


@app.post("/scan")
async def scan_music(workers: Optional[int] = Query(None)):
    """
    启动或拉取后台音频特征提取任务的进度流 (NDJSON)。
    接受 workers 参数控制并行线程数，硬上限为 CPU总核心数 - 1。
    """
    if not scan_manager.is_running:
        cpu = get_cpu_info()
        max_allowed = cpu["max_workers"]
        actual_workers = max(1, min(workers or max_allowed, max_allowed))
        scan_manager.logs.clear()
        scan_manager.task = asyncio.create_task(_run_scan_background(workers=actual_workers))

    q = asyncio.Queue()
    scan_manager.listeners.append(q)

    async def stream_generator():
        try:
            while True:
                try:
                    line = await asyncio.wait_for(q.get(), timeout=1.0)
                    yield line
                except asyncio.TimeoutError:
                    if not scan_manager.is_running and q.empty():
                        break
        finally:
            if q in scan_manager.listeners:
                scan_manager.listeners.remove(q)

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")


@app.get("/scan/status")
async def get_scan_status():
    """获取当前扫描任务运行状态、CPU节点信息与最近日志记录。"""
    return {
        "is_running": scan_manager.is_running,
        "status": scan_manager.status,
        "logs": scan_manager.logs,
        "cpu_info": get_cpu_info()
    }


@app.post("/scan/stop")
async def stop_scan():
    """中途中止后台特征提取任务。"""
    if not scan_manager.is_running:
        return {"message": "当前没有在运行的扫描任务。"}
    scan_manager.cancel_requested = True
    return {"message": "已向后台扫描任务发送中止信号。"}


@app.post("/recommend")
async def recommend_songs(req: RecommendRequest):
    """Retrieves similar tracks for the requested seed track_id."""
    try:
        if req.exclude_style:
            # 步骤 1: 基于声音特征向量，随机抽取一首声学特征差异大的歌曲作为新种子
            new_seed = db.get_random_track_diff_sound(track_id=req.track_id)
            if new_seed is None:
                raise HTTPException(status_code=400, detail="无法找到不同声音特征的歌曲，请确认曲库中有足够多的音乐。")

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
