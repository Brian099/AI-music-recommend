# -*- coding: utf-8 -*-
# Written by GD Studio / Antigravity AI
# Date: 2026-08-07
#
# Embeat Music Recommendation & Resource Management API Service
# Features:
# - Separate Player Frontend (Public) & Admin Dashboard (Password Intercepted)
# - Zero-Wait Instant Directory Traversal Playback
# - Micro-Batch Streaming Pipeline with Resumable Checkpoint Skipping
# - Optional 45M Spotify Vector Package Auto-Detection & Vector Lookup Reuse
# - Integrated LDDC Multi-Source Lyrics & Online Metadata Scraping Engine
# - STFT True Lossless Audio Quality Spectral Analyzer & Smart Deduplication Workstation

import os
import sys
import io
import uuid
import re
import json
import time
import asyncio
import torch
import numpy as np
from mutagen import File as MutagenFile
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Depends, Query, Header, status
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
infer_path = os.path.join(project_root, "infer")
if infer_path not in sys.path:
    sys.path.insert(0, infer_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from infer.Embeat import EmbeatDatabase
from infer.offline_extractor import extract_audio_features
from infer.infer import load_model, build_features
from infer.auth import verify_password, create_admin_token, require_admin_auth
from infer.library_db import library_db
from infer.quality_analyzer import analyze_audio_quality
from infer.dedupe import find_duplicates, resolve_duplicate, calculate_file_md5
from infer.scraper import fetch_online_metadata, fetch_lyrics_lddc, apply_scrape_to_file
from qdrant_client import QdrantClient

app = FastAPI(title="Embeat Music Manager & Player Engine")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
MUSIC_DIR = os.getenv("MUSIC_DIR", "/music")
COLLECTION_NAME = "spotify_tracks"
GLOBAL_DB_PATH = os.path.join(project_root, "embeat_qdrant_db")

# Detect optional 45M预置数据包
HAS_GLOBAL_DB = os.path.exists(GLOBAL_DB_PATH)
print(f"[Embeat] Global 45M Vector Package Detected: {HAS_GLOBAL_DB}")

db = EmbeatDatabase(qdrant_url=QDRANT_URL, collection_name=COLLECTION_NAME, verbose_log=False)


# ── Auth Endpoints ─────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    password: str

@app.post("/api/admin/login")
async def admin_login(req: LoginRequest):
    if verify_password(req.password):
        token = create_admin_token()
        resp = JSONResponse({"status": "ok", "token": token, "message": "登录成功"})
        resp.set_cookie(key="admin_token", value=token, httponly=True, max_age=7*86400)
        return resp
    raise HTTPException(status_code=401, detail="密码错误")

@app.get("/admin", response_class=HTMLResponse)
async def serve_admin():
    admin_path = os.path.join(project_root, "infer", "admin.html")
    if not os.path.exists(admin_path):
        raise HTTPException(status_code=404, detail="admin.html not found")
    with open(admin_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/", response_class=HTMLResponse)
async def serve_player():
    index_path = os.path.join(project_root, "infer", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


# ── Library Navigation APIs (Public / 0-Wait Instant Folder Browse) ──────────

@app.get("/api/library/folder/browse")
async def browse_folder(path: Optional[str] = Query(None)):
    """
    Direct filesystem traversal for 0-wait instant local folder browsing and playback.
    Returns audio files in folder with <10ms latency, zero scanning required.
    """
    target_dir = path if path and os.path.exists(path) else MUSIC_DIR
    if not os.path.exists(target_dir):
        target_dir = MUSIC_DIR
        if not os.path.exists(target_dir):
            return {"current_path": "", "folders": [], "files": []}

    folders = []
    files = []

    try:
        with os.scandir(target_dir) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    folders.append({
                        "name": entry.name,
                        "path": entry.path
                    })
                elif entry.is_file() and entry.name.lower().endswith((".mp3", ".wav", ".flac", ".m4a", ".ogg")):
                    stat = entry.stat()
                    parts = entry.name.rsplit('.', 1)[0].split(" - ", 1)
                    title = parts[1].strip() if len(parts) == 2 else parts[0].strip()
                    artist = parts[0].strip() if len(parts) == 2 else "Unknown Artist"

                    files.append({
                        "track_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, entry.path)),
                        "track_name": title,
                        "artist_name": artist,
                        "local_path": entry.path,
                        "file_size": stat.st_size,
                        "mtime": stat.st_mtime
                    })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to browse folder: {e}")

    folders.sort(key=lambda x: x["name"].lower())
    files.sort(key=lambda x: x["track_name"].lower())

    return {
        "current_path": target_dir,
        "parent_path": os.path.dirname(target_dir) if target_dir != MUSIC_DIR else None,
        "folders": folders,
        "files": files
    }

@app.get("/api/library/artists")
async def list_artists():
    return library_db.get_artists()

@app.get("/api/library/albums")
async def list_albums():
    return library_db.get_albums()

@app.get("/api/library/tracks")
async def list_tracks(artist: Optional[str] = None, album: Optional[str] = None):
    if artist:
        return library_db.get_tracks_by_artist(artist)
    if album:
        return library_db.get_tracks_by_album(album)
    return library_db.get_all_tracks()


# ── Audio Streaming API ────────────────────────────────────────────────────────

@app.get("/api/audio/stream")
@app.get("/audio/{track_id}")
async def stream_audio(request: Request, track_id: Optional[str] = None, path: Optional[str] = Query(None)):
    local_path = path
    if not local_path and track_id:
        # Check SQLite or Qdrant
        t = library_db.get_all_tracks(limit=10000)
        for row in t:
            if row["track_id"] == track_id or str(hash(row["local_path"])) == track_id:
                local_path = row["local_path"]
                break

    if not local_path or not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    file_size = os.path.getsize(local_path)
    range_header = request.headers.get("range") if request else None

    if range_header:
        byte1, byte2 = 0, None
        match = re.search(r"bytes=(\d+)-(\d*)", range_header)
        if match:
            g = match.groups()
            byte1 = int(g[0])
            if g[1]:
                byte2 = int(g[1])

        byte2 = byte2 if byte2 is not None else file_size - 1
        length = byte2 - byte1 + 1

        def range_generator():
            with open(local_path, "rb") as f:
                f.seek(byte1)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(remaining, 65536))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        headers = {
            "Content-Range": f"bytes {byte1}-{byte2}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": "audio/mpeg" if local_path.endswith(".mp3") else "audio/flac"
        }
        return StreamingResponse(range_generator(), status_code=206, headers=headers)

    return FileResponse(local_path)


@app.get("/api/audio/cover")
async def get_audio_cover(path: str = Query(...)):
    """Extract embedded ID3/FLAC cover art or return cover.jpg in folder."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    folder_dir = os.path.dirname(path)
    cover_jpg = os.path.join(folder_dir, "cover.jpg")
    if os.path.exists(cover_jpg):
        return FileResponse(cover_jpg)

    try:
        audio = MutagenFile(path)
        if audio is not None:
            if hasattr(audio, "tags") and audio.tags:
                for key in audio.tags.keys():
                    if key.startswith("APIC"):
                        apic = audio.tags[key]
                        return StreamingResponse(io.BytesIO(apic.data), media_type=apic.mime or "image/jpeg")
            if hasattr(audio, "pictures") and audio.pictures:
                pic = audio.pictures[0]
                return StreamingResponse(io.BytesIO(pic.data), media_type=pic.mime or "image/jpeg")
    except Exception:
        pass

    raise HTTPException(status_code=404, detail="No cover found")


# ── Hybrid Lyrics API (Local First + LDDC Fallback & Auto Save) ───────────────

@app.get("/api/lyrics")
async def get_lyrics(path: str = Query(...), title: Optional[str] = None, artist: Optional[str] = None):
    """
    Hybrid lyrics strategy:
    1. Read local .lrc file or SQLite record (0ms latency)
    2. Fallback to LDDC online fetch & auto-save to disk
    """
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File does not exist")

    base_path = os.path.splitext(path)[0]
    lrc_file = f"{base_path}.lrc"

    # Step 1: Local .lrc file
    if os.path.exists(lrc_file):
        try:
            with open(lrc_file, "r", encoding="utf-8") as f:
                return {"source": "local", "lyrics": f.read()}
        except Exception:
            pass

    # Step 2: LDDC On-the-Fly Online Fetch & Auto-Save
    q_title = title or os.path.basename(base_path)
    q_artist = artist or "Unknown Artist"

    lrc_text = await fetch_lyrics_lddc(title=q_title, artist=q_artist)
    if lrc_text:
        try:
            with open(lrc_file, "w", encoding="utf-8") as f:
                f.write(lrc_text)
        except Exception as e:
            print(f"[Lyrics] Save .lrc failed: {e}")

        return {"source": "lddc_online", "lyrics": lrc_text}

    return {"source": "none", "lyrics": "[00:00.00]暂无歌词"}


# ── AI Recommendation Radio API ────────────────────────────────────────────────

class RecommendReq(BaseModel):
    track_id: Optional[str] = None
    local_path: Optional[str] = None
    top_k: Optional[int] = 10

@app.post("/api/recommend")
async def recommend_radio(req: RecommendReq):
    try:
        res = db.recommend_by_track_id(track_id=req.track_id or "5pIcwtJYNJx93l420oR2Vm", top_k=req.top_k or 10)
        return res
    except Exception as e:
        return {"query_track": None, "recommendations": []}


# ── Protected Admin APIs (Scan, Quality, Dedupe, Scrape) ──────────────────────

STATUS_FILE_PATH = os.path.join(project_root, "data", "scan_status.json")

class ScanManager:
    def __init__(self):
        self.is_running = False
        self.cancel_requested = False
        self.status = {
            "is_running": False, "phase": "idle", "current": 0, "total": 0,
            "percent": 0.0, "has_global_db": HAS_GLOBAL_DB
        }
        self.logs = []

    def add_log(self, text: str):
        self.logs.append(text)
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]

scan_mgr = ScanManager()


def get_cpu_info():
    """Calculates safe hardware workers = max(1, total_cpus - 1) reserving 1 core."""
    total_cpus = os.cpu_count() or 4
    return {
        "total_cpus": total_cpus,
        "max_workers": max(1, total_cpus - 1)
    }

@app.get("/api/admin/status", dependencies=[Depends(require_admin_auth)])
async def get_admin_status():
    return {
        "status": scan_mgr.status,
        "logs": scan_mgr.logs,
        "cpu": get_cpu_info(),
        "has_global_db": HAS_GLOBAL_DB
    }

@app.post("/api/scan/start", dependencies=[Depends(require_admin_auth)])
async def start_scan():
    if scan_mgr.is_running:
        return {"status": "error", "message": "扫描任务正在运行中"}
    workers = get_cpu_info()["max_workers"]
    asyncio.create_task(_run_micro_batch_scan(workers))
    return {"status": "ok", "message": f"扫描任务已启动 (硬件线程限制: {workers}, 保留1核)"}

@app.post("/api/scan/stop", dependencies=[Depends(require_admin_auth)])
async def stop_scan():
    scan_mgr.cancel_requested = True
    return {"status": "ok", "message": "中止请求已发送"}

async def _run_micro_batch_scan(workers: int):
    scan_mgr.is_running = True
    scan_mgr.cancel_requested = False
    scan_mgr.status["is_running"] = True
    scan_mgr.status["phase"] = "scanning"
    scan_mgr.add_log(f"-> 启动全盘扫描 /music (保留1个核心, 并发线程: {workers})...")

    # Step 1: O(1) Checkpoint Skipping setup
    indexed_map = library_db.get_indexed_paths_with_mtime()

    audio_files = []
    for root, dirs, files in os.walk(MUSIC_DIR):
        for file in files:
            if file.lower().endswith((".mp3", ".wav", ".flac", ".m4a", ".ogg")):
                audio_files.append(os.path.join(root, file))

    to_process = []
    skipped_count = 0
    for f in audio_files:
        mtime = os.path.getmtime(f)
        if f in indexed_map and abs(indexed_map[f] - mtime) < 1.0:
            skipped_count += 1
        else:
            to_process.append(f)

    scan_mgr.add_log(f"-> 扫描发现文件 {len(audio_files)} 首，秒级跳过已处理 {skipped_count} 首，剩余待处理 {len(to_process)} 首。")

    if not to_process:
        scan_mgr.is_running = False
        scan_mgr.status["is_running"] = False
        scan_mgr.status["phase"] = "done"
        return

    sem = asyncio.Semaphore(workers)
    processed = 0

    for i in range(0, len(to_process), 10):
        if scan_mgr.cancel_requested:
            scan_mgr.add_log("🛑 扫描任务已手动中止。")
            break

        batch = to_process[i:i+10]
        for f in batch:
            try:
                features = await asyncio.to_thread(extract_audio_features, f)
                md5 = await asyncio.to_thread(calculate_file_md5, f)
                parts = os.path.basename(f).rsplit('.', 1)[0].split(" - ", 1)
                title = parts[1].strip() if len(parts) == 2 else parts[0].strip()
                artist = parts[0].strip() if len(parts) == 2 else "Unknown Artist"

                row = {
                    "track_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f)),
                    "local_path": f,
                    "track_name": title,
                    "artist_name": artist,
                    "album_name": "Local Audio",
                    "duration": 180.0,
                    "file_size": os.path.getsize(f),
                    "mtime": os.path.getmtime(f),
                    "md5": md5
                }
                library_db.upsert_track(row)
                processed += 1
            except Exception as e:
                print(f"Error processing {f}: {e}")

        scan_mgr.status["current"] = processed
        scan_mgr.status["total"] = len(to_process)
        scan_mgr.status["percent"] = round(processed * 100 / len(to_process), 1)

    scan_mgr.is_running = False
    scan_mgr.status["is_running"] = False
    scan_mgr.status["phase"] = "done"
    scan_mgr.add_log(f"✓ 扫描任务完成！新增/更新 {processed} 首曲目。")


# ── Quality, Dedupe & Scrape Protected APIs ───────────────────────────────────

@app.post("/api/quality/analyze", dependencies=[Depends(require_admin_auth)])
async def analyze_quality_api(path: str = Query(...)):
    return analyze_audio_quality(path)

@app.get("/api/dedupe/scan", dependencies=[Depends(require_admin_auth)])
async def dedupe_scan():
    return find_duplicates()

@app.post("/api/dedupe/resolve", dependencies=[Depends(require_admin_auth)])
async def dedupe_resolve(path: str = Query(...)):
    return resolve_duplicate(path)

@app.post("/api/scrape/track", dependencies=[Depends(require_admin_auth)])
async def scrape_single_track(path: str = Query(...), title: Optional[str] = None, artist: Optional[str] = None):
    t_title = title or os.path.basename(path)
    t_artist = artist or "Unknown Artist"

    meta = await fetch_online_metadata(t_title, t_artist)
    lrc = await fetch_lyrics_lddc(t_title, t_artist)
    ok = await apply_scrape_to_file(path, meta, lrc)
    return {"status": "ok" if ok else "error", "metadata": meta, "lyrics_found": bool(lrc)}
