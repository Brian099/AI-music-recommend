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
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
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
from infer.model_infer import load_model, build_features
from infer.auth import verify_password, create_admin_token, require_admin_auth
from infer.library_db import library_db
from infer.quality_analyzer import analyze_audio_quality
from infer.dedupe import find_duplicates, resolve_duplicate, resolve_batch_duplicates, calculate_file_md5
from infer.fingerprint import fingerprint_service, is_chromaprint_available
from infer.scraper import fetch_online_metadata, fetch_lyrics_lddc, apply_scrape_to_file
try:
    from qdrant_client import QdrantClient
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False

import logging

LOG_DIR = os.path.join(project_root, "data")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

logger = logging.getLogger("embeat")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(stream_handler)

app = FastAPI(title="Embeat Music Manager & Player Engine")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
MUSIC_DIR = os.getenv("MUSIC_DIR", "/music")
COLLECTION_NAME = "spotify_tracks"
GLOBAL_DB_PATH = os.path.join(project_root, "embeat_qdrant_db")

# Detect optional 45M预置数据包
HAS_GLOBAL_DB = os.path.exists(GLOBAL_DB_PATH)
logger.info(f"[Embeat] Global 45M Vector Package Detected: {HAS_GLOBAL_DB} | Log File: {LOG_FILE}")

try:
    db = EmbeatDatabase(qdrant_url=QDRANT_URL, collection_name=COLLECTION_NAME, verbose_log=False)
except Exception as e:
    logger.warning(f"[Embeat] EmbeatDatabase initialization warning (Qdrant offline): {e}")
    db = None


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


def parse_audio_file_info(file_path: str, check_db: bool = True) -> dict:
    """
    Smart 3-stage audio metadata parser:
    1. Reads SQLite database (if scraped or previously saved with valid artist)
    2. Reads Mutagen ID3 / FLAC tags from local file
    3. Fallback to smart filename regex (cleans leading track numbers, splits title/artist)
    """
    if check_db:
        existing = library_db.get_track_by_path(file_path)
        if existing and existing.get("artist_name") and existing.get("artist_name") != "Unknown Artist":
            return {
                "title": existing.get("track_name"),
                "artist": existing.get("artist_name"),
                "album": existing.get("album_name") or "Local Audio",
                "duration": existing.get("duration") or 180.0,
                "bitrate": existing.get("bitrate") or 0,
                "sample_rate": existing.get("sample_rate") or 44100
            }

    duration = 180.0
    bitrate = 0
    sample_rate = 44100
    title_str = ""
    art_str = ""
    alb_str = ""

    try:
        mf = MutagenFile(file_path)
        if mf:
            if hasattr(mf, "info"):
                duration = float(getattr(mf.info, "length", 180.0))
                bitrate = int(getattr(mf.info, "bitrate", 0))
                sample_rate = int(getattr(mf.info, "sample_rate", 44100))

            art = mf.get("artist") or mf.get("TPE1") or mf.get("ARTIST")
            tit = mf.get("title") or mf.get("TIT2") or mf.get("TITLE")
            alb = mf.get("album") or mf.get("TALB") or mf.get("ALBUM")

            def _clean_tag(t):
                if isinstance(t, (list, tuple)) and len(t) > 0:
                    return str(t[0]).strip()
                elif isinstance(t, str):
                    return t.strip()
                return ""

            art_str = _clean_tag(art)
            tit_str = _clean_tag(tit)
            alb_str = _clean_tag(alb)

            if art_str and art_str.lower() != "unknown artist" and tit_str:
                return {
                    "title": tit_str,
                    "artist": art_str,
                    "album": alb_str or "Local Audio",
                    "duration": duration,
                    "bitrate": bitrate,
                    "sample_rate": sample_rate
                }
    except Exception:
        pass

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    clean_name = re.sub(r"^(?:\d{1,3}|\[?\d{1,3}\]?)[\s.\-_·]*(?=[^\s\d.])", "", base_name).strip()

    title = tit_str or clean_name
    artist = art_str or "Unknown Artist"

    if not tit_str or artist == "Unknown Artist":
        if " - " in clean_name:
            parts = clean_name.split(" - ", 1)
            artist, title = parts[0].strip(), parts[1].strip()
        elif "-" in clean_name and not clean_name.startswith("-"):
            parts = clean_name.split("-", 1)
            p1, p2 = parts[0].strip(), parts[1].strip()
            if len(p2) <= 10 and not any(c.isdigit() for c in p2):
                title, artist = p1, p2
            else:
                artist, title = p1, p2
        elif "_" in clean_name:
            parts = clean_name.split("_", 1)
            artist, title = parts[0].strip(), parts[1].strip()

    return {
        "title": title or base_name,
        "artist": artist or "Unknown Artist",
        "album": alb_str or "Local Audio",
        "duration": duration,
        "bitrate": bitrate,
        "sample_rate": sample_rate
    }


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
                    info = parse_audio_file_info(entry.path)

                    track_obj = {
                        "track_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, entry.path)),
                        "track_name": info["title"],
                        "artist_name": info["artist"],
                        "local_path": entry.path,
                        "file_size": stat.st_size,
                        "mtime": stat.st_mtime
                    }
                    files.append(track_obj)

                    # Instant auto-indexing into SQLite library_db so Artists & Albums populate with 0 waiting!
                    try:
                        library_db.upsert_track({
                            "track_id": track_obj["track_id"],
                            "local_path": track_obj["local_path"],
                            "track_name": track_obj["track_name"],
                            "artist_name": track_obj["artist_name"],
                            "album_name": info.get("album", "Local Audio"),
                            "file_size": track_obj["file_size"],
                            "mtime": track_obj["mtime"]
                        })
                    except Exception:
                        pass
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
    1. Read local .lrc file (0ms latency)
    2. Fallback to online fetch & auto-save to disk
    """
    logger.info(f"[API /api/lyrics] Request received | path='{path}', title='{title}', artist='{artist}'")
    q_title = title
    q_artist = artist

    if path:
        base_path = os.path.splitext(path)[0]
        if not q_title:
            q_title = os.path.basename(base_path)
        
        lrc_file = f"{base_path}.lrc"
        if os.path.exists(lrc_file):
            try:
                with open(lrc_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    logger.info(f"[API /api/lyrics] Local .lrc file hit at '{lrc_file}' | length={len(content)}")
                    return {"source": "local", "lyrics": content}
            except Exception as e:
                logger.warning(f"[API /api/lyrics] Failed reading local .lrc file '{lrc_file}': {e}")

    # Online Fetch & Auto-Save
    q_title = q_title or "Unknown Track"
    q_artist = q_artist or "Unknown Artist"

    logger.info(f"[API /api/lyrics] Executing online lyrics search for q_title='{q_title}', q_artist='{q_artist}'...")
    lrc_text = await fetch_lyrics_lddc(title=q_title, artist=q_artist)

    if lrc_text:
        logger.info(f"[API /api/lyrics] Online lyrics SUCCESS | length={len(lrc_text)} chars")
        if path and os.path.exists(os.path.dirname(path)):
            base_path = os.path.splitext(path)[0]
            lrc_file = f"{base_path}.lrc"
            try:
                with open(lrc_file, "w", encoding="utf-8") as f:
                    f.write(lrc_text)
                logger.info(f"[API /api/lyrics] Saved online lyrics to local file '{lrc_file}'")
            except Exception as e:
                logger.error(f"[API /api/lyrics] Save .lrc failed: {e}")

        return {"source": "online", "lyrics": lrc_text}

    logger.warning(f"[API /api/lyrics] Online lyrics NOT FOUND for '{q_title}' - '{q_artist}'")
    return {"source": "none", "lyrics": "[00:00.00]暂无歌词"}


# ── AI Recommendation Radio API ────────────────────────────────────────────────

class RecommendReq(BaseModel):
    track_id: Optional[str] = None
    local_path: Optional[str] = None
    top_k: Optional[int] = 20

@app.post("/api/recommend")
async def recommend_radio(req: RecommendReq):
    """
    Seed-based AI Roaming Radio generator:
    1. Looks up seed track by local_path or track_id
    2. Retrieves vector acoustic recommendations or smart artist/genre fallback
    """
    try:
        seed_track = None
        if req.local_path:
            seed_track = library_db.get_track_by_path(req.local_path)
            if not seed_track:
                base = os.path.basename(req.local_path)
                parts = base.rsplit('.', 1)[0].split(" - ", 1)
                t_title = parts[1].strip() if len(parts) == 2 else parts[0].strip()
                t_artist = parts[0].strip() if len(parts) == 2 else "Unknown Artist"
                seed_track = {
                    "track_id": req.track_id or str(uuid.uuid5(uuid.NAMESPACE_DNS, req.local_path)),
                    "local_path": req.local_path,
                    "track_name": t_title,
                    "artist_name": t_artist,
                    "album_name": "Seed Track"
                }

        recs = []
        if req.track_id:
            res = db.recommend_by_track_id(track_id=req.track_id, top_k=req.top_k or 10)
            if res and isinstance(res, dict) and "recommendations" in res:
                recs = res["recommendations"]

        all_tracks = library_db.get_all_tracks()
        if seed_track:
            seed_artist = seed_track.get("artist_name", "").lower()
            same_artist = [t for t in all_tracks if t.get("local_path") != seed_track.get("local_path") and t.get("artist_name", "").lower() == seed_artist]
            other_tracks = [t for t in all_tracks if t.get("local_path") != seed_track.get("local_path") and t.get("artist_name", "").lower() != seed_artist]
            import random
            random.shuffle(same_artist)
            random.shuffle(other_tracks)
            playlist = [seed_track] + same_artist + other_tracks
            return {"query_track": seed_track, "recommendations": playlist[:req.top_k or 20]}

        import random
        random_list = list(all_tracks)
        random.shuffle(random_list)
        return {"query_track": None, "recommendations": random_list[:req.top_k or 20]}
    except Exception as e:
        logger.error(f"[API /api/recommend] Exception: {e}")
        all_tracks = library_db.get_all_tracks()
        return {"query_track": None, "recommendations": all_tracks}


# ── Protected Admin APIs (Scan, Quality, Dedupe, Scrape) ──────────────────────

STATUS_FILE_PATH = os.path.join(project_root, "data", "scan_status.json")

class TaskManager:
    def __init__(self, task_name: str):
        self.task_name = task_name
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

scan_mgr = TaskManager("Scan")
scrape_task_mgr = TaskManager("Scrape")
vector_task_mgr = TaskManager("Vector")


def get_cpu_info():
    """Calculates safe hardware workers = max(1, total_cpus - 1) reserving 1 core."""
    total_cpus = os.cpu_count() or 4
    return {
        "total_cpus": total_cpus,
        "max_workers": max(1, total_cpus - 1)
    }

@app.get("/api/admin/status", dependencies=[Depends(require_admin_auth)])
async def get_admin_status():
    fp_stats = library_db.get_fingerprint_stats()
    fp_progress = fingerprint_service.get_progress()
    return {
        "scan": {
            "status": scan_mgr.status,
            "logs": scan_mgr.logs,
        },
        "scrape": {
            "status": scrape_task_mgr.status,
            "logs": scrape_task_mgr.logs,
        },
        "vector": {
            "status": vector_task_mgr.status,
            "logs": vector_task_mgr.logs,
        },
        "fingerprint": {
            "status": fp_progress,
            "stats": fp_stats,
            "logs": fingerprint_service.logs,
            "is_available": is_chromaprint_available()
        },
        "status": scan_mgr.status,  # Fallback backward compatibility
        "logs": scan_mgr.logs,      # Fallback backward compatibility
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

def _process_single_audio_file(file_path: str) -> Optional[Dict[str, Any]]:
    try:
        if not os.path.exists(file_path):
            return None
        size = os.path.getsize(file_path)
        mtime = os.path.getmtime(file_path)
        md5 = calculate_file_md5(file_path)
        info = parse_audio_file_info(file_path, check_db=False)

        return {
            "track_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, file_path)),
            "local_path": file_path,
            "track_name": info["title"],
            "artist_name": info["artist"],
            "album_name": info.get("album", "Local Audio"),
            "duration": info.get("duration", 180.0),
            "bitrate": info.get("bitrate", 0),
            "sample_rate": info.get("sample_rate", 0),
            "format": os.path.splitext(file_path)[1].lstrip(".").lower(),
            "file_size": size,
            "mtime": mtime,
            "md5": md5
        }
    except Exception as e:
        logger.error(f"Error parsing audio file {file_path}: {e}")
        return None


async def _run_micro_batch_scan(workers: int):
    scan_mgr.is_running = True
    scan_mgr.cancel_requested = False
    scan_mgr.status["is_running"] = True
    scan_mgr.status["phase"] = "scanning"
    scan_mgr.add_log("-> 启动全盘轻量极速扫描 /music (防卡死顺序流式读取 + 批量事务提交)...")

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
        try:
            mtime = os.path.getmtime(f)
            if f in indexed_map and abs(indexed_map[f] - mtime) < 1.0:
                skipped_count += 1
            else:
                to_process.append(f)
        except OSError:
            continue

    scan_mgr.add_log(f"-> 扫描发现文件 {len(audio_files)} 首，秒级跳过已处理 {skipped_count} 首，剩余待处理 {len(to_process)} 首。")

    if not to_process:
        scan_mgr.is_running = False
        scan_mgr.status["is_running"] = False
        scan_mgr.status["phase"] = "done"
        scan_mgr.status["percent"] = 100
        scan_mgr.add_log(f"✓ 所有 {len(audio_files)} 首曲目均已在数据库中（断点秒级跳过），无需重复处理！")
        return

    processed = 0
    batch_buffer = []
    batch_size = 50
    loop = asyncio.get_event_loop()

    for i, file_path in enumerate(to_process):
        if scan_mgr.cancel_requested:
            scan_mgr.add_log("🛑 扫描任务已手动中止。")
            break

        # Sequential file processing (0 disk thrashing)
        row = await loop.run_in_executor(None, _process_single_audio_file, file_path)
        if row:
            batch_buffer.append(row)

        processed += 1

        # Commit batch to SQLite and yield control to event loop
        if len(batch_buffer) >= batch_size or processed == len(to_process):
            await loop.run_in_executor(None, library_db.upsert_tracks_batch, batch_buffer)
            batch_buffer = []

            scan_mgr.status["current"] = processed
            scan_mgr.status["total"] = len(to_process)
            scan_mgr.status["percent"] = round(processed * 100 / len(to_process), 1)
            scan_mgr.add_log(f"-> 顺序入库进度: {processed}/{len(to_process)} 首 ({scan_mgr.status['percent']}%) - 最新: {os.path.basename(file_path)}")
            # Short async sleep to allow UI polling to update seamlessly
            await asyncio.sleep(0.001)

    # Bulk refresh artist and album aggregate stats in one fast query
    await loop.run_in_executor(None, library_db.refresh_library_aggregates)

    scan_mgr.is_running = False
    scan_mgr.status["is_running"] = False
    scan_mgr.status["phase"] = "done"
    scan_mgr.status["percent"] = 100
    scan_mgr.add_log(f"✓ 扫描任务完成！新增/更新入库 {processed} 首曲目。")


# ── Batch Scraping Dedicated APIs & Task Manager ─────────────────────────────

@app.get("/api/scrape/batch/status", dependencies=[Depends(require_admin_auth)])
async def get_scrape_batch_status():
    return {
        "status": scrape_task_mgr.status,
        "logs": scrape_task_mgr.logs,
    }

@app.post("/api/scrape/batch/start", dependencies=[Depends(require_admin_auth)])
async def start_scrape_batch():
    if scrape_task_mgr.is_running:
        return {"status": "error", "message": "批量刮削任务正在运行中"}
    workers = get_cpu_info()["max_workers"]
    asyncio.create_task(_run_batch_scrape(workers))
    return {"status": "ok", "message": f"批量刮削任务已启动 (并发线程: {workers})"}

@app.post("/api/scrape/batch/stop", dependencies=[Depends(require_admin_auth)])
async def stop_scrape_batch():
    scrape_task_mgr.cancel_requested = True
    return {"status": "ok", "message": "刮削中止请求已发送"}

async def _run_batch_scrape(workers: int):
    scrape_task_mgr.is_running = True
    scrape_task_mgr.cancel_requested = False
    scrape_task_mgr.status["is_running"] = True
    scrape_task_mgr.status["phase"] = "scraping"
    scrape_task_mgr.add_log(f"-> 启动全库曲目批量元数据与歌词在线刮削 (并发任务: {workers})...")

    tracks = library_db.get_all_tracks(limit=50000, offset=0)
    if not tracks:
        scrape_task_mgr.is_running = False
        scrape_task_mgr.status["is_running"] = False
        scrape_task_mgr.status["phase"] = "done"
        scrape_task_mgr.status["percent"] = 100
        scrape_task_mgr.add_log("✓ 数据库中暂无待刮削的曲目。")
        return

    to_process = []
    skipped_count = 0
    for t in tracks:
        if t.get("scraped_at"):
            skipped_count += 1
        else:
            to_process.append(t)

    if skipped_count > 0:
        scrape_task_mgr.add_log(f"-> 发现曲目 {len(tracks)} 首，断点秒级跳过已完成刮削曲目 {skipped_count} 首，剩余待处理 {len(to_process)} 首。")

    if not to_process:
        scrape_task_mgr.is_running = False
        scrape_task_mgr.status["is_running"] = False
        scrape_task_mgr.status["phase"] = "done"
        scrape_task_mgr.status["percent"] = 100
        scrape_task_mgr.add_log(f"✓ 所有 {len(tracks)} 首曲目均已完成在线刮削（断点秒级跳过），无需重复刮削！")
        return

    scrape_task_mgr.status["total"] = len(to_process)
    processed = 0
    sem = asyncio.Semaphore(max(1, workers))

    async def _scrape_single(track):
        nonlocal processed
        if scrape_task_mgr.cancel_requested:
            return

        path = track.get("local_path")
        if not path or not os.path.exists(path):
            processed += 1
            scrape_task_mgr.status["current"] = processed
            scrape_task_mgr.status["percent"] = round(processed * 100 / len(to_process), 1)
            return

        async with sem:
            if scrape_task_mgr.cancel_requested:
                return
            info = parse_audio_file_info(path)
            title = info.get("title") or track.get("track_name") or os.path.basename(path)
            artist = info.get("artist") if info.get("artist") != "Unknown Artist" else (track.get("artist_name") or "Unknown Artist")

            try:
                meta = await fetch_online_metadata(title, artist, file_path=path)
                lrc = await fetch_lyrics_lddc(title, artist, file_path=path)
                await apply_scrape_to_file(path, meta, lrc)
                scrape_task_mgr.add_log(f"✓ 已完成刮削与标签嵌入: {os.path.basename(path)} [{meta.get('title', title)} - {meta.get('artist', artist)}]")
            except Exception as e:
                scrape_task_mgr.add_log(f"⚠️ 刮削出错 {os.path.basename(path)}: {e}")

            processed += 1
            scrape_task_mgr.status["current"] = processed
            scrape_task_mgr.status["percent"] = round(processed * 100 / len(to_process), 1)

    tasks = [_scrape_single(t) for t in to_process]
    await asyncio.gather(*tasks)

    scrape_task_mgr.is_running = False
    scrape_task_mgr.status["is_running"] = False
    scrape_task_mgr.status["phase"] = "done"
    scrape_task_mgr.status["percent"] = 100
    scrape_task_mgr.add_log(f"✓ 批量刮削任务处理完毕！已处理 {processed}/{len(to_process)} 首曲目。")


# ── Acoustic Vector Extraction & Qdrant Dedicated APIs & Task Manager ────────

@app.get("/api/vector/extract/status", dependencies=[Depends(require_admin_auth)])
async def get_vector_extract_status():
    return {
        "status": vector_task_mgr.status,
        "logs": vector_task_mgr.logs,
    }

@app.post("/api/vector/extract/start", dependencies=[Depends(require_admin_auth)])
async def start_vector_extract():
    if vector_task_mgr.is_running:
        return {"status": "error", "message": "声学向量提取任务正在运行中"}
    workers = get_cpu_info()["max_workers"]
    asyncio.create_task(_run_batch_vector_extraction(workers))
    return {"status": "ok", "message": f"声学向量提取任务已启动 (并发线程: {workers})"}

@app.post("/api/vector/extract/stop", dependencies=[Depends(require_admin_auth)])
async def stop_vector_extract():
    vector_task_mgr.cancel_requested = True
    return {"status": "ok", "message": "向量提取中止请求已发送"}

async def _run_batch_vector_extraction(workers: int):
    vector_task_mgr.is_running = True
    vector_task_mgr.cancel_requested = False
    vector_task_mgr.status["is_running"] = True
    vector_task_mgr.status["phase"] = "extracting"
    vector_task_mgr.add_log(f"-> 启动全库 AI 声学向量提取与 Qdrant 索引建库任务 (多核并发线程: {workers})...")

    tracks = library_db.get_all_tracks(limit=50000, offset=0)
    if not tracks:
        vector_task_mgr.is_running = False
        vector_task_mgr.status["is_running"] = False
        vector_task_mgr.status["phase"] = "done"
        vector_task_mgr.status["percent"] = 100
        vector_task_mgr.add_log("✓ 数据库中暂无音频文件。")
        return

    processed = 0

    try:
        from infer.offline_extractor import extract_audio_features
        from infer.model_infer import load_model, build_features
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qdrant_models
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        import torch

        checkpoint_path = os.path.join(project_root, "checkpoints", "EmbeatMLP", "model.pt")
        model = None
        if os.path.isfile(checkpoint_path):
            try:
                model = load_model(checkpoint_path=checkpoint_path, device="cpu")
                vector_task_mgr.add_log("-> EmbeatMLP 神经网络模型加载成功")
            except Exception as me:
                vector_task_mgr.add_log(f"⚠️ EmbeatMLP 模型加载跳过: {me}")

        # Connect Qdrant
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        qdrant_path = os.path.join(project_root, "data", "qdrant_storage")
        os.makedirs(qdrant_path, exist_ok=True)
        
        try:
            q_client = QdrantClient(url=qdrant_url, timeout=3.0)
            q_client.get_collections()
        except Exception:
            q_client = QdrantClient(path=qdrant_path)

        collection_name = "embeat_tracks"
        if not q_client.collection_exists(collection_name):
            q_client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=64,
                    distance=qdrant_models.Distance.COSINE,
                    datatype=qdrant_models.Datatype.FLOAT32,
                )
            )

        # Check existing Qdrant vector points for O(1) breakpoint skipping
        existing_point_ids = set()
        candidate_map = {}
        for track in tracks:
            path = track.get("local_path", "")
            t_id = track.get("track_id") or str(uuid.uuid5(uuid.NAMESPACE_DNS, path))
            p_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, t_id))
            candidate_map[p_id] = track

        all_p_ids = list(candidate_map.keys())
        for i in range(0, len(all_p_ids), 500):
            batch_ids = all_p_ids[i:i+500]
            try:
                retrieved = q_client.retrieve(collection_name=collection_name, ids=batch_ids, with_payload=False, with_vectors=False)
                for r in retrieved:
                    existing_point_ids.add(str(r.id))
            except Exception:
                pass

        to_process = [t for p_id, t in candidate_map.items() if p_id not in existing_point_ids]
        skipped_count = len(tracks) - len(to_process)

        if skipped_count > 0:
            vector_task_mgr.add_log(f"-> 发现曲目 {len(tracks)} 首，断点秒级跳过已提取向量曲目 {skipped_count} 首，剩余待处理 {len(to_process)} 首。")

        if not to_process:
            vector_task_mgr.is_running = False
            vector_task_mgr.status["is_running"] = False
            vector_task_mgr.status["phase"] = "done"
            vector_task_mgr.status["percent"] = 100
            vector_task_mgr.add_log(f"✓ 所有 {len(tracks)} 首曲目均已完成 AI 声学向量提取与 Qdrant 索引建库（断点秒级跳过），无需重复提取！")
            return

        vector_task_mgr.status["total"] = len(to_process)
        processed = 0

        # Adaptive multi-core worker pool
        cpu_count = os.cpu_count() or 4
        calc_workers = max(2, min(16, cpu_count - 1))
        executor = ThreadPoolExecutor(max_workers=calc_workers)
        loop = asyncio.get_running_loop()
        batch_size = max(16, calc_workers * 2)

        def _extract_worker(path: str):
            try:
                if not os.path.exists(path):
                    return None
                return extract_audio_features(path)
            except Exception as e:
                logger.error(f"Error in extract_audio_features for {path}: {e}")
                return None

        try:
            for i in range(0, len(to_process), batch_size):
                if vector_task_mgr.cancel_requested:
                    vector_task_mgr.add_log("🛑 向量提取任务已手动中止。")
                    break

                batch = to_process[i:i + batch_size]
                tasks = [loop.run_in_executor(executor, _extract_worker, t.get("local_path", "")) for t in batch]
                batch_features = await asyncio.gather(*tasks)

                valid_rows = []
                for track, feats in zip(batch, batch_features):
                    path = track.get("local_path", "")
                    if feats:
                        artist_genres = feats.pop("artist_genres", "pop")
                        artist_genre_idx = feats.pop("artist_genre_idx", 1)
                        row = {
                            "track_id": track.get("track_id") or str(uuid.uuid5(uuid.NAMESPACE_DNS, path)),
                            "track_name": track.get("track_name") or os.path.basename(path),
                            "artist_name": track.get("artist_name") or "Unknown Artist",
                            "artist_idx": abs(hash(track.get("artist_name", ""))) % 1000 + 1,
                            "artist_genres": artist_genres,
                            "artist_genre_idx": artist_genre_idx,
                            "related_artist_idxs": [],
                            "album_name": track.get("album_name") or "Local Audio",
                            "isrc": f"LOCAL_{abs(hash(path)) % 1000000}",
                            "popularity": 0.5,
                            "local_path": path,
                            **feats
                        }
                        valid_rows.append(row)

                if valid_rows:
                    if model:
                        computed_feat = build_features(samples=valid_rows, torch_device=torch.device("cpu"))
                        with torch.no_grad():
                            embs = model(computed_feat).cpu().numpy()
                    else:
                        embs = np.random.randn(len(valid_rows), 64).astype(np.float32)

                    points = []
                    for row, emb in zip(valid_rows, embs):
                        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, row["track_id"]))
                        payload = {k: row[k] for k in (
                            "track_id", "track_name", "popularity", "artist_name",
                            "artist_idx", "artist_genres", "artist_genre_idx",
                            "related_artist_idxs", "album_name", "isrc", "local_path"
                        ) if k in row}
                        points.append(qdrant_models.PointStruct(id=point_id, vector=emb.tolist(), payload=payload))

                    if points:
                        q_client.upsert(collection_name=collection_name, points=points)

                processed += len(batch)
                vector_task_mgr.status["current"] = processed
                vector_task_mgr.status["percent"] = round(processed * 100 / len(to_process), 1)
                vector_task_mgr.add_log(f"-> 并发提取与向量建库进度: {processed}/{len(to_process)} ({vector_task_mgr.status['percent']}%)")

        finally:
            if executor:
                executor.shutdown(wait=False)

    except Exception as e:
        vector_task_mgr.add_log(f"❌ 向量提取任务化异常: {e}")

    vector_task_mgr.is_running = False
    vector_task_mgr.status["is_running"] = False
    vector_task_mgr.status["phase"] = "done"
    vector_task_mgr.status["percent"] = 100
    vector_task_mgr.add_log(f"✓ 声学向量提取与 Qdrant 索引任务完成！已处理 {processed}/{len(to_process)} 首曲目。")


# ── Quality, Dedupe & Fingerprint Protected APIs ──────────────────────────────

class DedupeBatchRequest(BaseModel):
    paths: List[str]
    safe_trash: bool = True


@app.post("/api/quality/analyze", dependencies=[Depends(require_admin_auth)])
async def analyze_quality_api(path: str = Query(...)):
    return analyze_audio_quality(path)

# ── Fingerprint Management APIs (Ported from Songloft) ─────────────────────────

@app.get("/api/fingerprint/status", dependencies=[Depends(require_admin_auth)])
async def fingerprint_status_api():
    stats = library_db.get_fingerprint_stats()
    available = is_chromaprint_available()
    progress = fingerprint_service.get_progress()
    return {
        "is_available": available,
        "stats": stats,
        "progress": progress
    }

@app.get("/api/fingerprint/progress", dependencies=[Depends(require_admin_auth)])
async def fingerprint_progress_api():
    return {
        "progress": fingerprint_service.get_progress(),
        "logs": fingerprint_service.logs
    }

@app.post("/api/fingerprint/start", dependencies=[Depends(require_admin_auth)])
async def fingerprint_start_api(mode: str = Query("missing", description="missing, recompute_all, retry_failed")):
    return await fingerprint_service.start(mode=mode)

@app.post("/api/fingerprint/stop", dependencies=[Depends(require_admin_auth)])
async def fingerprint_stop_api():
    fingerprint_service.cancel()
    return {"status": "ok", "message": "指纹计算中止请求已发送"}

# ── Deduplication APIs ────────────────────────────────────────────────────────

@app.get("/api/dedupe/scan", dependencies=[Depends(require_admin_auth)])
async def dedupe_scan():
    return await asyncio.to_thread(find_duplicates)

@app.post("/api/dedupe/resolve", dependencies=[Depends(require_admin_auth)])
async def dedupe_resolve(path: str = Query(...)):
    return await asyncio.to_thread(resolve_duplicate, path)

@app.post("/api/dedupe/resolve_batch", dependencies=[Depends(require_admin_auth)])
async def dedupe_resolve_batch_api(req: DedupeBatchRequest):
    return await asyncio.to_thread(resolve_batch_duplicates, req.paths, safe_trash=req.safe_trash)

@app.post("/api/scrape/track", dependencies=[Depends(require_admin_auth)])
async def scrape_single_track(path: str = Query(...), title: Optional[str] = None, artist: Optional[str] = None):
    t_title = title or os.path.basename(path)
    t_artist = artist or "Unknown Artist"

    meta = await fetch_online_metadata(t_title, t_artist, file_path=path)
    lrc = await fetch_lyrics_lddc(t_title, t_artist, file_path=path)
    ok = await apply_scrape_to_file(path, meta, lrc)
    return {"status": "ok" if ok else "error", "metadata": meta, "lyrics_found": bool(lrc)}
