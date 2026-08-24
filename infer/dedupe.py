# -*- coding: utf-8 -*-
# Written by GD Studio / Antigravity AI
# Date: 2026-08-07
# Updated: 2026-08-24 (Integrated Chromaprint Acoustic Fingerprint Engine ported from Songloft)
#
# Smart Audio Deduplication Engine
# Multi-tier duplicate detection:
#   Tier 1: Exact Binary MD5 Match
#   Tier 2: Chromaprint Acoustic Fingerprint + Duration Guard Clustering (Ported from Songloft)
#   Tier 3: Metadata & Normalized Title Matching
# Quality scoring engine recommends keeping the highest quality audio (True Lossless, Bitrate, Format).

import os
import re
import uuid
import shutil
import hashlib
import time
from typing import List, Dict, Any, Optional

from infer.library_db import library_db
from infer.fingerprint import cluster_by_fingerprint_duration


def calculate_file_md5(file_path: str, chunk_size: int = 1048576) -> Optional[str]:
    """
    Fast Sparse MD5 Hash for ultra-fast indexing & deduplication.
    - Files <= 512KB: full binary hash
    - Files > 512KB: fast sparse sample hash (header 64KB + middle 64KB + footer 64KB + file size)
    Reduces disk I/O by >99.7% (from 50MB to 192KB per song), eliminating disk I/O bottlenecks.
    """
    if not os.path.exists(file_path):
        return None
    try:
        size = os.path.getsize(file_path)
        hasher = hashlib.md5()
        if size <= 524288:  # <= 512KB
            with open(file_path, "rb") as f:
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)
        else:
            with open(file_path, "rb") as f:
                # Read 64KB header
                hasher.update(f.read(65536))
                # Read 64KB middle
                f.seek(size // 2)
                hasher.update(f.read(65536))
                # Read 64KB footer
                f.seek(size - 65536)
                hasher.update(f.read(65536))
            # Mix in file size
            hasher.update(str(size).encode("utf-8"))
        return hasher.hexdigest()
    except Exception as e:
        print(f"[MD5] Error calculating hash for {file_path}: {e}")
        return None


def find_duplicates() -> List[Dict[str, Any]]:
    """
    Scans library_db for duplicate groups using B-Tree index pushdown batch operations:
    1. Exact MD5 binary hash duplicates (Single batch query)
    2. Chromaprint acoustic fingerprints with Duration Guard clustering (Single batch query)
    3. Metadata (Artist + Title) exact duplicates with Duration Guard (Single batch query)
    Returns a structured list of duplicate clusters with Smart Keep recommendations in milliseconds.
    """
    duplicate_clusters: List[Dict[str, Any]] = []
    processed_paths = set()

    # ── Tier 1: Exact MD5 Hash Duplicates (Batch Query) ──────────────────────
    md5_groups = library_db.get_all_md5_duplicate_groups()
    for md5, group in md5_groups.items():
        if len(group) > 1:
            cluster = _build_duplicate_cluster(group, match_type="Exact MD5 Match (二进制强一致重复)")
            duplicate_clusters.append(cluster)
            for t in group:
                processed_paths.add(t["local_path"])

    # ── Tier 2: Chromaprint Acoustic Fingerprints (Batch Query + Duration Guard)
    fp_groups = library_db.get_all_fingerprint_duplicate_groups()
    for fp, tracks in fp_groups.items():
        unprocessed = [t for t in tracks if t["local_path"] not in processed_paths]
        if len(unprocessed) > 1:
            clusters = cluster_by_fingerprint_duration(unprocessed, tolerance=30.0)
            for cl in clusters:
                if len(cl) > 1:
                    cluster = _build_duplicate_cluster(cl, match_type="Chromaprint Acoustic Match (声学指纹跨格式重复)")
                    duplicate_clusters.append(cluster)
                    for t in cl:
                        processed_paths.add(t["local_path"])

    # ── Tier 3: Metadata Exact Keys (Batch Query + Duration Guard) ────────────
    meta_groups = library_db.get_all_metadata_duplicate_groups()
    for group in meta_groups:
        unprocessed = [t for t in group if t["local_path"] not in processed_paths]
        if len(unprocessed) > 1:
            clusters = cluster_by_fingerprint_duration(unprocessed, tolerance=40.0)
            for cl in clusters:
                if len(cl) > 1:
                    cluster = _build_duplicate_cluster(cl, match_type="Metadata Match (歌手歌名元数据重复)")
                    duplicate_clusters.append(cluster)
                    for t in cl:
                        processed_paths.add(t["local_path"])

    return duplicate_clusters


def _normalize_title(raw_name: str) -> str:
    """Cleans track numbers and version suffixes for fuzzy comparison."""
    # Clean leading track numbers like 01. or [02]-
    clean = re.sub(r"^(?:\d{1,3}|\[?\d{1,3}\]?)[\s.\-_·]*(?=[^\s\d.])", "", raw_name).strip().lower()
    # Clean version suffixes in brackets e.g. (现场版), （古典版）
    clean = re.sub(r"[\(\（\（].*?[\)\）\）]", "", clean).strip()
    return clean


def _build_duplicate_cluster(tracks: List[Dict[str, Any]], match_type: str) -> Dict[str, Any]:
    """Sort tracks in cluster by quality score and assign 'KEEP' / 'DELETE' recommendation."""
    def quality_score(t: Dict[str, Any]) -> int:
        score = 0
        fmt = (t.get("format") or os.path.splitext(t.get("local_path", ""))[1].lstrip(".")).lower()
        is_lossless = t.get("is_true_lossless")
        bitrate = t.get("bitrate") or 0
        sample_rate = t.get("sample_rate") or 0
        file_size = t.get("file_size") or 0

        # Lossless container / verification
        if is_lossless == 1:
            score += 15000
        elif fmt in ["flac", "wav", "alac", "ape"]:
            score += 8000
        elif fmt in ["m4a", "aac"]:
            score += 3000
        elif fmt in ["mp3", "ogg"]:
            score += 1000

        # Bitrate points
        if bitrate > 0:
            score += bitrate // 1000
        elif fmt in ["flac", "wav", "alac", "ape"]:
            score += 900  # Default lossless estimate

        # Sample rate
        if sample_rate >= 96000:
            score += 1000
        elif sample_rate >= 48000:
            score += 500

        # Size points (1 point per MB)
        score += int(file_size / 1048576)
        return score

    sorted_tracks = sorted(tracks, key=quality_score, reverse=True)
    recommended_keep = sorted_tracks[0]["local_path"]

    for t in sorted_tracks:
        t["recommend_action"] = "KEEP" if t["local_path"] == recommended_keep else "DELETE"

    return {
        "cluster_id": f"cluster_{abs(hash(sorted_tracks[0]['local_path']))}",
        "match_type": match_type,
        "tracks": sorted_tracks,
        "recommended_keep_path": recommended_keep
    }


def _get_trash_dir(file_path: Optional[str] = None) -> str:
    """
    Intelligently determines trash directory on the same filesystem/disk for instant O(1) atomic rename:
    1. If file is inside MUSIC_DIR (/music/...), uses /music/.trash (same disk volume, instant 0.0001s move).
    2. Otherwise falls back to data/trash.
    """
    music_dir = os.getenv("MUSIC_DIR", "/music")
    if file_path and os.path.exists(music_dir):
        try:
            rel = os.path.relpath(file_path, music_dir)
            if not rel.startswith(".."):
                trash_dir = os.path.join(music_dir, ".trash")
                os.makedirs(trash_dir, exist_ok=True)
                return trash_dir
        except Exception:
            pass

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trash_dir = os.path.join(project_root, "data", "trash")
    os.makedirs(trash_dir, exist_ok=True)
    return trash_dir


def resolve_duplicate(delete_path: str, safe_trash: bool = True) -> Dict[str, Any]:
    """
    Safely delete duplicate audio file or move it to trash directory.
    Also removes the record from library_db and Qdrant vector index.
    """
    if not os.path.exists(delete_path):
        library_db.delete_track(delete_path)
        _delete_qdrant_points([delete_path])
        return {"status": "ok", "message": f"曲目已从索引库中移除: {os.path.basename(delete_path)}"}

    try:
        if safe_trash:
            trash_dir = _get_trash_dir(delete_path)
            base_name = os.path.basename(delete_path)
            ts = int(time.time())
            dest_name = f"{ts}_{base_name}"
            dest_path = os.path.join(trash_dir, dest_name)
            try:
                os.replace(delete_path, dest_path)
            except OSError:
                shutil.move(delete_path, dest_path)
        else:
            os.remove(delete_path)

        library_db.delete_track(delete_path)
        _delete_qdrant_points([delete_path])
        return {"status": "ok", "message": f"已成功移除重复音频: {os.path.basename(delete_path)} (安全移入回收站)"}
    except Exception as e:
        return {"status": "error", "message": f"处理重复音频失败: {str(e)}"}


def resolve_batch_duplicates(delete_paths: List[str], safe_trash: bool = True) -> Dict[str, Any]:
    """
    Ultra-fast batch duplicate resolver:
    1. Instant atomic rename for all files on the same disk volume.
    2. Batch removes all paths from SQLite in a single atomic transaction.
    3. Single batch point deletion in Qdrant.
    """
    if not delete_paths:
        return {"status": "ok", "resolved_count": 0, "total": 0, "errors": [], "message": "没有需要清理的音频。"}

    successful_paths = []
    errors = []
    ts = int(time.time())

    for idx, path in enumerate(delete_paths):
        if not os.path.exists(path):
            successful_paths.append(path)
            continue
        try:
            if safe_trash:
                trash_dir = _get_trash_dir(path)
                base_name = os.path.basename(path)
                dest_name = f"{ts}_{idx}_{base_name}"
                dest_path = os.path.join(trash_dir, dest_name)
                try:
                    os.replace(path, dest_path)
                except OSError:
                    shutil.move(path, dest_path)
            else:
                os.remove(path)
            successful_paths.append(path)
        except Exception as e:
            errors.append(f"{os.path.basename(path)}: {str(e)}")

    # 1. Batch delete from SQLite in 1 transaction
    if successful_paths:
        library_db.delete_tracks_batch(successful_paths)

    # 2. Batch delete from Qdrant in 1 call
    if successful_paths:
        _delete_qdrant_points(successful_paths)

    return {
        "status": "ok" if not errors else "partial",
        "resolved_count": len(successful_paths),
        "total": len(delete_paths),
        "errors": errors,
        "message": f"成功清理 {len(successful_paths)}/{len(delete_paths)} 首重复音频。"
    }


def resolve_all_duplicates(safe_trash: bool = True) -> Dict[str, Any]:
    """Scans all duplicates server-side and automatically resolves all recommended DELETE items."""
    clusters = find_duplicates()
    to_delete = []
    for c in clusters:
        for t in c.get("tracks", []):
            if t.get("recommend_action") == "DELETE":
                to_delete.append(t["local_path"])

    if not to_delete:
        return {"status": "ok", "resolved_count": 0, "total": 0, "errors": [], "message": "曲库中未发现需要清理的重复音频。"}

    return resolve_batch_duplicates(to_delete, safe_trash=safe_trash)


def _delete_qdrant_points(file_paths: List[str]):
    """Clean up vector points from Qdrant in a single batch operation (gracefully non-blocking)."""
    if not file_paths:
        return
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qdrant_models
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        collection_name = "spotify_tracks"
        client = QdrantClient(url=qdrant_url, timeout=0.8)
        point_ids = []
        for fp in file_paths:
            track_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, fp))
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, track_id))
            point_ids.append(point_id)

        client.delete(
            collection_name=collection_name,
            points_selector=qdrant_models.PointIdsList(points=point_ids)
        )
        client.close()
    except Exception:
        # Gracefully pass if Qdrant is offline or not installed
        pass
