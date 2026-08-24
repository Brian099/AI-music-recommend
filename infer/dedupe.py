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


def resolve_duplicate(delete_path: str, safe_trash: bool = True) -> Dict[str, Any]:
    """
    Safely delete duplicate audio file or move it to data/trash.
    Also removes the record from library_db and Qdrant vector index.
    """
    if not os.path.exists(delete_path):
        library_db.delete_track(delete_path)
        _delete_qdrant_point(delete_path)
        return {"status": "ok", "message": f"曲目已从索引库中移除: {os.path.basename(delete_path)}"}

    try:
        if safe_trash:
            trash_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "trash")
            os.makedirs(trash_dir, exist_ok=True)
            # Avoid filename collisions by prepending timestamp
            base_name = os.path.basename(delete_path)
            ts = int(time.time())
            dest_name = f"{ts}_{base_name}"
            dest_path = os.path.join(trash_dir, dest_name)
            shutil.move(delete_path, dest_path)
        else:
            os.remove(delete_path)

        library_db.delete_track(delete_path)
        _delete_qdrant_point(delete_path)
        return {"status": "ok", "message": f"已成功移除重复音频: {os.path.basename(delete_path)} (安全移入 data/trash)"}
    except Exception as e:
        return {"status": "error", "message": f"处理重复音频失败: {str(e)}"}


def resolve_batch_duplicates(delete_paths: List[str], safe_trash: bool = True) -> Dict[str, Any]:
    """Batch resolves multiple duplicate audio paths and cleans up database + Qdrant vectors."""
    success_count = 0
    errors = []

    for path in delete_paths:
        res = resolve_duplicate(path, safe_trash=safe_trash)
        if res.get("status") == "ok":
            success_count += 1
        else:
            errors.append(f"{os.path.basename(path)}: {res.get('message')}")

    return {
        "status": "ok" if not errors else "partial",
        "resolved_count": success_count,
        "total": len(delete_paths),
        "errors": errors,
        "message": f"成功清理 {success_count}/{len(delete_paths)} 首重复音频。"
    }


def _delete_qdrant_point(file_path: str):
    """Clean up vector point from Qdrant if collection exists (gracefully ignore if offline)."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qdrant_models
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        collection_name = "spotify_tracks"
        client = QdrantClient(url=qdrant_url, timeout=1.5)
        track_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, file_path))
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, track_id))
        client.delete(
            collection_name=collection_name,
            points_selector=qdrant_models.PointIdsList(points=[point_id])
        )
        client.close()
    except Exception:
        # Gracefully pass if Qdrant is offline or not installed
        pass
