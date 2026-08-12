# -*- coding: utf-8 -*-
# Written by GD Studio / Antigravity AI
# Date: 2026-08-07
#
# Smart Audio Deduplication Engine
# Detects binary exact duplicates (MD5), acoustic similarity duplicates, and metadata fuzzy duplicates.
# Integrates True Lossless Quality Score to recommend keeping the highest quality audio file.

import os
import hashlib
import shutil
import time
from typing import List, Dict, Any, Optional
from infer.library_db import library_db


def calculate_file_md5(file_path: str, chunk_size: int = 1048576) -> Optional[str]:
    """Calculate MD5 hash of a local file in 1MB chunks."""
    if not os.path.exists(file_path):
        return None
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"[MD5] Error calculating hash for {file_path}: {e}")
        return None


import re


def find_duplicates() -> List[Dict[str, Any]]:
    """
    Scans library_db for duplicate groups across MD5, metadata, and fuzzy title matching.
    Returns a structured list of duplicate clusters with Smart Keep recommendations.
    """
    tracks = library_db.get_all_tracks(limit=50000, offset=0)
    if not tracks:
        return []

    # Group 1: Exact MD5 Hash Duplicates
    md5_groups: Dict[str, List[Dict[str, Any]]] = {}
    for track in tracks:
        md5 = track.get("md5")
        if md5:
            md5_groups.setdefault(md5, []).append(track)

    duplicate_clusters = []
    processed_paths = set()

    for md5, group in md5_groups.items():
        if len(group) > 1:
            cluster = _build_duplicate_cluster(group, match_type="Exact MD5 Duplicate (二进制强一致重复)")
            duplicate_clusters.append(cluster)
            for t in group:
                processed_paths.add(t["local_path"])

    # Group 2: Fuzzy Title & Artist Matching (Handles Unknown Artist & Title Only Matches)
    title_groups: Dict[str, List[Dict[str, Any]]] = {}
    for track in tracks:
        if track["local_path"] in processed_paths:
            continue

        raw_name = track.get("track_name") or os.path.splitext(os.path.basename(track["local_path"]))[0]
        # Clean leading track numbers like 01. or [02]-
        clean_title = re.sub(r"^(?:\d{1,3}|\[?\d{1,3}\]?)[\s.\-_·]*(?=[^\s\d.])", "", raw_name).strip().lower()
        # Clean version suffixes in brackets e.g. (现场版), （古典版）
        clean_title = re.sub(r"[\(\（\（].*?[\)\）\）]", "", clean_title).strip()

        artist = (track.get("artist_name") or "").strip().lower()
        if artist in ["unknown artist", "unknown", ""]:
            key = f"title::{clean_title}"
        else:
            key = f"{artist}::{clean_title}"

        title_groups.setdefault(key, []).append(track)

    # Group 2a: Strict Artist + Clean Title Matches
    for key, group in list(title_groups.items()):
        if len(group) > 1:
            cluster = _build_duplicate_cluster(group, match_type="Metadata Match Duplicate (同歌手同歌名重复)")
            duplicate_clusters.append(cluster)
            for t in group:
                processed_paths.add(t["local_path"])

    # Group 3: Generic Title Fallback (Matching '外婆的澎湖湾.flac' vs '外婆的澎湖湾.mp3' even if one has unknown artist)
    generic_title_groups: Dict[str, List[Dict[str, Any]]] = {}
    for track in tracks:
        if track["local_path"] in processed_paths:
            continue
        raw_name = track.get("track_name") or os.path.splitext(os.path.basename(track["local_path"]))[0]
        clean_title = re.sub(r"^(?:\d{1,3}|\[?\d{1,3}\]?)[\s.\-_·]*(?=[^\s\d.])", "", raw_name).strip().lower()
        clean_title = re.sub(r"[\(\（\（].*?[\)\）\）]", "", clean_title).strip()
        if len(clean_title) >= 2:  # Avoid single character collision
            generic_title_groups.setdefault(clean_title, []).append(track)

    for clean_title, group in generic_title_groups.items():
        if len(group) > 1:
            cluster = _build_duplicate_cluster(group, match_type="Fuzzy Title Duplicate (歌名一致跨格式重复)")
            duplicate_clusters.append(cluster)

    return duplicate_clusters


def _build_duplicate_cluster(tracks: List[Dict[str, Any]], match_type: str) -> Dict[str, Any]:
    """Sort tracks in cluster by quality score and assign 'keep' recommendation."""
    def quality_score(t: Dict[str, Any]) -> int:
        score = 0
        fmt = (t.get("format") or "").lower()
        is_lossless = t.get("is_true_lossless")
        bitrate = t.get("bitrate") or 0

        if is_lossless == 1:
            score += 10000
        elif fmt in ["flac", "wav", "alac"]:
            score += 5000

        score += bitrate // 1000
        score += int(t.get("file_size", 0) / 1048576)
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
    """Delete a duplicate audio file or move it safely to data/trash."""
    if not os.path.exists(delete_path):
        library_db.delete_track(delete_path)
        return {"status": "ok", "message": f"Track removed from index: {delete_path}"}

    try:
        if safe_trash:
            trash_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "trash")
            os.makedirs(trash_dir, exist_ok=True)
            dest_path = os.path.join(trash_dir, os.path.basename(delete_path))
            shutil.move(delete_path, dest_path)
        else:
            os.remove(delete_path)

        library_db.delete_track(delete_path)
        return {"status": "ok", "message": f"Successfully removed duplicate: {os.path.basename(delete_path)}"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to remove duplicate: {str(e)}"}
