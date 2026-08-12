# -*- coding: utf-8 -*-
# Written by GD Studio / Antigravity AI
# Date: 2026-08-12
#
# Fused Multi-Source Metadata Scraping Engine & Mutagen ID3/FLAC Persistence
# Integrates 5-platform aggregation engine (NetEase, QQ Music, Kugou, Soda, Apple Music)
# with mutagen tag writing, embedded artwork, LRC lyric saving, and SQLite indexing.

import os
import sys
import re
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from mutagen import File as MutagenFile
from mutagen.id3 import ID3, TIT2, TPE1, TALB, TDRC, TCON, TRCK, APIC, USLT
from mutagen.flac import FLAC, Picture

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
infer_path = os.path.join(project_root, "infer")
if infer_path not in sys.path:
    sys.path.insert(0, infer_path)

from infer.library_db import library_db

# Import search aggregator
from infer.search import aggregate
from infer.search.sources import SOURCE_REGISTRY

logger = logging.getLogger("embeat")

# LDDC Lyrics fallback scrapers
AVAILABLE_SCRAPERS = []

try:
    from LDDC.core.api.lyrics.qm import QMScraper
    AVAILABLE_SCRAPERS.append(QMScraper())
except Exception as e:
    logger.debug(f"[Scraper] QMScraper load notice: {e}")

try:
    from LDDC.core.api.lyrics.ne import NEScraper
    AVAILABLE_SCRAPERS.append(NEScraper())
except Exception as e:
    logger.debug(f"[Scraper] NEScraper load notice: {e}")

try:
    from LDDC.core.api.lyrics.lrclib import LRCLIBScraper
    AVAILABLE_SCRAPERS.append(LRCLIBScraper())
except Exception as e:
    logger.debug(f"[Scraper] LRCLIBScraper load notice: {e}")

try:
    from LDDC.core.api.lyrics.kg import KGScraper
    AVAILABLE_SCRAPERS.append(KGScraper())
except Exception as e:
    logger.debug(f"[Scraper] KGScraper load notice: {e}")

try:
    from LDDC.core.algorithm import match_best_lyric
    _MATCH_BEST_AVAILABLE = True
except Exception as e:
    _MATCH_BEST_AVAILABLE = False

_LDDC_AVAILABLE = len(AVAILABLE_SCRAPERS) > 0 and _MATCH_BEST_AVAILABLE


def build_keyword(title: str, artist: str, file_path: str = "") -> str:
    """
    Constructs search keyword:
    Prefers clean filename / smart mutagen info.
    Strips leading track number prefixes like 01., [02]-, and parses title & artist.
    """
    clean_artist = "" if not artist or artist.lower() == "unknown artist" else artist.strip()
    clean_title = re.sub(r"^\d+[\.\s\-_]+", "", title or "").strip()

    if file_path:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        clean_name = re.sub(r"^(?:\d{1,3}|\[?\d{1,3}\]?)[\s.\-_·]*(?=[^\s\d.])", "", base_name).strip()

        if not clean_artist:
            if " - " in clean_name:
                parts = clean_name.split(" - ", 1)
                clean_artist, clean_title = parts[0].strip(), parts[1].strip()
            elif "-" in clean_name and not clean_name.startswith("-"):
                parts = clean_name.split("-", 1)
                p1, p2 = parts[0].strip(), parts[1].strip()
                if len(p2) <= 10 and not any(c.isdigit() for c in p2):
                    clean_title, clean_artist = p1, p2
                else:
                    clean_artist, clean_title = p1, p2
        elif not clean_title:
            clean_title = clean_name

    return " ".join(x for x in [clean_title, clean_artist] if x).strip()


def _candidate_complete(item: dict, wants: List[str]) -> bool:
    """Checks if search candidate contains non-empty values for required fields."""
    key_map = {"cover": "picUrl", "year": "date"}
    for field in wants:
        key = key_map.get(field, field)
        if not item.get(key):
            return False
    return True


def contains_cjk(text: str) -> bool:
    """Checks if string contains CJK / Chinese characters."""
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def score_candidate(item: dict, query_keyword: str, query_artist: str = "") -> float:
    """
    Ranks candidates based on CJK language alignment, title matching, and artist similarity.
    Prevents picking English translated titles (e.g. Burning/Anesthesia) when query is Chinese.
    """
    score = 0.0
    cand_title = (item.get("title") or "").strip()
    cand_artist = (item.get("artist") or "").strip()

    q_has_cjk = contains_cjk(query_keyword)
    c_has_cjk = contains_cjk(cand_title)

    # 1. CJK Language Alignment Bonus
    if q_has_cjk and c_has_cjk:
        score += 80.0
    elif q_has_cjk and not c_has_cjk:
        score -= 50.0  # Penalize non-Chinese primary titles for Chinese queries

    # 2. Title Match Bonus
    clean_q = re.sub(r"^\d+[\.\s\-_]+", "", query_keyword).strip().lower()
    clean_c = re.sub(r"^\d+[\.\s\-_]+", "", cand_title).strip().lower()

    if clean_c == clean_q:
        score += 100.0
    elif clean_q in clean_c or clean_c in clean_q:
        score += 40.0

    # 3. Artist Match Bonus
    if query_artist and query_artist.lower() not in ["unknown artist", "unknown", ""]:
        q_art = query_artist.lower()
        c_art = cand_artist.lower()
        if q_art in c_art or c_art in q_art:
            score += 80.0

    # 4. Cover & Date Completeness Bonus
    if item.get("picUrl"):
        score += 10.0
    if item.get("date"):
        score += 10.0

    return score


def _search_first_complete(keyword: str, sources: List[str], wants: List[str], page_size: int = 10, timeout: int = 8, artist: str = "") -> Tuple[Optional[dict], List[dict]]:
    """
    Searches across platforms, scores candidates by relevance & CJK alignment,
    and returns the best complete candidate.
    """
    all_flat = []
    for platform in sources:
        groups, _total = aggregate.search_songs(
            keyword, [platform], page=1, page_size=page_size, timeout=timeout
        )
        items = []
        for g in groups:
            for item in g.get("items") or []:
                item = dict(item)
                item["_platform"] = g["pluginId"]
                item["_score"] = score_candidate(item, keyword, artist)
                items.append(item)
        all_flat.extend(items)

    # Sort candidates by relevance score
    all_flat.sort(key=lambda x: x.get("_score", 0), reverse=True)

    for item in all_flat:
        if _candidate_complete(item, wants):
            return item, all_flat

    return (all_flat[0] if all_flat else None), all_flat


def _complement_candidate(flat: List[dict], wants: List[str]) -> Optional[dict]:
    """
    Merges a complete candidate from multi-platform search results.
    Takes flat[0] as base.
    Missing title/artist is complemented from subsequent candidates.
    Other missing fields (album/cover/year/trackNumber) are complemented ONLY IF
    candidate title and artist match base title and artist exactly (same song).
    """
    if not flat:
        return None
    base = dict(flat[0])
    field_map = {
        "album": "album",
        "cover": "picUrl",
        "year": "date",
        "trackNumber": "trackNumber",
        "discNumber": "discNumber",
    }
    for field in ("title", "artist"):
        if field not in wants or base.get(field):
            continue
        for cand in flat[1:]:
            if cand.get(field):
                base[field] = cand[field]
                break

    base_title = (base.get("title") or "").strip()
    base_artist = (base.get("artist") or "").strip()

    for field, key in field_map.items():
        if field not in wants or base.get(key):
            continue
        for cand in flat[1:]:
            if (cand.get("title") or "").strip() != base_title:
                continue
            if (cand.get("artist") or "").strip() != base_artist:
                continue
            if cand.get(key):
                base[key] = cand[key]
                break
    return base


def sync_fetch_online_metadata(title: str, artist: str, file_path: str = "") -> Dict[str, Any]:
    """
    Synchronous multi-source metadata aggregator:
    Queries NetEase, QQ Music, Kugou, Soda, and Apple Music using early-stop & complement logic.
    """
    result = {
        "title": title,
        "artist": artist,
        "album": "",
        "year": None,
        "genre": "",
        "cover_url": "",
        "track_number": None,
        "disc_number": None,
        "platform": ""
    }

    keyword = build_keyword(title, artist, file_path=file_path)
    if not keyword:
        return result

    sources = ["netease", "qq", "kugou", "soda", "apple"]
    wants = ["title", "artist", "album", "cover", "year"]

    try:
        candidate, flat = _search_first_complete(keyword, sources, wants, artist=artist)
        if candidate is None:
            candidate = _complement_candidate(flat, wants)

        if candidate:
            result["title"] = candidate.get("title") or title
            result["artist"] = candidate.get("artist") or artist
            result["album"] = candidate.get("album") or ""
            result["cover_url"] = candidate.get("picUrl") or ""
            result["platform"] = candidate.get("_platform") or ""

            # Upgrade iTunes cover resolution if applicable
            if "100x100bb.jpg" in result["cover_url"]:
                result["cover_url"] = result["cover_url"].replace("100x100bb.jpg", "600x600bb.jpg")

            date_str = str(candidate.get("date") or candidate.get("year") or "")
            if date_str:
                match = re.search(r"\b(19\d\d|20\d\d)\b", date_str)
                if match:
                    result["year"] = int(match.group(1))

            tn = candidate.get("trackNumber") or candidate.get("track_no")
            if tn and str(tn).isdigit():
                result["track_number"] = int(tn)

            dn = candidate.get("discNumber") or candidate.get("disc_no")
            if dn and str(dn).isdigit():
                result["disc_number"] = int(dn)

            if candidate.get("genre"):
                result["genre"] = candidate.get("genre")

            result["candidate"] = candidate

    except Exception as e:
        logger.error(f"[Scraper] Exception during aggregated metadata search for '{keyword}': {e}")

    return result


async def fetch_online_metadata(title: str, artist: str, file_path: str = "") -> Dict[str, Any]:
    """Async wrapper for multi-source metadata search."""
    return await asyncio.to_thread(sync_fetch_online_metadata, title, artist, file_path)


def sync_fetch_lyrics_aggregated(title: str, artist: str, file_path: str = "") -> Optional[str]:
    """
    Synchronous lyric fetcher across 5 aggregate platforms (NetEase, QQ, Kugou, Soda, Apple).
    """
    clean_title = re.sub(r'^\d+[\.\s\-_]+', '', title or '').strip()
    clean_artist = "" if not artist or artist == "Unknown Artist" else artist.strip()

    song_obj = {
        "title": clean_title,
        "artist": clean_artist,
        "album": "",
        "duration": 0
    }

    platforms = ["netease", "qq", "kugou", "soda", "apple"]
    for p in platforms:
        try:
            res = aggregate.get_lyrics(p, song_obj, timeout=6)
            if res and res.get("rawPlainLrc") and len(res["rawPlainLrc"].strip()) > 10:
                logger.info(f"[Scraper Log] Fetched lyrics successfully from platform: {p}")
                return res["rawPlainLrc"]
        except Exception as e:
            logger.debug(f"[Scraper Log] Platform {p} lyric fetch error: {e}")

    return None


async def _fetch_lyrics_fallback(title: str, artist: str) -> Optional[str]:
    """Fallback lyric fetcher via direct HTTP requests."""
    import requests
    clean_title = re.sub(r'^\d+[\.\s\-_]+', '', title or '').strip()
    clean_artist = "" if not artist or artist == "Unknown Artist" else artist.strip()

    queries = []
    if clean_artist:
        queries.append(f"{clean_title} {clean_artist}")
    queries.append(clean_title)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0.0.0 Safari/537.36"
    }

    for q in queries:
        try:
            search_res = await asyncio.to_thread(
                requests.get,
                "http://music.163.com/api/search/get/web",
                params={"csrf_token": "", "type": 1, "offset": 0, "total": "true", "limit": 3, "s": q},
                headers=headers,
                timeout=5
            )
            if search_res.status_code == 200:
                data = search_res.json()
                songs = data.get("result", {}).get("songs", [])
                if songs:
                    song_id = songs[0]["id"]
                    lrc_res = await asyncio.to_thread(
                        requests.get,
                        "http://music.163.com/api/song/lyric",
                        params={"os": "pc", "id": song_id, "lv": -1, "kv": -1, "tv": -1},
                        headers=headers,
                        timeout=5
                    )
                    if lrc_res.status_code == 200:
                        lrc_data = lrc_res.json()
                        lrc_str = lrc_data.get("lrc", {}).get("lyric", "")
                        if lrc_str and len(lrc_str.strip()) > 10:
                            return lrc_str
        except Exception:
            continue

    return None


async def fetch_lyrics_lddc(title: str, artist: str, duration: float = 0.0, file_path: str = "") -> Optional[str]:
    """
    Fetch lyrics using multi-source aggregate engine -> LDDC -> direct HTTP fallback.
    """
    # 1. Try aggregated 5-platform engine
    lrc_agg = await asyncio.to_thread(sync_fetch_lyrics_aggregated, title, artist, file_path)
    if lrc_agg:
        return lrc_agg

    # 2. Try LDDC scrapers
    clean_title = re.sub(r'^\d+[\.\s\-_]+', '', title or '').strip()
    clean_artist = "" if not artist or artist == "Unknown Artist" else artist.strip()

    if _LDDC_AVAILABLE and AVAILABLE_SCRAPERS:
        query = f"{clean_title} {clean_artist}".strip()
        candidates = []

        for scraper in AVAILABLE_SCRAPERS:
            try:
                res = await asyncio.to_thread(scraper.search, query, page=1)
                if res and hasattr(res, 'results') and res.results:
                    candidates.extend(res.results)
                elif isinstance(res, list):
                    candidates.extend(res)
            except Exception:
                continue

        if candidates:
            try:
                best_match = match_best_lyric(candidates, title=clean_title, artist=clean_artist or "Unknown", duration=duration)
                if best_match and hasattr(best_match, 'fetch_lyric'):
                    lrc_text = await asyncio.to_thread(best_match.fetch_lyric)
                    if lrc_text:
                        return lrc_text
            except Exception:
                pass

    # 3. Fallback to direct HTTP
    return await _fetch_lyrics_fallback(clean_title, clean_artist)


async def apply_scrape_to_file(local_path: str, metadata: Dict[str, Any], lrc_text: Optional[str] = None) -> bool:
    """
    Writes scraped ID3 / FLAC tags and embedded cover image to local audio file via mutagen,
    saves .lrc file to local directory, and updates SQLite database.
    """
    if not os.path.exists(local_path):
        return False

    folder_dir = os.path.dirname(local_path)
    base_name = os.path.splitext(os.path.basename(local_path))[0]
    lrc_path = os.path.join(folder_dir, f"{base_name}.lrc")
    cover_file_path = os.path.join(folder_dir, "cover.jpg")

    # Download artwork image with browser User-Agent
    cover_data = None
    if metadata.get("cover_url"):
        try:
            if _HTTPX_AVAILABLE:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0.0.0 Safari/537.36"}
                async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
                    res = await client.get(metadata["cover_url"])
                    if res.status_code == 200 and len(res.content) > 32:
                        cover_data = res.content
                        with open(cover_file_path, "wb") as f:
                            f.write(cover_data)
        except Exception as e:
            logger.error(f"[Scraper] Download cover image failed for {metadata.get('cover_url')}: {e}")

    # Save .lrc file
    if lrc_text:
        try:
            with open(lrc_path, "w", encoding="utf-8") as f:
                f.write(lrc_text)
        except Exception as e:
            logger.error(f"[Scraper] Failed to save .lrc file: {e}")

    # Write mutagen embedded tags
    ext = os.path.splitext(local_path)[1].lower()
    try:
        if ext == ".mp3":
            try:
                tags = ID3(local_path)
            except Exception:
                tags = ID3()
            
            if metadata.get("title"):
                tags["TIT2"] = TIT2(encoding=3, text=metadata["title"])
            if metadata.get("artist"):
                tags["TPE1"] = TPE1(encoding=3, text=metadata["artist"])
            if metadata.get("album"):
                tags["TALB"] = TALB(encoding=3, text=metadata["album"])
            if metadata.get("genre"):
                tags["TCON"] = TCON(encoding=3, text=metadata["genre"])
            if metadata.get("year"):
                tags["TDRC"] = TDRC(encoding=3, text=str(metadata["year"]))
            if metadata.get("track_number"):
                tags["TRCK"] = TRCK(encoding=3, text=str(metadata["track_number"]))
            if cover_data:
                tags["APIC"] = APIC(encoding=3, mime="image/jpeg", type=3, desc="Cover", data=cover_data)
            if lrc_text:
                tags["USLT"] = USLT(encoding=3, lang="eng", desc="Lyrics", text=lrc_text)
            tags.save(local_path)

        elif ext == ".flac":
            audio = FLAC(local_path)
            if metadata.get("title"):
                audio["TITLE"] = metadata["title"]
            if metadata.get("artist"):
                audio["ARTIST"] = metadata["artist"]
            if metadata.get("album"):
                audio["ALBUM"] = metadata["album"]
            if metadata.get("genre"):
                audio["GENRE"] = metadata["genre"]
            if metadata.get("year"):
                audio["DATE"] = str(metadata["year"])
            if metadata.get("track_number"):
                audio["TRACKNUMBER"] = str(metadata["track_number"])

            if cover_data:
                pic = Picture()
                pic.type = 3
                pic.mime = "image/jpeg"
                pic.desc = "Cover"
                pic.data = cover_data
                audio.clear_pictures()
                audio.add_picture(pic)
            audio.save()
    except Exception as e:
        logger.error(f"[Scraper] Mutagen tag writing exception for {local_path}: {e}")

    # Update SQLite database
    track_record = {
        "local_path": local_path,
        "track_name": metadata.get("title") or base_name,
        "artist_name": metadata.get("artist") or "Unknown Artist",
        "album_name": metadata.get("album") or "Unknown Album",
        "genre": metadata.get("genre"),
        "year": metadata.get("year"),
        "cover_path": cover_file_path if cover_data else None,
        "lyrics_path": lrc_path if lrc_text else None,
        "scraped_at": time.time()
    }
    library_db.upsert_track(track_record)
    return True
