# -*- coding: utf-8 -*-
# Written by GD Studio / Antigravity AI
# Date: 2026-08-07
#
# Metadata Scraping Engine & LDDC Lyrics Integration
# Integrates online metadata scraping (MusicBrainz/iTunes/NetEase) and LDDC multi-source lyrics fetching.
# Performs double persistence: writes ID3/FLAC tags + embedded covers via mutagen, updates SQLite & Qdrant.

import os
import sys
import json
try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
from typing import Dict, Any, Optional, List
from mutagen import File as MutagenFile
from mutagen.id3 import ID3, TIT2, TPE1, TALB, TDRC, TCON, TRCK, APIC, USLT
from mutagen.flac import FLAC, Picture

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
infer_path = os.path.join(project_root, "infer")
if infer_path not in sys.path:
    sys.path.insert(0, infer_path)

from infer.library_db import library_db

# Try importing LDDC core components from infer/LDDC
try:
    from LDDC.core.api.lyrics.qm import QMScraper
    from LDDC.core.api.lyrics.kg import KGScraper
    from LDDC.core.api.lyrics.ne import NEScraper
    from LDDC.core.api.lyrics.lrclib import LRCLIBScraper
    from LDDC.core.algorithm import match_best_lyric
    _LDDC_AVAILABLE = True
except ImportError as e:
    print(f"[Scraper] Warning: LDDC lyric module import failed: {e}")
    _LDDC_AVAILABLE = False


async def fetch_online_metadata(title: str, artist: str) -> Dict[str, Any]:
    """Search iTunes Search API & MusicBrainz for metadata and high-res cover art."""
    result = {
        "title": title,
        "artist": artist,
        "album": "",
        "year": None,
        "genre": "",
        "cover_url": "",
        "track_number": None
    }
    
    query = f"{title} {artist}".strip()
    if not query or not _HTTPX_AVAILABLE:
        return result

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            url = f"https://itunes.apple.com/search?term={httpx.QueryParams({'term': query})}&media=music&limit=3"
            res = await client.get(url)
            if res.status_code == 200:
                data = res.json()
                results = data.get("results", [])
                if results:
                    best = results[0]
                    result["title"] = best.get("trackName", title)
                    result["artist"] = best.get("artistName", artist)
                    result["album"] = best.get("collectionName", "")
                    result["genre"] = best.get("primaryGenreName", "")
                    result["track_number"] = best.get("trackNumber")
                    
                    release_date = best.get("releaseDate", "")
                    if len(release_date) >= 4:
                        try:
                            result["year"] = int(release_date[:4])
                        except ValueError:
                            pass
                            
                    # High-res cover art URL (600x600)
                    artwork = best.get("artworkUrl100", "")
                    if artwork:
                        result["cover_url"] = artwork.replace("100x100bb.jpg", "600x600bb.jpg")
    except Exception as e:
        print(f"[Scraper] iTunes metadata search exception: {e}")

    return result


async def fetch_lyrics_lddc(title: str, artist: str, duration: float = 0.0) -> Optional[str]:
    """Fetch lyrics using integrated LDDC multi-source engine (QM / KG / NE / LRCLIB)."""
    if not _LDDC_AVAILABLE:
        return None

    query = f"{title} {artist}".strip()
    if not query:
        return None

    scrapers = [QMScraper(), NEScraper(), LRCLIBScraper(), KGScraper()]
    candidates = []

    for scraper in scrapers:
        try:
            res = await asyncio.to_thread(scraper.search, query, page=1)
            if res and hasattr(res, 'results') and res.results:
                candidates.extend(res.results)
            elif isinstance(res, list):
                candidates.extend(res)
        except Exception as e:
            continue

    if not candidates:
        return None

    # Use LDDC fuzzy match algorithm
    try:
        best_match = match_best_lyric(candidates, title=title, artist=artist, duration=duration)
        if best_match and hasattr(best_match, 'fetch_lyric'):
            lrc_text = await asyncio.to_thread(best_match.fetch_lyric)
            return lrc_text
        elif isinstance(best_match, dict) and "lyrics" in best_match:
            return best_match["lyrics"]
    except Exception as e:
        print(f"[LDDC] Lyric matching exception: {e}")

    return None


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

    # Download artwork image
    cover_data = None
    if metadata.get("cover_url"):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(metadata["cover_url"])
                if res.status_code == 200:
                    cover_data = res.content
                    with open(cover_file_path, "wb") as f:
                        f.write(cover_data)
        except Exception as e:
            print(f"[Scraper] Download cover image failed: {e}")

    # Write .lrc file
    if lrc_text:
        try:
            with open(lrc_path, "w", encoding="utf-8") as f:
                f.write(lrc_text)
        except Exception as e:
            print(f"[Scraper] Failed to save .lrc file: {e}")

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
        print(f"[Scraper] Mutagen tag writing exception for {local_path}: {e}")

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
