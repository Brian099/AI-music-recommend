# -*- coding: utf-8 -*-
# Written by GD Studio / Antigravity AI
# Date: 2026-08-07
#
# SQLite Database Storage & Indexing Engine for Local Music Library
# Manages relational schemas for tracks, artists, albums, folders, metadata, and quality badges.

import os
import sqlite3
import time
from typing import Optional, List, Dict, Any, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(project_root, "data", "library.db")


class LibraryDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initialize relational database tables and indices."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Tracks table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracks (
                track_id TEXT PRIMARY KEY,
                local_path TEXT UNIQUE NOT NULL,
                track_name TEXT NOT NULL,
                artist_name TEXT NOT NULL,
                album_name TEXT,
                duration REAL DEFAULT 0.0,
                bitrate INTEGER DEFAULT 0,
                sample_rate INTEGER DEFAULT 0,
                format TEXT,
                file_size INTEGER DEFAULT 0,
                mtime REAL DEFAULT 0.0,
                md5 TEXT,
                genre TEXT,
                year INTEGER,
                track_number INTEGER,
                cover_path TEXT,
                lyrics_path TEXT,
                is_true_lossless INTEGER DEFAULT NULL,
                cutoff_freq INTEGER DEFAULT NULL,
                quality_rating TEXT,
                scraped_at REAL DEFAULT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            """)

            # Artists table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS artists (
                artist_name TEXT PRIMARY KEY,
                track_count INTEGER DEFAULT 0,
                album_count INTEGER DEFAULT 0,
                genre TEXT,
                cover_path TEXT,
                updated_at REAL NOT NULL
            );
            """)

            # Albums table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS albums (
                album_key TEXT PRIMARY KEY,
                album_name TEXT NOT NULL,
                artist_name TEXT NOT NULL,
                track_count INTEGER DEFAULT 0,
                year INTEGER,
                cover_path TEXT,
                updated_at REAL NOT NULL
            );
            """)

            # Folders table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS folders (
                folder_path TEXT PRIMARY KEY,
                parent_path TEXT NOT NULL,
                folder_name TEXT NOT NULL,
                track_count INTEGER DEFAULT 0,
                updated_at REAL NOT NULL
            );
            """)

            # Indexes for ultra-fast querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracks_album ON tracks(album_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracks_md5 ON tracks(md5);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracks_path ON tracks(local_path);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_folders_parent ON folders(parent_path);")

            conn.commit()

    def upsert_track(self, track: Dict[str, Any]):
        """Upsert a single track and update artist/album/folder relationships."""
        now = time.time()
        local_path = track["local_path"]
        folder_path = os.path.dirname(local_path)
        parent_path = os.path.dirname(folder_path)
        folder_name = os.path.basename(folder_path) or folder_path
        album_name = track.get("album_name") or "Unknown Album"
        artist_name = track.get("artist_name") or "Unknown Artist"
        album_key = f"{artist_name} - {album_name}"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO tracks (
                track_id, local_path, track_name, artist_name, album_name,
                duration, bitrate, sample_rate, format, file_size, mtime, md5,
                genre, year, track_number, cover_path, lyrics_path,
                is_true_lossless, cutoff_freq, quality_rating, scraped_at,
                created_at, updated_at
            ) VALUES (
                :track_id, :local_path, :track_name, :artist_name, :album_name,
                :duration, :bitrate, :sample_rate, :format, :file_size, :mtime, :md5,
                :genre, :year, :track_number, :cover_path, :lyrics_path,
                :is_true_lossless, :cutoff_freq, :quality_rating, :scraped_at,
                :created_at, :updated_at
            ) ON CONFLICT(local_path) DO UPDATE SET
                track_name = excluded.track_name,
                artist_name = excluded.artist_name,
                album_name = excluded.album_name,
                duration = excluded.duration,
                bitrate = excluded.bitrate,
                sample_rate = excluded.sample_rate,
                format = excluded.format,
                file_size = excluded.file_size,
                mtime = excluded.mtime,
                md5 = COALESCE(excluded.md5, tracks.md5),
                genre = COALESCE(excluded.genre, tracks.genre),
                year = COALESCE(excluded.year, tracks.year),
                track_number = COALESCE(excluded.track_number, tracks.track_number),
                cover_path = COALESCE(excluded.cover_path, tracks.cover_path),
                lyrics_path = COALESCE(excluded.lyrics_path, tracks.lyrics_path),
                is_true_lossless = COALESCE(excluded.is_true_lossless, tracks.is_true_lossless),
                cutoff_freq = COALESCE(excluded.cutoff_freq, tracks.cutoff_freq),
                quality_rating = COALESCE(excluded.quality_rating, tracks.quality_rating),
                scraped_at = COALESCE(excluded.scraped_at, tracks.scraped_at),
                updated_at = excluded.updated_at;
            """, {
                "track_id": track.get("track_id"),
                "local_path": local_path,
                "track_name": track.get("track_name", "Unknown Track"),
                "artist_name": artist_name,
                "album_name": album_name,
                "duration": track.get("duration", 0.0),
                "bitrate": track.get("bitrate", 0),
                "sample_rate": track.get("sample_rate", 0),
                "format": track.get("format", os.path.splitext(local_path)[1].lstrip('.')),
                "file_size": track.get("file_size", 0),
                "mtime": track.get("mtime", 0.0),
                "md5": track.get("md5"),
                "genre": track.get("genre"),
                "year": track.get("year"),
                "track_number": track.get("track_number"),
                "cover_path": track.get("cover_path"),
                "lyrics_path": track.get("lyrics_path"),
                "is_true_lossless": track.get("is_true_lossless"),
                "cutoff_freq": track.get("cutoff_freq"),
                "quality_rating": track.get("quality_rating"),
                "scraped_at": track.get("scraped_at"),
                "created_at": now,
                "updated_at": now
            })

            # Upsert folder info
            cursor.execute("""
            INSERT INTO folders (folder_path, parent_path, folder_name, track_count, updated_at)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(folder_path) DO UPDATE SET
                track_count = (SELECT COUNT(*) FROM tracks WHERE local_path LIKE folders.folder_path || '/%'),
                updated_at = ?;
            """, (folder_path, parent_path, folder_name, now, now))

            # Upsert artist info
            cursor.execute("""
            INSERT INTO artists (artist_name, track_count, album_count, genre, cover_path, updated_at)
            VALUES (?, 1, 1, ?, ?, ?)
            ON CONFLICT(artist_name) DO UPDATE SET
                track_count = (SELECT COUNT(*) FROM tracks WHERE artist_name = excluded.artist_name),
                album_count = (SELECT COUNT(DISTINCT album_name) FROM tracks WHERE artist_name = excluded.artist_name),
                genre = COALESCE(excluded.genre, artists.genre),
                cover_path = COALESCE(excluded.cover_path, artists.cover_path),
                updated_at = ?;
            """, (artist_name, track.get("genre"), track.get("cover_path"), now, now))

            # Upsert album info
            cursor.execute("""
            INSERT INTO albums (album_key, album_name, artist_name, track_count, year, cover_path, updated_at)
            VALUES (?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(album_key) DO UPDATE SET
                track_count = (SELECT COUNT(*) FROM tracks WHERE album_name = excluded.album_name AND artist_name = excluded.artist_name),
                year = COALESCE(excluded.year, albums.year),
                cover_path = COALESCE(excluded.cover_path, albums.cover_path),
                updated_at = ?;
            """, (album_key, album_name, artist_name, track.get("year"), track.get("cover_path"), now, now))

            conn.commit()

    def get_indexed_paths_with_mtime(self) -> Dict[str, float]:
        """Returns {local_path: mtime} map for O(1) instant resumable checkpoint skipping."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT local_path, mtime FROM tracks;")
            return {row["local_path"]: row["mtime"] for row in cursor.fetchall()}

    def get_all_tracks(self, limit: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks ORDER BY track_name ASC LIMIT ? OFFSET ?;", (limit, offset))
            return [dict(row) for row in cursor.fetchall()]

    def get_artists(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM artists ORDER BY track_count DESC, artist_name ASC;")
            return [dict(row) for row in cursor.fetchall()]

    def get_albums(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM albums ORDER BY album_name ASC;")
            return [dict(row) for row in cursor.fetchall()]

    def get_tracks_by_artist(self, artist_name: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks WHERE artist_name = ? ORDER BY album_name, track_number, track_name;", (artist_name,))
            return [dict(row) for row in cursor.fetchall()]

    def get_tracks_by_album(self, album_name: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks WHERE album_name = ? ORDER BY track_number, track_name;", (album_name,))
            return [dict(row) for row in cursor.fetchall()]

    def get_track_by_path(self, local_path: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks WHERE local_path = ?;", (local_path,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_track(self, local_path: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM tracks WHERE local_path = ?;", (local_path,))
            conn.commit()


library_db = LibraryDatabase()
