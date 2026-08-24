# -*- coding: utf-8 -*-
# Written by GD Studio / Antigravity AI
# Ported & Adapted from Songloft Project (songloft/internal/services/fingerprint.go)
# Date: 2026-08-24
#
# Chromaprint Acoustic Fingerprint Engine & Background Task Service
# Computes acoustic fingerprints via ffmpeg chromaprint muxer with duration guard clustering.

import os
import re
import time
import shutil
import asyncio
import subprocess
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from infer.library_db import library_db


DURATION_REGEX = re.compile(r"Duration:\s+(\d+):(\d+):(\d+)\.(\d+)")
FINGERPRINT_SAMPLE_SECONDS = 120.0
FINGERPRINT_TIMEOUT = 30.0
DUPLICATE_DURATION_TOLERANCE = 30.0

_chromaprint_available: Optional[bool] = None
_ffmpeg_path: Optional[str] = None


def get_ffmpeg_path() -> Optional[str]:
    """Finds ffmpeg executable in PATH or standard locations."""
    global _ffmpeg_path
    if _ffmpeg_path:
        return _ffmpeg_path
    path = shutil.which("ffmpeg")
    if path:
        _ffmpeg_path = path
        return path
    # Common Windows fallback locations
    candidates = [
        "ffmpeg",
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "ffmpeg", "bin", "ffmpeg.exe"),
        "C:\\ffmpeg\\bin\\ffmpeg.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            _ffmpeg_path = c
            return c
    return None


def is_chromaprint_available() -> bool:
    """Checks if ffmpeg supports the chromaprint muxer and caches the result."""
    global _chromaprint_available
    if _chromaprint_available is not None:
        return _chromaprint_available

    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        _chromaprint_available = False
        return False

    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-muxers"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5.0,
            encoding="utf-8",
            errors="ignore"
        )
        if proc.returncode == 0 and "chromaprint" in proc.stdout.lower():
            _chromaprint_available = True
        else:
            _chromaprint_available = False
    except Exception as e:
        print(f"[Fingerprint] Check chromaprint error: {e}")
        _chromaprint_available = False

    return _chromaprint_available


def parse_duration_from_stderr(stderr_text: str) -> float:
    """Parses Duration: HH:MM:SS.frac from ffmpeg output."""
    match = DURATION_REGEX.search(stderr_text)
    if not match:
        return 0.0
    try:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        frac_str = match.group(4)
        frac = float("0." + frac_str) if frac_str else 0.0
        return hours * 3600 + minutes * 60 + seconds + frac
    except Exception:
        return 0.0


def extract_fingerprint(
    file_path: str,
    start_seconds: float = 0.0,
    end_seconds: float = 0.0
) -> Tuple[Optional[str], float, Optional[str]]:
    """
    Calls ffmpeg chromaprint muxer to extract audio fingerprint.
    Returns: (fingerprint_base64, full_duration_seconds, error_message)
    """
    if not os.path.exists(file_path):
        return None, 0.0, "File not found"

    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        return None, 0.0, "ffmpeg executable not found"

    args = [ffmpeg, "-hide_banner"]
    if start_seconds > 0:
        args.extend(["-ss", f"{start_seconds:.3f}"])

    args.extend(["-i", file_path, "-map", "0:a:0", "-map_metadata", "-1"])

    sample = FINGERPRINT_SAMPLE_SECONDS
    if end_seconds > start_seconds:
        track_len = end_seconds - start_seconds
        if 0 < track_len < sample:
            sample = track_len

    args.extend([
        "-t", f"{sample:.3f}",
        "-f", "chromaprint",
        "-fp_format", "base64",
        "-"
    ])

    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=FINGERPRINT_TIMEOUT,
            encoding="utf-8",
            errors="ignore"
        )
        if proc.returncode != 0:
            return None, 0.0, f"ffmpeg error: {proc.stderr[:200]}"

        fingerprint = proc.stdout.strip().split("\n")[0].strip()
        if not fingerprint:
            return None, 0.0, "ffmpeg returned empty fingerprint"

        if end_seconds > start_seconds:
            track_len = end_seconds - start_seconds
            return fingerprint, track_len, None

        duration = parse_duration_from_stderr(proc.stderr)
        if start_seconds > 0 and duration > start_seconds:
            return fingerprint, duration - start_seconds, None

        return fingerprint, duration, None

    except subprocess.TimeoutExpired:
        return None, 0.0, "ffmpeg timeout (30s)"
    except Exception as e:
        return None, 0.0, str(e)


def cluster_by_fingerprint_duration(
    songs: List[Dict[str, Any]],
    tolerance: float = DUPLICATE_DURATION_TOLERANCE
) -> List[List[Dict[str, Any]]]:
    """
    Ported from Songloft's clusterByFingerprintDuration:
    Clusters songs with identical fingerprint by full duration.
    Only returns clusters with >= 2 songs.

    Guards:
    - Conservative tolerance to avoid false positives (e.g. audiobooks with identical 120s intro)
    - Adjacent chaining: 300s, 301.5s, 303s correctly cluster together
    - Unknown duration (duration = 0) is not split, avoiding breaking real duplicate pairs
    """
    if len(songs) < 2:
        return []

    def get_fp_dur(s: Dict[str, Any]) -> float:
        d = s.get("fingerprint_duration") or s.get("duration") or 0.0
        return float(d)

    sorted_songs = sorted(songs, key=get_fp_dur)

    clusters: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = [sorted_songs[0]]

    for i in range(1, len(sorted_songs)):
        prev = sorted_songs[i - 1]
        cur = sorted_songs[i]

        prev_dur = get_fp_dur(prev)
        cur_dur = get_fp_dur(cur)

        splittable = (prev_dur > 0 and cur_dur > 0 and (cur_dur - prev_dur) > tolerance)

        if not splittable:
            current.append(cur)
        else:
            if len(current) > 1:
                clusters.append(current)
            current = [cur]

    if len(current) > 1:
        clusters.append(current)

    return clusters


class FingerprintService:
    """
    Async background fingerprint calculation service.
    Features concurrency limits, progress tracking, cancellation, and modes (missing, recompute_all, retry_failed).
    """
    def __init__(self):
        self.is_running = False
        self.cancel_requested = False
        self.status = "idle"  # idle, running, done, cancelled
        self.computed = 0
        self.failed = 0
        self.total = 0
        self.logs: List[str] = []
        self._executor: Optional[ThreadPoolExecutor] = None

    def add_log(self, text: str):
        self.logs.append(text)
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]

    def get_progress(self) -> Dict[str, Any]:
        percent = round(self.computed * 100 / max(1, self.total), 1) if self.total > 0 else 0.0
        return {
            "status": self.status,
            "is_running": self.is_running,
            "computed": self.computed,
            "failed": self.failed,
            "total": self.total,
            "percent": percent,
            "is_available": is_chromaprint_available()
        }

    def cancel(self):
        if self.is_running:
            self.cancel_requested = True
            self.status = "cancelled"
            self.add_log("🛑 收到中止指纹计算请求...")

    async def start(self, mode: str = "missing") -> Dict[str, Any]:
        """
        Modes:
        - 'missing': only songs without fingerprint and not attempted
        - 'recompute_all': clears all fingerprints and recomputes all
        - 'retry_failed': resets failed attempts and retries them
        """
        if self.is_running:
            return {"status": "error", "message": "指纹计算任务正在运行中"}

        if not is_chromaprint_available():
            return {"status": "error", "message": "系统未检测到支持 chromaprint 的 ffmpeg"}

        self.is_running = True
        self.cancel_requested = False
        self.status = "running"
        self.computed = 0
        self.failed = 0
        self.logs = []

        if mode == "recompute_all":
            self.add_log("-> 正在清空数据库中所有旧指纹...")
            library_db.clear_all_fingerprints()
        elif mode == "retry_failed":
            self.add_log("-> 正在重置先前失败项的已尝试标记...")
            library_db.reset_failed_fingerprints()

        to_process = library_db.list_without_fingerprint()
        self.total = len(to_process)

        if self.total == 0:
            self.is_running = False
            self.status = "done"
            self.add_log("✓ 没有需要计算指纹的音频文件。")
            return {"status": "ok", "total": 0, "message": "无待处理项"}

        self.add_log(f"-> 启动声学指纹并行提取，待处理曲目: {self.total} 首 (模式: {mode})")
        asyncio.create_task(self._run_compute(to_process))
        return {"status": "ok", "total": self.total, "message": f"任务已启动，共 {self.total} 首"}

    async def _run_compute(self, items: List[Dict[str, Any]]):
        # Limit worker count to max 4 to avoid 100% CPU lock
        cpu_count = os.cpu_count() or 4
        workers = min(4, max(1, cpu_count // 2))
        self._executor = ThreadPoolExecutor(max_workers=workers)
        loop = asyncio.get_event_loop()

        def _process_item(item: Dict[str, Any]) -> Tuple[bool, str]:
            path = item["local_path"]
            if not os.path.exists(path):
                # Transient disk unreachability: skip without permanently marking as failed
                return False, "unreachable"

            fp, dur, err = extract_fingerprint(path)
            now = time.time()
            if err or not fp:
                library_db.mark_fingerprint_attempted(path, now)
                return False, f"err: {err}"
            else:
                library_db.update_fingerprint(path, fp, dur, now)
                return True, "ok"

        try:
            for i, item in enumerate(items):
                if self.cancel_requested:
                    self.status = "cancelled"
                    self.add_log(f"🛑 指纹计算已手动中止。已完成 {self.computed} 首，失败 {self.failed} 首。")
                    break

                ok, msg = await loop.run_in_executor(self._executor, _process_item, item)
                if ok:
                    self.computed += 1
                else:
                    self.failed += 1

                if (i + 1) % 10 == 0 or (i + 1) == len(items):
                    pct = round((self.computed + self.failed) * 100 / max(1, self.total), 1)
                    self.add_log(f"-> 进度: {self.computed + self.failed}/{self.total} ({pct}%) - 成功: {self.computed}, 失败: {self.failed}")

            if not self.cancel_requested:
                self.status = "done"
                self.add_log(f"✓ 声学指纹计算完成！成功提取 {self.computed} 首，失败/跳过 {self.failed} 首。")

        except Exception as e:
            self.status = "error"
            self.add_log(f"❌ 指纹计算服务异常: {e}")
        finally:
            self.is_running = False
            if self._executor:
                self._executor.shutdown(wait=False)


fingerprint_service = FingerprintService()
