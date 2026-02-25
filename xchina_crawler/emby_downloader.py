from __future__ import annotations

"""
从数据库读取 videos 表，按 Emby（电影库）目录结构落盘：
- 默认每次下载前刷新下载页/详情页，获取最新 m3u8（防过期）
- 下载视频（ffmpeg）、封面 cover、截图 screenshot
- 生成 Kodi/Emby 兼容的 movie.nfo
- 下载成功后写回数据库标记，避免重复下载（即使你后续把文件上传网盘并删除本地）
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any
from urllib.parse import urljoin, urlparse
import threading
from collections import deque
import sys

import requests

from .config import load_config
from .db import (
    Db,
    mark_video_download_attempt,
    mark_video_download_done,
    mark_video_download_error,
    update_video_detail,
    update_video_series,
)
from .http_client import HttpClient
from .parsers import parse_video_page


_INVALID_FS_CHARS = re.compile(r'[\\/:*?"<>|]+')
_SAFE_NAME_MAX = 120 if os.name == "nt" else 180
_SAFE_FILE_MAX = 80 if os.name == "nt" else 180
_RE_BREADCRUMB_SERIES = re.compile(r"/videos/series-([a-f0-9]{8,})(?:/\d+\.html|\.html)", re.I)
_PRINT_LOCK = threading.Lock()
_RE_TITLE_STRIP_TOKEN = re.compile(r"(小黄书|xchina)", re.I)


def _print_safe(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg)


class ProgressRenderer:
    """
    简易进度渲染器：
    - dynamic=True 且 stdout 是 TTY 时：在同一屏动态刷新多行
    - 否则：退化为普通逐行打印
    """

    def __init__(self, *, dynamic: bool) -> None:
        self._dynamic = bool(dynamic and sys.stdout.isatty())
        self._lines: dict[str, str] = {}
        self._order: list[str] = []
        self._rendered_lines = 0

    def update(self, key: str, line: str) -> None:
        if not self._dynamic:
            _print_safe(line)
            return
        if key not in self._lines:
            self._order.append(key)
        self._lines[key] = line
        self._render()

    def log(self, line: str) -> None:
        if not self._dynamic:
            _print_safe(line)
            return
        with _PRINT_LOCK:
            # 移动到进度区域顶部、插入日志、再重绘
            if self._rendered_lines:
                sys.stdout.write(f"\x1b[{self._rendered_lines}A")
            sys.stdout.write("\x1b[2K\r" + line + "\n")
            self._rendered_lines = 0
            self._render_locked()

    def _render(self) -> None:
        with _PRINT_LOCK:
            self._render_locked()

    def _render_locked(self) -> None:
        if not self._dynamic:
            return
        # 回到当前进度区域顶部
        if self._rendered_lines:
            sys.stdout.write(f"\x1b[{self._rendered_lines}A")
        # 逐行清空并写入
        for k in self._order:
            sys.stdout.write("\x1b[2K\r" + self._lines.get(k, "") + "\n")
        self._rendered_lines = len(self._order)
        sys.stdout.flush()


def _shorten(s: str, *, max_len: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    if max_len <= 1:
        return s[:max_len]
    return s[: max_len - 1] + "…"


def _safe_name(s: str, *, fallback: str, max_len: int | None = None) -> str:
    s = (s or "").strip()
    if not s:
        s = fallback
    s = _INVALID_FS_CHARS.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    lim = int(max_len) if max_len is not None else int(_SAFE_NAME_MAX)
    return s[:lim] if len(s) > lim else s


def _clean_title(raw: str) -> str:
    """
    清理站点/父级板块后缀，例如：
    - "... - 中文AV - 小黄书 xChina" -> "... - <子分类>"
    """

    s = (raw or "").strip()
    if not s:
        return s
    parts = [p.strip() for p in s.split(" - ") if p.strip()]
    if len(parts) <= 1:
        return s

    def should_strip(token: str) -> bool:
        t = token.strip()
        if not t:
            return True
        tl = t.lower()
        if tl == "中文av":
            return True
        if _RE_TITLE_STRIP_TOKEN.search(t):
            return True
        return False

    while parts and should_strip(parts[-1]):
        parts.pop()

    return " - ".join(parts) if parts else s


def _strip_trailing_title_tokens(title: str, *, tokens: list[str]) -> str:
    """
    从标题末尾剥离一组已知“后缀 token”（通常来自板块/子分类名称）。

    仅在 token 恰好作为以 " - " 分隔的最后若干段时才会剥离，避免误伤标题主体。
    """

    s = (title or "").strip()
    if not s:
        return s

    parts = [p.strip() for p in s.split(" - ") if p.strip()]
    if not parts:
        return s

    token_set = {t.strip().lower() for t in (tokens or []) if isinstance(t, str) and t.strip()}
    if not token_set:
        return s

    while parts and parts[-1].lower() in token_set:
        parts.pop()

    return " - ".join(parts) if parts else s


def _ext_from_url(url: str) -> str | None:
    try:
        p = urlparse(url).path
    except Exception:
        return None
    _, ext = os.path.splitext(p)
    ext = ext.lower().strip()
    if not ext or len(ext) > 6:
        return None
    return ext


def _download_file(url: str, *, out_path: Path, headers: dict[str, str], timeout_seconds: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with requests.get(url, headers=headers, stream=True, timeout=timeout_seconds) as resp:
        resp.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                f.write(chunk)
    tmp_path.replace(out_path)


def _copy_file(src: Path, dst: Path, *, force: bool) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if force or not dst.exists():
        shutil.copy2(src, dst)


def _format_hhmmss(seconds: float | int) -> str:
    try:
        s = max(0.0, float(seconds))
    except Exception:
        return "??:??:??"
    hh = int(s // 3600)
    mm = int((s % 3600) // 60)
    ss = int(s % 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _estimate_m3u8_duration_seconds(
    url: str, *, headers: dict[str, str], timeout_seconds: int, max_bytes: int = 1_000_000
) -> int | None:
    """
    从 m3u8 清单估算总时长（秒）。

    说明：
    - 若是 media playlist：累加 #EXTINF
    - 若是 master playlist：选 BANDWIDTH 最大的变体再累加
    """

    def fetch_text(u: str) -> str:
        with requests.get(u, headers=headers, stream=True, timeout=timeout_seconds) as resp:
            resp.raise_for_status()
            chunks: list[bytes] = []
            size = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                chunks.append(chunk)
                size += len(chunk)
                if size >= max_bytes:
                    break
        return b"".join(chunks).decode("utf-8", errors="replace")

    def sum_extinf(text: str) -> int | None:
        vals = re.findall(r"#EXTINF:([0-9]+(?:\.[0-9]+)?)", text)
        if not vals:
            return None
        total = 0.0
        for v in vals:
            try:
                total += float(v)
            except ValueError:
                continue
        if total <= 0:
            return None
        return int(round(total))

    try:
        first = fetch_text(url)
    except Exception:
        return None

    d = sum_extinf(first)
    if d is not None:
        return d

    # master playlist: pick max bandwidth variant
    if "#EXT-X-STREAM-INF" not in first:
        return None

    best_bw = -1
    best_uri: str | None = None
    lines = [ln.strip() for ln in first.splitlines()]
    for i, ln in enumerate(lines):
        if not ln.startswith("#EXT-X-STREAM-INF:"):
            continue
        m = re.search(r"BANDWIDTH=(\d+)", ln)
        bw = int(m.group(1)) if m else 0
        uri = None
        for j in range(i + 1, min(i + 6, len(lines))):
            nxt = lines[j]
            if not nxt or nxt.startswith("#"):
                continue
            uri = nxt
            break
        if uri and bw >= best_bw:
            best_bw = bw
            best_uri = uri

    if not best_uri:
        return None

    try:
        second_url = urljoin(url, best_uri)
        second = fetch_text(second_url)
    except Exception:
        return None
    return sum_extinf(second)


def _select_best_variant_from_master(master_text: str) -> str | None:
    best_bw = -1
    best_uri: str | None = None
    lines = [ln.strip() for ln in master_text.splitlines()]
    for i, ln in enumerate(lines):
        if not ln.startswith("#EXT-X-STREAM-INF:"):
            continue
        m = re.search(r"BANDWIDTH=(\d+)", ln)
        bw = int(m.group(1)) if m else 0
        uri = None
        for j in range(i + 1, min(i + 6, len(lines))):
            nxt = lines[j]
            if not nxt or nxt.startswith("#"):
                continue
            uri = nxt
            break
        if uri and bw >= best_bw:
            best_bw = bw
            best_uri = uri
    return best_uri


@dataclass(frozen=True)
class MediaPlaylist:
    base_url: str
    text: str
    segment_urls: list[str]
    segment_durations: list[float]
    map_url: str | None
    key_url: str | None
    key_iv: str | None


def _fetch_text(url: str, *, headers: dict[str, str], timeout_seconds: int) -> str:
    with requests.get(url, headers=headers, timeout=timeout_seconds) as resp:
        resp.raise_for_status()
        return resp.text


def _parse_media_playlist(url: str, text: str) -> MediaPlaylist:
    base_url = url
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    seg_urls: list[str] = []
    seg_durs: list[float] = []
    map_url: str | None = None
    key_url: str | None = None
    key_iv: str | None = None

    pending_dur: float | None = None
    for ln in lines:
        if ln.startswith("#EXT-X-MAP:"):
            m = re.search(r'URI="([^"]+)"', ln)
            if m:
                map_url = urljoin(base_url, m.group(1))
        elif ln.startswith("#EXT-X-KEY:"):
            if "METHOD=AES-128" in ln:
                m = re.search(r'URI="([^"]+)"', ln)
                if m:
                    key_url = urljoin(base_url, m.group(1))
                m_iv = re.search(r"IV=([^,]+)", ln)
                if m_iv:
                    key_iv = m_iv.group(1).strip()
        elif ln.startswith("#EXTINF:"):
            m = re.match(r"#EXTINF:([0-9]+(?:\.[0-9]+)?)", ln)
            if m:
                try:
                    pending_dur = float(m.group(1))
                except ValueError:
                    pending_dur = None
        elif ln.startswith("#"):
            continue
        else:
            seg_urls.append(urljoin(base_url, ln))
            seg_durs.append(float(pending_dur or 0.0))
            pending_dur = None

    return MediaPlaylist(
        base_url=base_url,
        text=text,
        segment_urls=seg_urls,
        segment_durations=seg_durs,
        map_url=map_url,
        key_url=key_url,
        key_iv=key_iv,
    )


def _load_media_playlist(
    url: str, *, headers: dict[str, str], timeout_seconds: int
) -> MediaPlaylist:
    text = _fetch_text(url, headers=headers, timeout_seconds=timeout_seconds)
    if "#EXT-X-STREAM-INF" in text and "#EXTINF" not in text:
        best = _select_best_variant_from_master(text)
        if not best:
            raise RuntimeError("m3u8 主播放列表未包含可用清晰度（variants）")
        url2 = urljoin(url, best)
        text2 = _fetch_text(url2, headers=headers, timeout_seconds=timeout_seconds)
        return _parse_media_playlist(url2, text2)
    return _parse_media_playlist(url, text)


def _write_local_m3u8(
    *,
    out_path: Path,
    original_text: str,
    segments: list[str],
    segment_present: list[bool],
    map_name: str | None,
    key_name: str | None,
) -> None:
    """
    将远端 media playlist 重写为本地 playlist：
    - 保留原始 EXTINF/ENDLIST 等结构
    - KEY/MAP URI 指向本地文件名
    - segment URI 依序替换为本地文件名
    """

    seg_idx = 0
    out_lines: list[str] = []
    pending_segment_meta: list[str] = []
    for raw in original_text.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if ln.startswith("#EXT-X-KEY:") and key_name:
            out_lines.append(re.sub(r'URI="[^"]+"', f'URI="{key_name}"', ln))
            continue
        if ln.startswith("#EXT-X-MAP:") and map_name:
            out_lines.append(re.sub(r'URI="[^"]+"', f'URI="{map_name}"', ln))
            continue
        if ln.startswith("#"):
            # 与 segment 相关的元信息：先暂存，遇到 segment 决定是否写入（用于跳过缺失分片）
            if ln.startswith("#EXTINF:") or ln.startswith("#EXT-X-BYTERANGE:") or ln == "#EXT-X-DISCONTINUITY":
                pending_segment_meta.append(ln)
            else:
                out_lines.append(ln)
            continue
        # segment uri
        if seg_idx >= len(segments):
            raise RuntimeError("重写 m3u8 时分片数量不一致")
        present = True
        if seg_idx < len(segment_present):
            present = bool(segment_present[seg_idx])
        if present:
            out_lines.extend(pending_segment_meta)
            out_lines.append(segments[seg_idx])
        pending_segment_meta = []
        seg_idx += 1

    if seg_idx != len(segments):
        raise RuntimeError("重写 m3u8 时分片数量不一致")
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _run_aria2(
    *,
    aria2c_path: str,
    input_path: Path,
    dir_path: Path,
    concurrent_segments: int,
    headers: dict[str, str],
    show_progress: bool,
    renderer: ProgressRenderer,
    renderer_key: str,
    display_name: str,
    segment_names: list[str],
    segment_durations: list[float],
    total_duration_seconds: int | None,
    progress_interval_seconds: float,
    log_path: Path,
) -> None:
    if shutil.which(aria2c_path) is None:
        raise RuntimeError(f"未找到 aria2c：{aria2c_path!r}")

    dir_path.mkdir(parents=True, exist_ok=True)
    cmd = [
        aria2c_path,
        "-c",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--file-allocation=none",
        "--max-tries=10",
        "--retry-wait=1",
        "-j",
        str(int(concurrent_segments)),
        "-x",
        str(int(concurrent_segments)),
        "-s",
        str(int(concurrent_segments)),
        "--summary-interval=0",
        "--log-level=warn",
        "-l",
        str(log_path),
        "-d",
        str(dir_path),
        "-i",
        str(input_path),
    ]
    for hk, hv in headers.items():
        cmd += ["--header", f"{hk}: {hv}"]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    tail = deque(maxlen=80)
    tail_lock = threading.Lock()

    def pump() -> None:
        for raw in proc.stdout:  # blocks in this thread only
            s = raw.rstrip("\r\n")
            if not s:
                continue
            with tail_lock:
                tail.append(s)

    t = threading.Thread(target=pump, name=f"aria2-log-{renderer_key}", daemon=True)
    t.start()

    expected = set(segment_names)
    dur_map = {segment_names[i]: float(segment_durations[i] or 0.0) for i in range(min(len(segment_names), len(segment_durations)))}

    last_print = 0.0
    last_bytes = 0
    last_t = time.time()
    seg_total = len(segment_names)
    while True:
        rc = proc.poll()
        now = time.time()
        if show_progress and (now - last_print) >= max(0.2, float(progress_interval_seconds)):
            last_print = now
            done = 0
            done_dur = 0.0
            bytes_now = 0
            try:
                in_progress: set[str] = set()
                seg_sizes: dict[str, int] = {}
                for ent in os.scandir(dir_path):
                    if not ent.is_file():
                        continue
                    name = ent.name
                    try:
                        st = ent.stat()
                    except FileNotFoundError:
                        continue
                    if name.endswith(".aria2"):
                        base = name[: -len(".aria2")]
                        if base:
                            in_progress.add(base)
                        continue
                    if name in expected and st.st_size > 0:
                        seg_sizes[name] = int(st.st_size)

                if seg_sizes:
                    bytes_now = sum(seg_sizes.values())
                    for name in seg_sizes.keys():
                        if name in in_progress:
                            continue
                        done += 1
                        done_dur += dur_map.get(name, 0.0)
            except FileNotFoundError:
                pass

            dt = max(1e-6, now - last_t)
            net = (bytes_now - last_bytes) / (1024.0 * 1024.0) / dt
            last_bytes = bytes_now
            last_t = now
            if total_duration_seconds and total_duration_seconds > 0:
                pct = min(99.9, (done_dur / float(total_duration_seconds)) * 100.0)
                renderer.update(
                    renderer_key,
                    f"[{display_name}] {pct:5.1f}% {done:4d}/{seg_total} 分片 {_format_hhmmss(done_dur)}/{_format_hhmmss(total_duration_seconds)} 网速={net:4.1f}MiB/s",
                )
            else:
                renderer.update(
                    renderer_key,
                    f"[{display_name}] {done:4d}/{seg_total} 分片 {_format_hhmmss(done_dur)} 网速={net:4.1f}MiB/s",
                )

        if rc is not None:
            break
        time.sleep(0.05)

    if rc != 0:
        with tail_lock:
            tail_s = "\n".join([x for x in tail if x][-25:])
        log_tail = ""
        try:
            if log_path.exists():
                t = log_path.read_text(encoding="utf-8", errors="replace")
                log_tail = t[-4000:]
        except Exception:
            log_tail = ""
        extra = f"\n--- aria2.log tail ---\n{log_tail}" if log_tail else ""
        raise RuntimeError(f"aria2c 失败（rc={rc}）。最近输出：\n{tail_s}{extra}")


def _download_m3u8_with_aria2(
    *,
    m3u8_url: str,
    out_path: Path,
    tmp_dir: Path,
    aria2c_path: str,
    concurrent_segments: int,
    headers: dict[str, str],
    show_progress: bool,
    renderer: ProgressRenderer,
    renderer_key: str,
    display_name: str,
    timeout_seconds: int,
    progress_interval_seconds: float,
    total_duration_seconds_hint: int | None,
    max_missing_segments: int,
) -> None:
    playlist = _load_media_playlist(m3u8_url, headers=headers, timeout_seconds=timeout_seconds)

    segments_dir = (tmp_dir / "segments").resolve()
    segments_dir.mkdir(parents=True, exist_ok=True)

    downloads: list[tuple[str, str]] = []  # (url, out_name)
    local_segment_names: list[str] = []

    if playlist.map_url:
        downloads.append((playlist.map_url, "init.mp4"))
    if playlist.key_url:
        downloads.append((playlist.key_url, "key.bin"))

    for idx, u in enumerate(playlist.segment_urls, start=1):
        name = f"{idx:06d}.ts"
        downloads.append((u, name))
        local_segment_names.append(name)

    input_txt = tmp_dir / "aria2.txt"
    lines: list[str] = []
    for u, name in downloads:
        lines.append(u)
        lines.append(f"  out={name}")
    input_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # aria2 下载（单视频分片并发）
    total_dur = total_duration_seconds_hint
    if not total_dur and playlist.segment_durations:
        s = sum([d for d in playlist.segment_durations if d > 0.0])
        if s > 0:
            total_dur = int(round(s))

    seg_names_for_progress = [n for _, n in downloads if n.endswith(".ts")]
    seg_durs = playlist.segment_durations
    aria2_log = tmp_dir / "aria2.log"
    aria2_error: Exception | None = None
    try:
        _run_aria2(
            aria2c_path=aria2c_path,
            input_path=input_txt,
            dir_path=segments_dir,
            concurrent_segments=concurrent_segments,
            headers=headers,
            show_progress=show_progress,
            renderer=renderer,
            renderer_key=renderer_key,
            display_name=display_name,
            segment_names=seg_names_for_progress,
            segment_durations=seg_durs,
            total_duration_seconds=total_dur,
            progress_interval_seconds=progress_interval_seconds,
            log_path=aria2_log,
        )
    except Exception as exc:  # noqa: BLE001
        aria2_error = exc

    # 校验关键文件存在（init/key 必须存在；ts 允许少量缺失）
    required: list[str] = []
    if playlist.map_url:
        required.append("init.mp4")
    if playlist.key_url:
        required.append("key.bin")
    for name in required:
        p = segments_dir / name
        if not p.exists() or p.stat().st_size <= 0:
            raise RuntimeError(f"缺少必需文件：{name}") from aria2_error

    # 统计缺失分片（缺片会导致画面/音频跳跃；可配置容错上限）
    missing_ts: list[str] = []
    present_mask: list[bool] = []
    for name in local_segment_names:
        p = segments_dir / name
        part = segments_dir / (name + ".aria2")
        ok = p.exists()
        if ok:
            try:
                ok = p.stat().st_size > 0
            except FileNotFoundError:
                ok = False
        if part.exists():
            ok = False
        present_mask.append(bool(ok))
        if not ok:
            missing_ts.append(name)

    if missing_ts:
        if len(missing_ts) > int(max_missing_segments):
            sample = ", ".join(missing_ts[:12])
            raise RuntimeError(
                f"缺失 ts 分片：{len(missing_ts)} > max_missing_segments={max_missing_segments}；sample={sample}"
            ) from aria2_error
        renderer.log(f"[警告] 允许缺失 ts 分片：{len(missing_ts)}（max={max_missing_segments}）")

    # aria2 即便所有文件都下载完成，也可能因为个别请求报错导致 rc!=0；这种情况按“文件齐全”继续。
    if aria2_error is not None:
        renderer.log(f"[警告] aria2 退出时有错误，但文件齐全，继续。err={aria2_error}")

    # 额外校验：AES-128 key 通常为 16 bytes；若明显不对，给出更清晰的错误。
    if playlist.key_url:
        key_p = segments_dir / "key.bin"
        try:
            key_size = key_p.stat().st_size
        except FileNotFoundError as exc:
            raise RuntimeError("缺少 key.bin") from exc
        if key_size != 16:
            sample = b""
            try:
                sample = key_p.open("rb").read(64)
            except Exception:
                sample = b""
            raise RuntimeError(f"key.bin 大小异常：size={key_size} sample={sample[:32]!r}")

    # 生成本地 m3u8（供 ffmpeg 合并/解密）
    local_m3u8 = segments_dir / "local.m3u8"
    _write_local_m3u8(
        out_path=local_m3u8,
        original_text=playlist.text,
        segments=local_segment_names,
        segment_present=present_mask,
        map_name=("init.mp4" if playlist.map_url else None),
        key_name=("key.bin" if playlist.key_url else None),
    )

    # ffmpeg 读取本地 m3u8 输出 mp4（无网络请求）
    out_abs = out_path.resolve()
    out_abs.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_abs.with_name(out_abs.stem + ".part" + out_abs.suffix)
    if tmp_out.exists():
        tmp_out.unlink()
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        # hls demuxer 会限制本地引用文件扩展名（例如 key.bin 会被拦截）；这里放开以支持本地 key 文件。
        "-allowed_extensions",
        "ALL",
        "-protocol_whitelist",
        "file,crypto,data",
        "-i",
        str(local_m3u8),
        "-f",
        "mp4",
        "-c",
        "copy",
        "-bsf:a",
        "aac_adtstoasc",
        str(tmp_out),
    ]
    subprocess.run(cmd, check=True, cwd=str(segments_dir))
    tmp_out.replace(out_abs)

def _ffmpeg_download_m3u8(
    *,
    m3u8_url: str,
    out_path: Path,
    user_agent: str,
    referer: str,
    duration_seconds: int | None,
    show_progress: bool,
    progress_prefix: str,
    progress_interval_seconds: float,
    on_progress,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ffmpeg 通过扩展名推断输出格式；Windows 下用 `.mp4.part` 会导致无法识别。
    tmp_path = out_path.with_name(out_path.stem + ".part" + out_path.suffix)
    if tmp_path.exists():
        tmp_path.unlink()

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-user_agent",
        user_agent,
        "-headers",
        f"Referer: {referer}\r\n",
        "-i",
        m3u8_url,
        "-f",
        "mp4",
        "-c",
        "copy",
        "-bsf:a",
        "aac_adtstoasc",
        str(tmp_path),
    ]

    if not show_progress:
        subprocess.run(cmd, check=True)
        tmp_path.replace(out_path)
        return

    # 进度：用 -progress pipe:1 输出 key=value（避免解析 stderr 的 “frame=…”）
    cmd_p = cmd[:-1] + ["-progress", "pipe:1", "-nostats", cmd[-1]]
    proc = subprocess.Popen(
        cmd_p,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None

    last_print = 0.0
    out_time_ms = 0
    total_size = 0
    speed = ""
    last_speed_t = 0.0
    last_speed_size = 0
    tail = deque(maxlen=60)
    for raw in proc.stdout:
        tail.append(raw.rstrip("\r\n"))
        line = raw.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k == "out_time_ms":
            try:
                out_time_ms = int(v)
            except ValueError:
                continue
        elif k == "total_size":
            try:
                total_size = int(v)
            except ValueError:
                continue
        elif k == "speed":
            speed = v.strip()
        elif k == "progress" and v == "end":
            break

        now = time.time()
        if now - last_print < max(0.2, float(progress_interval_seconds)):
            continue
        last_print = now

        out_sec = out_time_ms / 1_000_000.0
        size_mb = total_size / (1024.0 * 1024.0) if total_size else 0.0
        net_mib_s = None
        if total_size and last_speed_t > 0 and now > last_speed_t:
            delta = total_size - last_speed_size
            dt = now - last_speed_t
            if delta >= 0 and dt > 0.0:
                net_mib_s = (delta / (1024.0 * 1024.0)) / dt
        last_speed_t = now
        last_speed_size = total_size

        net_s = f"{net_mib_s:4.1f}MiB/s" if net_mib_s is not None else "?MiB/s"
        if duration_seconds and duration_seconds > 0:
            pct = min(99.9, out_sec / float(duration_seconds) * 100.0)
            msg = (
                f"[{progress_prefix}] {pct:5.1f}% "
                f"{_format_hhmmss(out_sec)}/{_format_hhmmss(duration_seconds)} "
                f"{size_mb:8.1f}MB 网速={net_s}"
            )
        else:
            msg = f"[{progress_prefix}] {_format_hhmmss(out_sec)} {size_mb:8.1f}MB 网速={net_s}"
        on_progress(msg)

    rc = proc.wait()
    if rc != 0:
        tail_s = "\n".join([x for x in tail if x][-20:])
        raise RuntimeError(f"ffmpeg 失败（rc={rc}）。最近输出：\n{tail_s}")

    tmp_path.replace(out_path)


def _extract_authors(jsonld: dict[str, Any] | None) -> list[str]:
    if not jsonld:
        return []

    val = jsonld.get("author") or jsonld.get("creator")
    items: list[Any]
    if isinstance(val, list):
        items = val
    elif val is None:
        return []
    else:
        items = [val]

    out: list[str] = []
    for it in items:
        name: str | None = None
        if isinstance(it, str):
            name = it.strip()
        elif isinstance(it, dict):
            n = it.get("name")
            if isinstance(n, str):
                name = n.strip()
        if name and name not in out:
            out.append(name)
    return out


def _extract_actors(jsonld: dict[str, Any] | None) -> list[str]:
    if not jsonld:
        return []

    val = jsonld.get("actor")
    items: list[Any]
    if isinstance(val, list):
        items = val
    elif val is None:
        return []
    else:
        items = [val]

    out: list[str] = []
    for it in items:
        name: str | None = None
        if isinstance(it, str):
            name = it.strip()
        elif isinstance(it, dict):
            n = it.get("name")
            if isinstance(n, str):
                name = n.strip()
        if name and name not in out:
            out.append(name)
    return out


def _extract_description(jsonld: dict[str, Any] | None) -> str | None:
    if not jsonld:
        return None
    d = jsonld.get("description")
    if isinstance(d, str):
        s = d.strip()
        return s or None
    return None


def _parse_fraction(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            num = float(a)
            den = float(b)
        except ValueError:
            return None
        if den == 0:
            return None
        return num / den
    try:
        return float(s)
    except ValueError:
        return None


def _aspect_ratio(width: int | None, height: int | None) -> str | None:
    if not width or not height or width <= 0 or height <= 0:
        return None
    import math

    g = math.gcd(int(width), int(height))
    w = int(width // g)
    h = int(height // g)
    return f"{w}:{h}"


def _ffprobe_streamdetails(path: Path) -> dict[str, Any] | None:
    """
    解析视频文件的 streamdetails，用于写入 movie.nfo 的 <fileinfo>。
    """

    if shutil.which("ffprobe") is None:
        return None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    except Exception:
        return None

    try:
        data = json.loads(res.stdout)
    except Exception:
        return None

    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    if not isinstance(streams, list):
        streams = []

    vstream = next((s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"), None)
    astream = next((s for s in streams if isinstance(s, dict) and s.get("codec_type") == "audio"), None)

    out: dict[str, Any] = {"video": None, "audio": None}

    duration_s: int | None = None
    dur_raw = None
    if isinstance(fmt, dict):
        dur_raw = fmt.get("duration")
    if isinstance(dur_raw, str):
        d = _parse_fraction(dur_raw)
        if d and d > 0:
            duration_s = int(round(d))

    if isinstance(vstream, dict):
        width = vstream.get("width")
        height = vstream.get("height")
        try:
            width_i = int(width) if width is not None else None
        except Exception:
            width_i = None
        try:
            height_i = int(height) if height is not None else None
        except Exception:
            height_i = None

        dar = None
        if isinstance(vstream.get("display_aspect_ratio"), str):
            dar = vstream.get("display_aspect_ratio")
        if not dar:
            dar = _aspect_ratio(width_i, height_i)

        fr = None
        for k in ("avg_frame_rate", "r_frame_rate"):
            if isinstance(vstream.get(k), str):
                fr = _parse_fraction(vstream.get(k))
                if fr:
                    break

        vbit = vstream.get("bit_rate")
        try:
            vbit_i = int(vbit) if vbit is not None else None
        except Exception:
            vbit_i = None

        lang = "und"
        tags = vstream.get("tags") if isinstance(vstream.get("tags"), dict) else {}
        if isinstance(tags, dict) and isinstance(tags.get("language"), str) and tags.get("language").strip():
            lang = tags.get("language").strip()

        disp = vstream.get("disposition") if isinstance(vstream.get("disposition"), dict) else {}
        default = bool(disp.get("default")) if isinstance(disp, dict) else False
        forced = bool(disp.get("forced")) if isinstance(disp, dict) else False

        out["video"] = {
            "codec": vstream.get("codec_name") or None,
            "bitrate": vbit_i,
            "width": width_i,
            "height": height_i,
            "aspect": dar,
            "framerate": (round(fr, 3) if fr else None),
            "language": lang,
            "scantype": "progressive",
            "default": default,
            "forced": forced,
            "duration_minutes": (int(duration_s // 60) if duration_s else None),
            "duration_seconds": duration_s,
        }

    if isinstance(astream, dict):
        abit = astream.get("bit_rate")
        try:
            abit_i = int(abit) if abit is not None else None
        except Exception:
            abit_i = None

        lang = "und"
        tags = astream.get("tags") if isinstance(astream.get("tags"), dict) else {}
        if isinstance(tags, dict) and isinstance(tags.get("language"), str) and tags.get("language").strip():
            lang = tags.get("language").strip()

        disp = astream.get("disposition") if isinstance(astream.get("disposition"), dict) else {}
        default = bool(disp.get("default")) if isinstance(disp, dict) else False
        forced = bool(disp.get("forced")) if isinstance(disp, dict) else False

        ch = astream.get("channels")
        try:
            ch_i = int(ch) if ch is not None else None
        except Exception:
            ch_i = None

        sr = astream.get("sample_rate")
        try:
            sr_i = int(sr) if sr is not None else None
        except Exception:
            sr_i = None

        out["audio"] = {
            "codec": astream.get("codec_name") or None,
            "bitrate": abit_i,
            "language": lang,
            "channels": ch_i,
            "samplingrate": sr_i,
            "default": default,
            "forced": forced,
        }

    if out["video"] is None and out["audio"] is None:
        return None
    return out


def _pick_series_from_breadcrumbs(breadcrumbs: list[Any]) -> tuple[str, str] | None:
    """
    从 /video JSON-LD BreadcrumbList 中选取“最具体”的 series（通常是最后一个 series 链接）。

    Returns:
        (series_id, series_name) 或 None
    """

    matches: list[tuple[str, str]] = []
    for bc in breadcrumbs:
        item = getattr(bc, "item", "") or ""
        m = _RE_BREADCRUMB_SERIES.search(item)
        if not m:
            continue
        sid = m.group(1)
        name = (getattr(bc, "name", "") or "").strip() or f"series-{sid}"
        matches.append((sid, name))
    return matches[-1] if matches else None


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _write_movie_nfo(
    *,
    out_path: Path,
    video_id: str,
    title: str,
    premiered: str | None,
    year: int | None,
    runtime_minutes: int | None,
    authors: list[str],
    actors: list[str],
    tags: list[str],
    series_name: str | None,
    plot: str | None,
    streamdetails: dict[str, Any] | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', "<movie>"]
    lines.append(f"  <title>{_xml_escape(title)}</title>")
    lines.append(f"  <originaltitle>{_xml_escape(title)}</originaltitle>")
    lines.append(f'  <uniqueid type="xchina" default="true">{_xml_escape(video_id)}</uniqueid>')
    if premiered:
        lines.append(f"  <premiered>{_xml_escape(premiered)}</premiered>")
    if year:
        lines.append(f"  <year>{year}</year>")
    if runtime_minutes:
        lines.append(f"  <runtime>{runtime_minutes}</runtime>")
    for a in authors:
        lines.append(f"  <studio>{_xml_escape(a)}</studio>")
    for name in actors:
        lines.append("  <actor>")
        lines.append(f"    <name>{_xml_escape(name)}</name>")
        lines.append("  </actor>")
    if series_name:
        lines.append(f"  <genre>{_xml_escape(series_name)}</genre>")
    seen: set[str] = set()
    for t in tags:
        t = (t or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        lines.append(f"  <tag>{_xml_escape(t)}</tag>")
    if plot:
        lines.append(f"  <plot>{_xml_escape(plot)}</plot>")

    if streamdetails:
        lines.append("  <fileinfo>")
        lines.append("    <streamdetails>")
        v = streamdetails.get("video") if isinstance(streamdetails, dict) else None
        if isinstance(v, dict):
            lines.append("      <video>")
            if v.get("codec"):
                lines.append(f"        <codec>{_xml_escape(str(v.get('codec')))}</codec>")
                lines.append(f"        <micodec>{_xml_escape(str(v.get('codec')))}</micodec>")
            if v.get("bitrate") is not None:
                lines.append(f"        <bitrate>{int(v.get('bitrate'))}</bitrate>")
            if v.get("width") is not None:
                lines.append(f"        <width>{int(v.get('width'))}</width>")
            if v.get("height") is not None:
                lines.append(f"        <height>{int(v.get('height'))}</height>")
            if v.get("aspect"):
                ar = str(v.get("aspect"))
                lines.append(f"        <aspect>{_xml_escape(ar)}</aspect>")
                lines.append(f"        <aspectratio>{_xml_escape(ar)}</aspectratio>")
            if v.get("framerate") is not None:
                lines.append(f"        <framerate>{v.get('framerate')}</framerate>")
            if v.get("language"):
                lines.append(f"        <language>{_xml_escape(str(v.get('language')))}</language>")
            if v.get("scantype"):
                lines.append(f"        <scantype>{_xml_escape(str(v.get('scantype')))}</scantype>")
            lines.append(f"        <default>{'True' if v.get('default') else 'False'}</default>")
            lines.append(f"        <forced>{'True' if v.get('forced') else 'False'}</forced>")
            if v.get("duration_minutes") is not None:
                lines.append(f"        <duration>{int(v.get('duration_minutes'))}</duration>")
            if v.get("duration_seconds") is not None:
                lines.append(f"        <durationinseconds>{int(v.get('duration_seconds'))}</durationinseconds>")
            lines.append("      </video>")
        a = streamdetails.get("audio") if isinstance(streamdetails, dict) else None
        if isinstance(a, dict):
            lines.append("      <audio>")
            if a.get("codec"):
                lines.append(f"        <codec>{_xml_escape(str(a.get('codec')))}</codec>")
                lines.append(f"        <micodec>{_xml_escape(str(a.get('codec')))}</micodec>")
            if a.get("bitrate") is not None:
                lines.append(f"        <bitrate>{int(a.get('bitrate'))}</bitrate>")
            if a.get("language"):
                lines.append(f"        <language>{_xml_escape(str(a.get('language')))}</language>")
            lines.append(f"        <scantype>progressive</scantype>")
            if a.get("channels") is not None:
                lines.append(f"        <channels>{int(a.get('channels'))}</channels>")
            if a.get("samplingrate") is not None:
                lines.append(f"        <samplingrate>{int(a.get('samplingrate'))}</samplingrate>")
            lines.append(f"        <default>{'True' if a.get('default') else 'False'}</default>")
            lines.append(f"        <forced>{'True' if a.get('forced') else 'False'}</forced>")
            lines.append("      </audio>")
        lines.append("    </streamdetails>")
        lines.append("  </fileinfo>")
    lines.append("</movie>")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class VideoRow:
    video_id: str
    title: str | None
    h1: str | None
    page_url: str | None
    canonical_url: str | None
    cover_url: str | None
    screenshot_url: str | None
    screenshot_urls: list[str] | None
    upload_date: datetime | None
    duration_seconds: int | None
    m3u8_url: str | None
    video_series_name: str | None
    video_series_source_key: str | None
    video_tags: list[str] | None
    jsonld: dict[str, Any] | None
    downloaded_at: datetime | None
    download_status: str | None


def _load_videos(
    db: Db,
    *,
    video_id: str | None,
    series_id: str | None,
    limit: int,
    include_downloaded: bool,
) -> list[VideoRow]:
    where = []
    params: list[Any] = []
    if video_id:
        where.append("video_id = %s")
        params.append(video_id)
    if series_id:
        where.append("video_series_source_key = %s")
        params.append(series_id)
    if not include_downloaded:
        where.append("downloaded_at is null")

    where_sql = ("where " + " and ".join(where)) if where else ""
    sql = f"""
select
  video_id, title, h1, page_url, canonical_url,
  cover_url, screenshot_url, screenshot_urls, upload_date, duration_seconds,
  m3u8_url, video_series_name, video_series_source_key, video_tags,
  jsonld, downloaded_at, download_status
from videos
{where_sql}
order by coalesce(upload_date, '1970-01-01'::timestamptz) desc, video_id desc
limit %s
    """.strip()
    params.append(int(limit))

    out: list[VideoRow] = []
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            for row in cur.fetchall():
                (
                    vid,
                    title,
                    h1,
                    page_url,
                    canonical_url,
                    cover_url,
                    screenshot_url,
                    screenshot_urls,
                    upload_date,
                    duration_seconds,
                    m3u8_url,
                    series_name,
                    series_key,
                    video_tags,
                    jsonld,
                    downloaded_at,
                    download_status,
                ) = row
                tags_list = None
                if isinstance(video_tags, str):
                    try:
                        v = json.loads(video_tags)
                        if isinstance(v, list):
                            tags_list = [str(x) for x in v if str(x).strip()]
                    except Exception:
                        tags_list = None
                elif isinstance(video_tags, list):
                    tags_list = [str(x) for x in video_tags if str(x).strip()]

                jsonld_dict = None
                if isinstance(jsonld, str):
                    try:
                        v = json.loads(jsonld)
                        if isinstance(v, dict):
                            jsonld_dict = v
                    except Exception:
                        jsonld_dict = None
                elif isinstance(jsonld, dict):
                    jsonld_dict = jsonld

                screenshot_urls_list = None
                if isinstance(screenshot_urls, str):
                    try:
                        v = json.loads(screenshot_urls)
                        if isinstance(v, list):
                            screenshot_urls_list = [str(x) for x in v if str(x).strip()]
                    except Exception:
                        screenshot_urls_list = None
                elif isinstance(screenshot_urls, list):
                    screenshot_urls_list = [str(x) for x in screenshot_urls if str(x).strip()]

                out.append(
                    VideoRow(
                        video_id=str(vid),
                        title=title,
                        h1=h1,
                        page_url=page_url,
                        canonical_url=canonical_url,
                        cover_url=cover_url,
                        screenshot_url=screenshot_url,
                        screenshot_urls=screenshot_urls_list,
                        upload_date=upload_date,
                        duration_seconds=duration_seconds,
                        m3u8_url=m3u8_url,
                        video_series_name=series_name,
                        video_series_source_key=series_key,
                        video_tags=tags_list,
                        jsonld=jsonld_dict,
                        downloaded_at=downloaded_at,
                        download_status=download_status,
                    )
                )
    return out


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="xchina_emby_downloader")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", help="输出根目录（默认会在其下创建 _working/ 与 complete/；可在配置 download.out_dir 设置）")
    ap.add_argument("--limit", type=int, help="最多处理多少条（可在配置 download.limit 设置）")
    ap.add_argument("--video-id")
    ap.add_argument("--series-id", help="按 videos.video_series_source_key 过滤")
    ap.add_argument("--include-downloaded", action="store_true", default=None, help="也处理已标记 downloaded 的记录")
    ap.add_argument("--no-refresh", action="store_true", default=None, help="不刷新 /video 页面（m3u8 可能过期）")
    ap.add_argument("--force", action="store_true", help="强制重新下载（忽略已有文件/DB 标记）")
    ap.add_argument("--no-move", action="store_true", default=None, help="不移动到 complete（直接落在 --out 下）")
    ap.add_argument("--work-subdir", help="工作目录子目录名（可在配置 download.work_subdir 设置）")
    ap.add_argument("--complete-subdir", help="完成目录子目录名（可在配置 download.complete_subdir 设置）")
    ap.add_argument("--workers", type=int, help="下载并发数（按视频维度；可在配置 download.workers 设置）")
    ap.add_argument("--no-progress", action="store_true", default=None, help="不显示下载进度（可在配置 download.show_progress 设置）")
    ap.add_argument("--no-dynamic-progress", action="store_true", default=None, help="禁用同屏动态刷新进度（可在配置 download.dynamic_progress 设置）")
    ap.add_argument("--engine", choices=["ffmpeg", "aria2"], help="下载引擎（可在配置 download.engine 设置）")
    ap.add_argument("--concurrent-segments", type=int, help="aria2 单视频分片并发数（可在配置 download.concurrent_segments 设置）")
    ap.add_argument("--max-missing-segments", type=int, help="aria2 模式允许缺失的 ts 分片数量（可在配置 download.max_missing_segments 设置）")
    args = ap.parse_args(argv)

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("PATH 中未找到 ffmpeg")

    cfg = load_config(args.config)
    db = Db(cfg.database_url)
    proxies: dict[str, str] = {}
    if getattr(cfg, "proxy_http", None):
        proxies["http"] = str(cfg.proxy_http)
    if getattr(cfg, "proxy_https", None):
        proxies["https"] = str(cfg.proxy_https)
    if (not proxies) and getattr(cfg, "proxy_url", None):
        proxies["http"] = str(cfg.proxy_url)
        proxies["https"] = str(cfg.proxy_url)

    client = HttpClient(
        cfg.base_urls,
        user_agent=cfg.user_agent,
        referer=cfg.referer,
        proxies=(proxies or None),
        trust_env=bool(getattr(cfg, "trust_env", False)),
        timeout_seconds=cfg.timeout_seconds,
        retries=cfg.retries,
        sleep_seconds=cfg.sleep_seconds,
    )

    out_dir = (args.out or cfg.download_out_dir or "").strip()
    if not out_dir:
        raise RuntimeError("缺少输出目录：请使用 --out 或在 config.yaml 中设置 download.out_dir")

    limit = int(args.limit) if args.limit is not None else int(cfg.download_limit)
    refresh_video_page = (not bool(args.no_refresh)) if args.no_refresh is not None else bool(cfg.download_refresh_video_page)
    move_to_complete = (not bool(args.no_move)) if args.no_move is not None else bool(cfg.download_move_to_complete)
    include_downloaded = (
        bool(args.include_downloaded) if args.include_downloaded is not None else bool(cfg.download_include_downloaded)
    )

    work_subdir = (args.work_subdir or cfg.download_work_subdir or "_working").strip() or "_working"
    complete_subdir = (args.complete_subdir or cfg.download_complete_subdir or "complete").strip() or "complete"
    workers = max(1, int(args.workers)) if args.workers is not None else max(1, int(cfg.download_workers))
    show_progress = (not bool(args.no_progress)) if args.no_progress is not None else bool(cfg.download_show_progress)
    progress_interval_seconds = float(cfg.download_progress_interval_seconds)
    dynamic_progress = (
        (not bool(args.no_dynamic_progress))
        if args.no_dynamic_progress is not None
        else bool(cfg.download_dynamic_progress)
    )
    name_max = int(cfg.download_name_max)
    engine = (args.engine or cfg.download_engine or "ffmpeg").strip().lower()
    if engine not in {"ffmpeg", "aria2"}:
        raise RuntimeError(f"无效的 download.engine：{engine!r}（应为 'ffmpeg' 或 'aria2'）")
    concurrent_segments = (
        max(1, int(args.concurrent_segments))
        if args.concurrent_segments is not None
        else max(1, int(cfg.download_concurrent_segments))
    )
    max_missing_segments = (
        max(0, int(args.max_missing_segments))
        if args.max_missing_segments is not None
        else max(0, int(cfg.download_max_missing_segments))
    )

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if not move_to_complete:
        work_root = out_root
        final_root = out_root
    else:
        work_root = out_root / work_subdir
        final_root = out_root / complete_subdir
        work_root.mkdir(parents=True, exist_ok=True)
        final_root.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": cfg.user_agent, "Referer": cfg.referer}

    videos = _load_videos(
        db,
        video_id=args.video_id,
        series_id=args.series_id,
        limit=max(1, int(limit)),
        include_downloaded=bool(include_downloaded or args.force),
    )

    if not videos:
        print("没有匹配到任何视频")
        return 0

    renderer = ProgressRenderer(dynamic=bool(dynamic_progress))

    def log(msg: str) -> None:
        renderer.log(msg)

    def process_one(v: VideoRow) -> None:
        if v.downloaded_at and not args.force:
            return

        try:
            vparsed = None
            series_name_for_dir = v.video_series_name
            series_id_for_db = v.video_series_source_key
            screenshot_urls_abs: list[str] = []

            if refresh_video_page:
                vp_res = client.fetch_path(f"/video/id-{v.video_id}.html")
                vparsed = parse_video_page(vp_res.text)
                ss_url = urljoin(vp_res.url, vparsed.screenshot_url) if vparsed.screenshot_url else None
                ss_urls = [urljoin(vp_res.url, u) for u in (vparsed.screenshot_urls or [])]
                screenshot_urls_abs = ss_urls or ([ss_url] if ss_url else [])
                update_video_detail(
                    db,
                    video_id=v.video_id,
                    h1=vparsed.h1,
                    title=vparsed.title,
                    canonical_url=vparsed.canonical_url,
                    cover_url=vparsed.cover_url,
                    screenshot_url=ss_url,
                    screenshot_urls=ss_urls,
                    m3u8_url=vparsed.m3u8_url,
                    poster_url=vparsed.poster_url,
                    upload_date=vparsed.upload_date,
                    duration_seconds=vparsed.duration_seconds,
                    content_rating=vparsed.content_rating,
                    is_family_friendly=vparsed.is_family_friendly,
                    jsonld=vparsed.video_object,
                    extract={
                        "breadcrumbs": [{"position": b.position, "name": b.name, "item": b.item} for b in vparsed.breadcrumbs]
                    },
                )

                if vparsed.upload_date:
                    pass
                if vparsed.duration_seconds:
                    pass

                picked = _pick_series_from_breadcrumbs(vparsed.breadcrumbs)
                if picked:
                    sid, sname = picked
                    series_id_for_db = sid
                    series_name_for_dir = sname
                    update_video_series(db, video_id=v.video_id, video_series_name=sname, video_series_source_key=sid)

            title = (v.title or v.h1 or f"xchina-{v.video_id}").strip()
            year = v.upload_date.year if v.upload_date else None
            premiered = v.upload_date.date().isoformat() if v.upload_date else None
            runtime_minutes = int(v.duration_seconds // 60) if v.duration_seconds else None
            if vparsed:
                title = title or (vparsed.h1 or vparsed.title or f"xchina-{v.video_id}")
                if vparsed.upload_date:
                    year = vparsed.upload_date.year
                    premiered = vparsed.upload_date.date().isoformat()
                if vparsed.duration_seconds:
                    runtime_minutes = int(vparsed.duration_seconds // 60)

            title = _clean_title(title)
            series_dir = _safe_name(series_name_for_dir or "Unknown", fallback="Unknown")
            base_name = _safe_name(title, fallback=f"xchina-{v.video_id}")

            file_title_raw = (vparsed.h1 if vparsed and vparsed.h1 else None) or v.h1 or title
            file_title = _clean_title(file_title_raw)
            file_base = _safe_name(file_title, fallback=f"xchina-{v.video_id}", max_len=_SAFE_FILE_MAX)

            # Directory naming: remove "[xchina-<id>]" suffix for cleaner paths.
            # Keep backward compatibility by also recognizing legacy paths with the id suffix.
            movie_base_preferred = f"{base_name} ({year})" if year else f"{base_name}"
            movie_base_legacy = f"{base_name} ({year}) [xchina-{v.video_id}]" if year else f"{base_name} [xchina-{v.video_id}]"

            preferred_movie_dir = work_root / series_dir / movie_base_preferred
            preferred_final_movie_dir = final_root / series_dir / movie_base_preferred
            preferred_video_path = preferred_movie_dir / f"{file_base}.mp4"
            preferred_final_video_path = preferred_final_movie_dir / f"{file_base}.mp4"

            legacy_movie_dir = work_root / series_dir / movie_base_legacy
            legacy_final_movie_dir = final_root / series_dir / movie_base_legacy
            legacy_video_path = legacy_movie_dir / f"{file_base}.mp4"
            legacy_final_video_path = legacy_final_movie_dir / f"{file_base}.mp4"

            if legacy_final_video_path.exists() and not args.force:
                mark_video_download_done(db, video_id=v.video_id, downloaded_path=str(legacy_final_video_path))
                return

            if preferred_final_video_path.exists() and not args.force:
                mark_video_download_done(db, video_id=v.video_id, downloaded_path=str(preferred_final_video_path))
                return

            if (not move_to_complete) and legacy_video_path.exists() and not args.force:
                mark_video_download_done(db, video_id=v.video_id, downloaded_path=str(legacy_video_path))
                return

            if (not move_to_complete) and preferred_video_path.exists() and not args.force:
                mark_video_download_done(db, video_id=v.video_id, downloaded_path=str(preferred_video_path))
                return

            def movie_dir_exists(movie_base: str) -> bool:
                return (work_root / series_dir / movie_base).exists() or (final_root / series_dir / movie_base).exists()

            # Avoid collisions after removing the id suffix (titles may repeat).
            movie_base = movie_base_preferred
            if movie_dir_exists(movie_base):
                i = 2
                while movie_dir_exists(f"{movie_base_preferred} ({i})"):
                    i += 1
                movie_base = f"{movie_base_preferred} ({i})"

            movie_dir = work_root / series_dir / movie_base
            final_movie_dir = final_root / series_dir / movie_base
            video_path = movie_dir / f"{file_base}.mp4"
            final_video_path = final_movie_dir / f"{file_base}.mp4"
            poster_path = movie_dir / "poster"
            extrafanart_dir = movie_dir / "extrafanart"
            nfo_path = video_path.with_suffix(".nfo")

            mark_video_download_attempt(db, video_id=v.video_id)

            # download video
            m3u8 = (vparsed.m3u8_url if vparsed and vparsed.m3u8_url else None) or v.m3u8_url
            if not m3u8:
                raise RuntimeError("缺少 m3u8_url（刷新失败或页面没有 m3u8）")

            duration_sec = None
            if vparsed and vparsed.duration_seconds:
                duration_sec = int(vparsed.duration_seconds)
            elif v.duration_seconds:
                duration_sec = int(v.duration_seconds)
            if duration_sec is None:
                duration_sec = _estimate_m3u8_duration_seconds(
                    m3u8, headers=headers, timeout_seconds=int(cfg.timeout_seconds)
                )
            if duration_sec is not None and runtime_minutes is None:
                runtime_minutes = max(1, int(duration_sec // 60))

            display_name = _shorten(title, max_len=name_max) or v.video_id
            if engine == "aria2":
                tmp_dir = movie_dir / ".hls"
                _download_m3u8_with_aria2(
                    m3u8_url=m3u8,
                    out_path=video_path,
                    tmp_dir=tmp_dir,
                    aria2c_path=str(cfg.download_aria2c_path),
                    concurrent_segments=int(concurrent_segments),
                    headers=headers,
                    show_progress=bool(show_progress),
                    renderer=renderer,
                    renderer_key=v.video_id,
                    display_name=display_name,
                    timeout_seconds=int(cfg.timeout_seconds),
                    progress_interval_seconds=progress_interval_seconds,
                    total_duration_seconds_hint=duration_sec,
                    max_missing_segments=int(max_missing_segments),
                )
                # 下载成功后清理分片目录（减少占用；若你希望保留以便排查/断点，可再加配置开关）
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass
            else:
                _ffmpeg_download_m3u8(
                    m3u8_url=m3u8,
                    out_path=video_path,
                    user_agent=cfg.user_agent,
                    referer=cfg.referer,
                    duration_seconds=duration_sec,
                    show_progress=bool(show_progress),
                    progress_prefix=display_name,
                    progress_interval_seconds=progress_interval_seconds,
                    on_progress=lambda line: renderer.update(v.video_id, line),
                )

            # images
            cover = (vparsed.cover_url if vparsed else None) or v.cover_url
            screenshot_urls = screenshot_urls_abs or v.screenshot_urls or []
            if not screenshot_urls:
                screenshot = v.screenshot_url
                if screenshot:
                    screenshot_urls = [screenshot]

            if cover:
                cover_u = urljoin(cfg.base_urls[0] + "/", cover) if cover.startswith("/") else cover
                ext = _ext_from_url(cover_u) or ".jpg"
                poster_dst = poster_path.with_suffix(ext)
                if args.force or not poster_dst.exists():
                    _download_file(cover_u, out_path=poster_dst, headers=headers, timeout_seconds=cfg.timeout_seconds)
                # Emby/Kodi common artwork: add fanart.* and thumb.* by copying poster.*
                _copy_file(poster_dst, movie_dir / f"fanart{ext}", force=bool(args.force))
                _copy_file(poster_dst, movie_dir / f"thumb{ext}", force=bool(args.force))
            if screenshot_urls:
                for i, u in enumerate(screenshot_urls, start=1):
                    u2 = urljoin(cfg.base_urls[0] + "/", u) if u.startswith("/") else u
                    ext = _ext_from_url(u2) or ".jpg"
                    dst = extrafanart_dir / f"fanart{i}{ext}"
                    if args.force or not dst.exists():
                        _download_file(u2, out_path=dst, headers=headers, timeout_seconds=cfg.timeout_seconds)

            # nfo
            jsonld_obj = (vparsed.video_object if vparsed else None) or v.jsonld
            authors = _extract_authors(jsonld_obj)
            actors = _extract_actors(jsonld_obj)
            tags = list(v.video_tags or [])
            if series_name_for_dir:
                tags.append(series_name_for_dir)
            plot = _extract_description(jsonld_obj)
            streamdetails = _ffprobe_streamdetails(video_path)
            nfo_title_raw = (vparsed.h1 if vparsed and vparsed.h1 else None) or v.h1 or title
            nfo_title = _clean_title(nfo_title_raw)
            suffix_tokens: list[str] = []
            if series_name_for_dir:
                suffix_tokens.append(series_name_for_dir)
            if vparsed and vparsed.breadcrumbs:
                for bc in vparsed.breadcrumbs:
                    item = getattr(bc, "item", "") or ""
                    if not _RE_BREADCRUMB_SERIES.search(item):
                        continue
                    name = (getattr(bc, "name", "") or "").strip()
                    if name:
                        suffix_tokens.append(name)
            nfo_title = _strip_trailing_title_tokens(nfo_title, tokens=suffix_tokens)
            if authors:
                nfo_title = f"{nfo_title} - {', '.join(authors)}"
            if args.force or not nfo_path.exists():
                _write_movie_nfo(
                    out_path=nfo_path,
                    video_id=v.video_id,
                    title=nfo_title,
                    premiered=premiered,
                    year=year,
                    runtime_minutes=runtime_minutes,
                    authors=authors,
                    actors=actors,
                    tags=tags,
                    series_name=series_name_for_dir,
                    plot=plot,
                    streamdetails=streamdetails,
                )

            final_path = video_path
            if move_to_complete:
                if final_movie_dir.exists():
                    if args.force:
                        shutil.rmtree(final_movie_dir)
                    else:
                        raise RuntimeError(f"目标目录已存在：{final_movie_dir}")
                final_movie_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(movie_dir), str(final_movie_dir))
                final_path = final_video_path

            mark_video_download_done(db, video_id=v.video_id, downloaded_path=str(final_path))
            renderer.update(v.video_id, f"[{_shorten(title, max_len=name_max)}] 完成 -> {final_path}")

        except Exception as exc:  # noqa: BLE001
            mark_video_download_error(db, video_id=v.video_id, error=str(exc))
            renderer.update(v.video_id, f"[{_shorten(v.title or v.h1 or v.video_id, max_len=name_max)}] 错误：{exc}")

    if workers <= 1:
        for v in videos:
            process_one(v)
        return 0

    # 多线程：按视频维度并发（ffmpeg 本身会占用 I/O/CPU；建议从 2~4 试起）
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_one, v) for v in videos]
        for fut in as_completed(futs):
            # 这里让异常在主线程抛出可见，但不终止其它任务
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                log(f"[警告] 工作线程失败：{exc}")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
