from __future__ import annotations

"""
HTML 快照脱敏工具。

用途：
- 将页面中可直接用于下载/播放的 URL 替换为占位符
- 仍保留页面结构与文本，便于做页面分析/特征提取/调试
"""

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SanitizedHtml:
    """
    脱敏后的 HTML 结果。

    Attributes:
        html: 脱敏后的 HTML 文本
        sha256: 脱敏后 HTML 的 sha256，用于去重/对比
        redactions: 各类替换命中的计数（例如 magnet/torrent/m3u8）
    """

    html: str
    sha256: str
    redactions: dict[str, int]


_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("magnet", re.compile(r"magnet:\?xt=urn:btih:[^\"'<\s]+", re.I)),
    ("torrent", re.compile(r"https?://[^\"'<\s]+\.torrent[^\"'<\s]*", re.I)),
    ("m3u8", re.compile(r"https?://[^\"'<\s]+\.m3u8[^\"'<\s]*", re.I)),
]

def redact_direct_urls(text: str) -> tuple[str, dict[str, int]]:
    """
    将文本中可直接用于下载/播放的直链做替换。

    Args:
        text: 任意文本（URL、HTML、日志等）

    Returns:
        (redacted_text, counts)
        - redacted_text: 替换后的文本
        - counts: 每类替换命中的次数
    """

    counts: dict[str, int] = {}
    out = text
    for name, pat in _PATTERNS:
        out, n = pat.subn(f"{name.upper()}_REDACTED", out)
        if n:
            counts[name] = counts.get(name, 0) + n
    return text, counts


def sanitize_html(html: str) -> SanitizedHtml:
    """
    对 HTML 进行脱敏替换，并计算 sha256。

    当前替换规则（命中即替换为 `XXX_REDACTED`）：
    - magnet 链接：`magnet:?xt=urn:btih:...`
    - torrent 直链：`http(s)://... .torrent ...`
    - m3u8 直链：`http(s)://... .m3u8 ...`

    Args:
        html: 原始 HTML 文本

    Returns:
        SanitizedHtml: 脱敏后的结果与统计
    """

    out, redactions = redact_direct_urls(html)

    sha = hashlib.sha256(out.encode("utf-8", "ignore")).hexdigest()
    return SanitizedHtml(html=out, sha256=sha, redactions=redactions)
