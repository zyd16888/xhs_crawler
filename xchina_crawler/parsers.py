from __future__ import annotations

"""
页面解析模块（BeautifulSoup）。

目前覆盖三类页面：
- series 列表页：解析板块/子分类、分页信息、视频卡片摘要
- video 详情页：解析 JSON-LD（VideoObject/BreadcrumbList）与基础元数据
- download 下载页：解析页面结构与链接信息（用于分析/后续清洗）

说明：
- 解析结果以“尽量结构化”为主，同时保留必要的原始信息（例如下载页所有 <a> 信息）
- 由于本项目默认不存储可直接下载/播放的直链，相关内容请遵循合规要求
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from typing import Any
from urllib.parse import urlparse

from .sanitize import redact_direct_urls

def _soup(html: str):
    """
    构造 BeautifulSoup 对象。

    之所以做成函数，是为了：
    - CLI `--help` 等场景不必强依赖 bs4
    - 真正解析页面时如果缺依赖，给出清晰报错

    Args:
        html: 原始 HTML 文本

    Returns:
        BeautifulSoup: 解析后的 DOM

    Raises:
        RuntimeError: 未安装 bs4 依赖
    """

    try:
        from bs4 import BeautifulSoup  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺少依赖 bs4：请先执行 `pip install -r requirements.txt`") from exc
    return BeautifulSoup(html, "html.parser")


_RE_SERIES_PAGE_URL = re.compile(r"/videos/series-([a-f0-9]{8,})/(\d+)\.html", re.I)
_RE_VIDEO_URL = re.compile(r"/video/id-([a-f0-9]{8,})\.html", re.I)
_RE_SERIES_URL = re.compile(r"/videos/series-([a-f0-9]{8,})\.html", re.I)


def _strip_tags(text: str) -> str:
    """
    简单去除 HTML 标签并反转义实体。

    Args:
        text: 可能包含标签的字符串

    Returns:
        纯文本字符串
    """

    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text).strip()


def parse_int(s: str | None) -> int | None:
    """
    将字符串解析为 int（失败返回 None）。

    Args:
        s: 输入字符串

    Returns:
        int 或 None
    """

    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def parse_hhmmss_to_seconds(text: str | None) -> int | None:
    """
    将 `HH:MM:SS` 或 `MM:SS` 文本转换为秒数。

    Args:
        text: 时长文本，例如 `01:58:14`

    Returns:
        秒数（int）或 None
    """

    if not text:
        return None
    parts = [p for p in text.strip().split(":") if p != ""]
    if len(parts) == 2:
        hh = 0
        mm, ss = parts
    elif len(parts) == 3:
        hh, mm, ss = parts
    else:
        return None
    try:
        return int(hh) * 3600 + int(mm) * 60 + int(ss)
    except ValueError:
        return None


_RE_ISO8601_DURATION = re.compile(r"^PT(?:(\\d+)H)?(?:(\\d+)M)?(?:(\\d+)S)?$", re.I)


def parse_iso8601_duration_seconds(duration: str | None) -> int | None:
    """
    解析 ISO8601 时长（VideoObject 常用）为秒数。

    例：`PT33M45S` -> 2025 秒

    Args:
        duration: ISO8601 duration 字符串

    Returns:
        秒数或 None
    """

    if not duration:
        return None
    m = _RE_ISO8601_DURATION.match(duration.strip())
    if not m:
        return None
    h, m_, s = m.groups()
    return (int(h or 0) * 3600) + (int(m_ or 0) * 60) + int(s or 0)


def parse_rfc3339(dt: str | None) -> datetime | None:
    """
    解析 RFC3339/ISO8601 时间字符串为 datetime。

    例：`2026-02-03T09:29:05+08:00`

    Args:
        dt: 时间字符串

    Returns:
        datetime 或 None
    """

    if not dt:
        return None
    raw = dt.strip()
    # Example: 2026-02-03T09:29:05+08:00
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class SeriesChild:
    """
    板块（board）侧边栏里的一条子分类（子 series）。

    Attributes:
        series_id: 子 series 的 id（用于拼 `/videos/series-<id>/...`）
        name: 子分类名称（如“麻豆传媒”）
        item_count_hint: 页面展示的数量提示（如 3549），仅作参考
        url_path: 子 series 的 href（通常是 `/videos/series-<id>.html`）
    """

    series_id: str
    name: str
    item_count_hint: int | None
    url_path: str


@dataclass(frozen=True)
class VideoCard:
    """
    series 列表页中的视频卡片摘要信息。

    该结构用于：
    - 列表页快速入库（视频基础信息）
    - 记录列表页观测（徽标、评论数、卡片时长等）

    Attributes:
        video_id: 视频 id
        title: 标题（可能为空）
        page_path: 详情页路径（相对）
        cover_url: 封面 URL（从 style background-image 提取）
        tags: 列表页 tags 的纯文本项（已过滤图标/空占位等）
        badge: 卡片 tags 的第一项（如“杏吧原版”）
        duration_text: 卡片显示的时长文本
        duration_seconds: 卡片时长转换为秒
        comment_count: 评论数（若页面提供）
        has_magnet_icon: 卡片是否展示“磁铁”图标（仅标记）
    """

    video_id: str
    title: str | None
    page_path: str
    cover_url: str | None
    tags: list[str]
    badge: str | None
    duration_text: str | None
    duration_seconds: int | None
    comment_count: int | None
    has_magnet_icon: bool


@dataclass(frozen=True)
class SeriesPageParsed:
    """
    series 列表页解析结果。

    注意区分两个名字：
    - series_name：当前页面 `<h1>`（当前 series 的真实名称，如“模特私拍”）
    - board_name：侧边栏“影片分类”里的 `<div class="parent">...`（板块父级名称，如“中文AV”）
    """

    series_id: str | None
    page_number: int | None
    series_name: str | None
    board_name: str | None
    last_page_number: int | None
    children: list[SeriesChild]
    video_cards: list[VideoCard]


def _extract_bg_image_url(style: str | None) -> str | None:
    """
    从 style 属性中提取 background-image URL。

    Args:
        style: style 字符串，例如 `background-image:url('https://...')`

    Returns:
        URL 或 None
    """

    if not style:
        return None
    m = re.search(r"background-image\s*:\s*url\((['\"]?)([^'\")]+)\1\)", style, re.I)
    if not m:
        return None
    return m.group(2).strip() or None


def _parse_series_page_bs4(path: str, html: str) -> SeriesPageParsed:
    """
    解析 series 列表页（BeautifulSoup 版本）。

    Args:
        path: 当前请求路径（用于解析 series_id/page_number），例如 `/videos/series-xxx/1.html`
        html: HTML 文本

    Returns:
        SeriesPageParsed
    """

    series_id: str | None = None
    page_number: int | None = None
    m = _RE_SERIES_PAGE_URL.search(path)
    if m:
        series_id, page_s = m.groups()
        page_number = int(page_s)
    else:
        m2 = _RE_SERIES_URL.search(path)
        if m2:
            (series_id,) = m2.groups()

    soup = _soup(html)

    series_name = None
    h1_el = soup.find("h1")
    if h1_el:
        series_name = h1_el.get_text(" ", strip=True) or None

    board_name = None
    series_box = soup.select_one("div.content-box.series")
    if series_box:
        parent_div = series_box.select_one("div.parent")
        if parent_div:
            board_name = parent_div.get_text(strip=True) or None

    last_page_number = None
    nums: list[int] = []
    for a in soup.find_all("a", href=True):
        href = str(a.get("href"))
        m_page = re.search(r"/videos/series-[^/]+/(\d+)\.html", href, re.I)
        if m_page:
            try:
                nums.append(int(m_page.group(1)))
            except ValueError:
                pass
    if nums:
        last_page_number = max(nums)

    children: list[SeriesChild] = []
    if series_box:
        for a in series_box.find_all("a", href=True):
            href = str(a.get("href"))
            m_sid = _RE_SERIES_URL.search(href)
            if not m_sid:
                continue
            sub_div = a.find("div", class_="sub")
            if not sub_div:
                continue
            text = sub_div.get_text(" ", strip=True)
            m_name_cnt = re.match(r"^(.*?)\s*\((\d{1,9})\)\s*$", text)
            if m_name_cnt:
                name = m_name_cnt.group(1).strip()
                cnt = parse_int(m_name_cnt.group(2))
            else:
                name = text.strip()
                cnt = None
            children.append(
                SeriesChild(
                    series_id=m_sid.group(1),
                    name=name,
                    item_count_hint=cnt,
                    url_path=href,
                )
            )

    video_cards: list[VideoCard] = []
    for item in soup.select("div.item.video"):
        a = item.find("a", href=_RE_VIDEO_URL)
        if not a or not a.get("href"):
            continue
        page_path = str(a.get("href"))
        m_vid = _RE_VIDEO_URL.search(page_path)
        if not m_vid:
            continue
        video_id = m_vid.group(1)

        title = None
        if a.get("title"):
            title = unescape(str(a.get("title"))).strip() or None
        if not title:
            title_a = item.select_one("div.title a")
            if title_a:
                title = title_a.get_text(strip=True) or None

        cover_url = None
        img_div = item.select_one("div.img")
        if img_div:
            cover_url = _extract_bg_image_url(img_div.get("style"))

        tags: list[str] = []
        tags_div = item.select_one("div.tags")
        if tags_div:
            seen: set[str] = set()
            for div in tags_div.find_all("div", recursive=False):
                classes = set(div.get("class") or [])
                if "empty" in classes:
                    continue
                # 排除：磁链/评论/时长等（通常带 <i> 图标）
                if div.find("i") is not None:
                    continue
                t = div.get_text(" ", strip=True) or ""
                t = t.strip()
                if not t:
                    continue
                if t in seen:
                    continue
                seen.add(t)
                tags.append(t)

        badge = None
        badge_div = item.select_one("div.tags > div")
        if badge_div:
            badge = badge_div.get_text(strip=True) or None

        duration_text = None
        duration_seconds = None
        clock_i = item.select_one("div.tags i.fa-clock")
        if clock_i and clock_i.parent:
            parent_text = clock_i.parent.get_text(" ", strip=True)
            m_dur = re.search(r"(\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})", parent_text)
            if m_dur:
                duration_text = m_dur.group(1)
                duration_seconds = parse_hhmmss_to_seconds(duration_text)

        comment_count = None
        comments_i = item.select_one("div.tags i.fa-comments")
        if comments_i and comments_i.parent:
            parent_text = comments_i.parent.get_text(" ", strip=True)
            m_cmt = re.search(r"(\d{1,9})", parent_text)
            if m_cmt:
                comment_count = parse_int(m_cmt.group(1))

        has_magnet_icon = bool(item.select_one("div.tags i.fa-magnet"))

        video_cards.append(
            VideoCard(
                video_id=video_id,
                title=title,
                page_path=page_path,
                cover_url=cover_url,
                tags=tags,
                badge=badge,
                duration_text=duration_text,
                duration_seconds=duration_seconds,
                comment_count=comment_count,
                has_magnet_icon=has_magnet_icon,
            )
        )

    return SeriesPageParsed(
        series_id=series_id,
        page_number=page_number,
        series_name=series_name,
        board_name=board_name,
        last_page_number=last_page_number,
        children=children,
        video_cards=video_cards,
    )


def parse_series_page(path: str, html: str) -> SeriesPageParsed:
    """
    对外入口：解析 series 列表页。

    Args:
        path: 请求路径
        html: HTML 文本

    Returns:
        SeriesPageParsed
    """

    return _parse_series_page_bs4(path, html)


@dataclass(frozen=True)
class BreadcrumbItem:
    """
    JSON-LD BreadcrumbList 中的单个面包屑条目。

    Attributes:
        position: 位置（从 1 开始）
        name: 文本名称
        item: URL（可能为空）
    """

    position: int
    name: str
    item: str


@dataclass(frozen=True)
class VideoPageParsed:
    """
    video 详情页解析结果。

    主要数据来源：
    - `<h1>` / `<title>` / canonical / og:image
    - JSON-LD VideoObject（uploadDate/duration/identifier 等）
    - JSON-LD BreadcrumbList（用于归类）
    """

    video_id: str | None
    h1: str | None
    title: str | None
    canonical_url: str | None
    cover_url: str | None
    screenshot_url: str | None
    m3u8_url: str | None
    poster_url: str | None
    upload_date: datetime | None
    duration_seconds: int | None
    content_rating: str | None
    is_family_friendly: bool | None
    video_object: dict[str, Any] | None
    breadcrumbs: list[BreadcrumbItem]


@dataclass(frozen=True)
class DownloadPageParsed:
    """
    download 下载页解析结果。

    本结构既用于“页面结构分析”（是否存在下载区块/按钮），也用于保存原始链接信息便于后续清洗。

    Attributes:
        title: 页面 title
        canonical_url: canonical URL
        has_download_section: 是否存在下载区块（div.download-section）
        has_magnet_button: 是否存在 magnet 按钮（a.btn.magnet）
        has_torrent_button: 是否存在 torrent 按钮（a.btn.download）
        external_link_domains: 页面中 http(s) 外链域名集合（排除站内相对路径、排除 magnet:）
        links: 页面所有 <a> 的信息（text/href/class），用于后续清洗与归因
        breadcrumbs: JSON-LD 面包屑（若存在）
    """

    title: str | None
    canonical_url: str | None
    m3u8_url: str | None
    poster_url: str | None
    magnet_uri: str | None
    torrent_url: str | None
    has_download_section: bool
    has_magnet_button: bool
    has_torrent_button: bool
    external_link_domains: list[str]
    links: list[dict[str, str]]
    breadcrumbs: list[BreadcrumbItem]


_RE_URL_M3U8 = re.compile(r"https?://[^\"'<\s]+\.m3u8[^\"'<\s]*", re.I)
_RE_JS_SRC_M3U8 = re.compile(r"\bsrc\s*:\s*(['\"])(https?://[^'\"]+?\.m3u8[^'\"]*)\1", re.I)
_RE_JS_POSTER = re.compile(r"\bposter\s*:\s*(['\"])(https?://[^'\"]+)\1", re.I)


def _extract_first_m3u8_url(html: str) -> str | None:
    m = _RE_JS_SRC_M3U8.search(html)
    if m:
        return m.group(2).strip() or None
    m2 = _RE_URL_M3U8.search(html)
    if m2:
        return m2.group(0).strip() or None
    return None


def _extract_first_poster_url(html: str) -> str | None:
    m = _RE_JS_POSTER.search(html)
    if m:
        return m.group(2).strip() or None
    return None


def _parse_jsonld_blocks(html: str) -> list[Any]:
    """
    用正则提取 JSON-LD 脚本块并 json.loads 解析。

    说明：
    - 这里不依赖 DOM，是为了兼容某些页面脚本块内容为空/被压缩等情况

    Args:
        html: HTML 文本

    Returns:
        解析后的 JSON 对象列表（可能是 dict 或 list）
    """

    blocks = re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.I | re.S)
    out: list[Any] = []
    for blk in blocks:
        blk = blk.strip()
        if not blk:
            continue
        try:
            out.append(json.loads(blk))
        except json.JSONDecodeError:
            continue
    return out


def _parse_video_page_bs4(html: str) -> VideoPageParsed:
    """
    解析 video 详情页（BeautifulSoup 版本）。

    Args:
        html: HTML 文本

    Returns:
        VideoPageParsed
    """

    soup = _soup(html)

    title = None
    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)

    h1 = None
    h1_el = soup.find("h1")
    if h1_el:
        h1 = h1_el.get_text(" ", strip=True) or None

    canonical_url = None
    can = soup.find("link", rel=lambda v: v and "canonical" in v)
    if can and can.get("href"):
        canonical_url = str(can.get("href")).strip() or None

    cover_url = None
    og = soup.find("meta", attrs={"property": "og:image"})
    if og and og.get("content"):
        cover_url = str(og.get("content")).strip() or None

    screenshot_url = None
    img = soup.find("img", src=re.compile(r"/screenshot/", re.I))
    if img and img.get("src"):
        screenshot_url = str(img.get("src")).strip() or None

    m3u8_url = _extract_first_m3u8_url(html)
    poster_url = _extract_first_poster_url(html)

    jsonld_blocks = []
    for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = s.string if s.string is not None else s.get_text()
        if not raw or not raw.strip():
            continue
        try:
            jsonld_blocks.append(json.loads(raw))
        except json.JSONDecodeError:
            continue

    flattened: list[Any] = []
    for blk in jsonld_blocks:
        if isinstance(blk, list):
            flattened.extend(blk)
        else:
            flattened.append(blk)

    video_object: dict[str, Any] | None = None
    breadcrumbs: list[BreadcrumbItem] = []
    for obj in flattened:
        if not isinstance(obj, dict):
            continue
        if obj.get("@type") == "VideoObject" and video_object is None:
            video_object = obj
        if obj.get("@type") == "BreadcrumbList":
            elems = obj.get("itemListElement") or []
            if isinstance(elems, list):
                for it in elems:
                    if not isinstance(it, dict):
                        continue
                    pos = parse_int(str(it.get("position"))) or 0
                    name = str(it.get("name") or "").strip()
                    item = str(it.get("item") or "").strip()
                    if pos and name:
                        breadcrumbs.append(BreadcrumbItem(position=pos, name=name, item=item))

    video_id = None
    upload_date = None
    duration_seconds = None
    content_rating = None
    is_family_friendly = None

    if video_object:
        vid = video_object.get("identifier")
        if isinstance(vid, str) and vid.strip():
            video_id = vid.strip()
        upload_date = parse_rfc3339(video_object.get("uploadDate"))
        duration_seconds = parse_iso8601_duration_seconds(video_object.get("duration"))
        content_rating = video_object.get("contentRating")
        if "isFamilyFriendly" in video_object:
            is_family_friendly = bool(video_object.get("isFamilyFriendly"))
        if not cover_url:
            thumb = video_object.get("thumbnailUrl")
            if isinstance(thumb, str) and thumb.strip():
                cover_url = thumb.strip()

    if not video_id:
        # fallback from canonical/url patterns in HTML
        any_url = canonical_url or ""
        m = re.search(r"/video/id-([a-f0-9]{8,})\.html", any_url, re.I)
        if m:
            video_id = m.group(1)
        else:
            m2 = re.search(r"/video/id-([a-f0-9]{8,})\.html", html, re.I)
            if m2:
                video_id = m2.group(1)

    return VideoPageParsed(
        video_id=video_id,
        h1=h1,
        title=title,
        canonical_url=canonical_url,
        cover_url=cover_url,
        screenshot_url=screenshot_url,
        m3u8_url=m3u8_url,
        poster_url=poster_url or cover_url,
        upload_date=upload_date,
        duration_seconds=duration_seconds,
        content_rating=content_rating,
        is_family_friendly=is_family_friendly,
        video_object=video_object,
        breadcrumbs=sorted(breadcrumbs, key=lambda x: x.position),
    )


def parse_video_page(html: str) -> VideoPageParsed:
    """
    对外入口：解析 video 详情页。

    Args:
        html: HTML 文本

    Returns:
        VideoPageParsed
    """

    return _parse_video_page_bs4(html)


def parse_download_page(html: str) -> DownloadPageParsed:
    """
    解析 download 下载页（BeautifulSoup）。

    Args:
        html: HTML 文本

    Returns:
        DownloadPageParsed
    """

    soup = _soup(html)

    title = None
    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)

    canonical_url = None
    can = soup.find("link", rel=lambda v: v and "canonical" in v)
    if can and can.get("href"):
        canonical_url = str(can.get("href")).strip() or None

    download_section = soup.select_one("div.download-section")
    has_download_section = download_section is not None

    has_magnet_button = bool(soup.select_one("a.btn.magnet"))
    has_torrent_button = bool(soup.select_one("a.btn.download"))

    magnet_uri = None
    magnet_a = soup.select_one("a.btn.magnet[href^='magnet:']") or soup.select_one("a[href^='magnet:']")
    if magnet_a and magnet_a.get("href"):
        magnet_uri = str(magnet_a.get("href")).strip() or None

    torrent_url = None
    torrent_a = soup.select_one("a.btn.download[href]") or soup.select_one("a[href$='.torrent']")
    if torrent_a and torrent_a.get("href"):
        torrent_url = str(torrent_a.get("href")).strip() or None

    m3u8_url = _extract_first_m3u8_url(html)
    poster_url = _extract_first_poster_url(html)
    if not poster_url:
        # download 页通常会展示 cover/screenshot 预览图，优先取 cover 作为 poster 的退化方案
        img = soup.find("img", src=re.compile(r"/cover/|/cover_", re.I))
        if img and img.get("src"):
            poster_url = str(img.get("src")).strip() or None

    # collect external link domains (exclude same-site relative paths)
    domains: set[str] = set()
    links: list[dict[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = str(a.get("href")).strip()
        if not href:
            continue
        link_text = a.get_text(" ", strip=True)[:200]
        link_class = " ".join(a.get("class") or [])[:200]
        href_redacted, _ = redact_direct_urls(href)
        links.append({"text": link_text, "href": href_redacted, "class": link_class})
        if href.startswith("/"):
            continue
        if href.startswith("magnet:"):
            continue
        try:
            u = urlparse(href)
        except Exception:
            continue
        if not u.scheme.startswith("http"):
            continue
        if not u.netloc:
            continue
        domains.add(u.netloc.lower())

    # breadcrumb from jsonld
    breadcrumbs: list[BreadcrumbItem] = []
    jsonld_blocks: list[Any] = []
    for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = s.string if s.string is not None else s.get_text()
        if not raw or not raw.strip():
            continue
        try:
            jsonld_blocks.append(json.loads(raw))
        except json.JSONDecodeError:
            continue

    flattened: list[Any] = []
    for blk in jsonld_blocks:
        if isinstance(blk, list):
            flattened.extend(blk)
        else:
            flattened.append(blk)

    for obj in flattened:
        if not isinstance(obj, dict):
            continue
        if obj.get("@type") != "BreadcrumbList":
            continue
        elems = obj.get("itemListElement") or []
        if isinstance(elems, list):
            for it in elems:
                if not isinstance(it, dict):
                    continue
                pos = parse_int(str(it.get("position"))) or 0
                name = str(it.get("name") or "").strip()
                item = str(it.get("item") or "").strip()
                if pos and name:
                    breadcrumbs.append(BreadcrumbItem(position=pos, name=name, item=item))

    return DownloadPageParsed(
        title=title,
        canonical_url=canonical_url,
        m3u8_url=m3u8_url,
        poster_url=poster_url,
        magnet_uri=magnet_uri,
        torrent_url=torrent_url,
        has_download_section=has_download_section,
        has_magnet_button=has_magnet_button,
        has_torrent_button=has_torrent_button,
        external_link_domains=sorted(domains),
        links=links,
        breadcrumbs=sorted(breadcrumbs, key=lambda x: x.position),
    )
