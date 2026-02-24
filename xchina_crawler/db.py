from __future__ import annotations

"""
数据库访问层（Postgres / psycopg3）。

特点：
- 不引入 ORM，直接执行 SQL（schema.sql）
- 以 upsert 为主，保证可重复运行/断点续跑
- 结构化字段 + jsonb 混合存储，便于后续清洗扩展
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


try:
    import psycopg
except Exception:  # noqa: BLE001
    psycopg = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Db:
    """
    数据库连接信息。

    Attributes:
        dsn: Postgres DSN
    """

    dsn: str

    def connect(self):
        """
        建立数据库连接。

        Returns:
            psycopg.Connection

        Raises:
            RuntimeError: 未安装 psycopg 依赖
        """

        if psycopg is None:
            raise RuntimeError("psycopg is not installed. Install requirements.txt first.")
        return psycopg.connect(self.dsn)


def init_db(db: Db, schema_path: str = "schema.sql") -> None:
    """
    初始化数据库表结构。

    Args:
        db: Db 配置
        schema_path: schema.sql 路径
    """

    schema_sql = Path(schema_path).read_text(encoding="utf-8")
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()


def upsert_taxonomy_node(
    db: Db,
    *,
    node_type: str,
    source_key: str,
    name: str,
    url: str | None,
    parent_id: int | None,
    item_count_hint: int | None,
) -> int:
    """
    插入或更新分类节点（taxonomy_nodes）。

    用于存放“板块/子分类”等 series 树结构。

    Args:
        db: Db 配置
        node_type: 节点类型（当前主要是 'video_series'）
        source_key: 外部唯一键（series_id）
        name: 节点名称
        url: 节点 URL（可为空）
        parent_id: 父节点 id（可为空）
        item_count_hint: 页面展示的数量提示（可为空）

    Returns:
        节点的自增 id（taxonomy_nodes.id）
    """

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
insert into taxonomy_nodes (node_type, source_key, name, url, parent_id, item_count_hint, last_seen_at, updated_at)
values (%s, %s, %s, %s, %s, %s, now(), now())
on conflict (source, node_type, source_key) do update set
  name = excluded.name,
  url = excluded.url,
  parent_id = coalesce(excluded.parent_id, taxonomy_nodes.parent_id),
  item_count_hint = coalesce(excluded.item_count_hint, taxonomy_nodes.item_count_hint),
  last_seen_at = now(),
  updated_at = now()
returning id
                """,
                (node_type, source_key, name, url, parent_id, item_count_hint),
            )
            (node_id,) = cur.fetchone()
        conn.commit()
        return int(node_id)


def get_taxonomy_node_id(db: Db, *, node_type: str, source_key: str) -> int | None:
    """
    查询分类节点 id。

    Args:
        db: Db 配置
        node_type: 节点类型
        source_key: 外部唯一键（series_id）

    Returns:
        节点 id 或 None
    """

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select id from taxonomy_nodes where source='xchina' and node_type=%s and source_key=%s",
                (node_type, source_key),
            )
            row = cur.fetchone()
            return int(row[0]) if row else None


def upsert_video_base(
    db: Db,
    *,
    video_id: str,
    title: str | None,
    page_url: str | None,
    canonical_url: str | None,
    cover_url: str | None,
    video_series_name: str | None = None,
    video_series_source_key: str | None = None,
    video_tags: list[str] | None = None,
) -> None:
    """
    upsert 视频基础信息（通常来自列表页）。

    Args:
        db: Db 配置
        video_id: 视频 id（主键）
        title: 标题（可为空）
        page_url: 详情页 URL（可为空）
        canonical_url: canonical（可为空，通常详情页补全）
        cover_url: 封面 URL（可为空）
        video_series_name: 当前抓取上下文下的 series 名称（可为空）
        video_series_source_key: 当前抓取上下文下的 series id（可为空）
        video_tags: 列表页 tags 纯文本项（可为空）
    """

    tags_json = json.dumps(video_tags, ensure_ascii=False) if video_tags else None

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
insert into videos (
  video_id, title, page_url, canonical_url, cover_url,
  video_series_name, video_series_source_key,
  video_tags,
  created_at, updated_at
)
values (%s, %s, %s, %s, %s, %s, %s, %s, now(), now())
on conflict (video_id) do update set
  title = coalesce(excluded.title, videos.title),
  page_url = coalesce(excluded.page_url, videos.page_url),
  canonical_url = coalesce(excluded.canonical_url, videos.canonical_url),
  cover_url = coalesce(excluded.cover_url, videos.cover_url),
  video_series_name = coalesce(excluded.video_series_name, videos.video_series_name),
  video_series_source_key = coalesce(excluded.video_series_source_key, videos.video_series_source_key),
  video_tags = coalesce(excluded.video_tags, videos.video_tags),
  updated_at = now()
                """,
                (
                    video_id,
                    title,
                    page_url,
                    canonical_url,
                    cover_url,
                    video_series_name,
                    video_series_source_key,
                    tags_json,
                ),
            )
        conn.commit()


def update_video_detail(
    db: Db,
    *,
    video_id: str,
    h1: str | None,
    title: str | None,
    canonical_url: str | None,
    cover_url: str | None,
    screenshot_url: str | None,
    upload_date: datetime | None,
    duration_seconds: int | None,
    content_rating: str | None,
    is_family_friendly: bool | None,
    jsonld: dict[str, Any] | None,
    extract: dict[str, Any] | None,
    m3u8_url: str | None = None,
    poster_url: str | None = None,
) -> None:
    """
    upsert 视频详情信息（通常来自详情页）。

    Args:
        db: Db 配置
        video_id: 视频 id
        h1: 页面 h1
        title: 页面 title
        canonical_url: canonical URL
        cover_url: 封面 URL
        screenshot_url: 截图 URL
        m3u8_url: 播放 m3u8 链接（可为空）
        poster_url: 播放器 poster 链接（可为空）
        upload_date: 上传时间（来自 JSON-LD）
        duration_seconds: 时长（秒，来自 JSON-LD）
        content_rating: 内容评级（来自 JSON-LD）
        is_family_friendly: 是否家庭友好（来自 JSON-LD）
        jsonld: JSON-LD VideoObject 原样（dict）
        extract: 额外解析结果（dict），例如 breadcrumbs 等
    """

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
insert into videos (
  video_id, h1, title, canonical_url, cover_url, screenshot_url,
  m3u8_url, poster_url,
  upload_date, duration_seconds, content_rating, is_family_friendly,
  jsonld, extract, created_at, updated_at, last_crawled_at
)
values (
  %s, %s, %s, %s, %s, %s,
  %s, %s,
  %s, %s, %s, %s,
  %s, %s, now(), now(), now()
)
on conflict (video_id) do update set
  h1 = coalesce(excluded.h1, videos.h1),
  title = coalesce(excluded.title, videos.title),
  canonical_url = coalesce(excluded.canonical_url, videos.canonical_url),
  cover_url = coalesce(excluded.cover_url, videos.cover_url),
  screenshot_url = coalesce(excluded.screenshot_url, videos.screenshot_url),
  m3u8_url = coalesce(excluded.m3u8_url, videos.m3u8_url),
  poster_url = coalesce(excluded.poster_url, videos.poster_url),
  upload_date = coalesce(excluded.upload_date, videos.upload_date),
  duration_seconds = coalesce(excluded.duration_seconds, videos.duration_seconds),
  content_rating = coalesce(excluded.content_rating, videos.content_rating),
  is_family_friendly = coalesce(excluded.is_family_friendly, videos.is_family_friendly),
  jsonld = coalesce(excluded.jsonld, videos.jsonld),
  extract = coalesce(excluded.extract, videos.extract),
  updated_at = now(),
  last_crawled_at = now()
                """,
                (
                    video_id,
                    h1,
                    title,
                    canonical_url,
                    cover_url,
                    screenshot_url,
                    m3u8_url,
                    poster_url,
                    upload_date,
                    duration_seconds,
                    content_rating,
                    is_family_friendly,
                    json.dumps(jsonld, ensure_ascii=False) if jsonld else None,
                    json.dumps(extract, ensure_ascii=False) if extract else None,
                ),
            )
        conn.commit()


def update_video_stream_links(
    db: Db,
    *,
    video_id: str,
    m3u8_url: str | None,
    poster_url: str | None,
) -> None:
    """
    单独更新 videos 表里的播放相关链接字段。

    适用场景：只抓了 download 页或其它页面，但仍能解析到 m3u8/poster。
    """

    if not m3u8_url and not poster_url:
        return

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
insert into videos (video_id, m3u8_url, poster_url, created_at, updated_at)
values (%s, %s, %s, now(), now())
on conflict (video_id) do update set
  m3u8_url = coalesce(excluded.m3u8_url, videos.m3u8_url),
  poster_url = coalesce(excluded.poster_url, videos.poster_url),
  updated_at = now()
                """,
                (video_id, m3u8_url, poster_url),
            )
        conn.commit()


def update_video_download_links(
    db: Db,
    *,
    video_id: str,
    magnet_uri: str | None,
    torrent_url: str | None,
) -> None:
    """
    单独更新 videos 表里的 download 页链接字段（magnet/torrent）。
    """

    if not magnet_uri and not torrent_url:
        return

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
insert into videos (video_id, video_magnet_uri, video_torrent_url, created_at, updated_at)
values (%s, %s, %s, now(), now())
on conflict (video_id) do update set
  video_magnet_uri = case when excluded.video_magnet_uri is not null then excluded.video_magnet_uri else videos.video_magnet_uri end,
  video_torrent_url = case when excluded.video_torrent_url is not null then excluded.video_torrent_url else videos.video_torrent_url end,
  updated_at = now()
                """,
                (video_id, magnet_uri, torrent_url),
            )
        conn.commit()


def update_video_series(
    db: Db,
    *,
    video_id: str,
    video_series_name: str | None,
    video_series_source_key: str | None,
) -> None:
    """
    更新 videos 表里的“影片具体所属 series”（用于纠正列表页的抓取上下文）。

    适用场景：
    - crawl-board 抓板块聚合列表页时，列表页无法精确区分子分类；
      但详情页/下载页的 JSON-LD BreadcrumbList 往往包含具体子分类 series。
    """

    if not video_series_source_key:
        return

    series_name = (video_series_name or "").strip() or f"series-{video_series_source_key}"

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
insert into videos (
  video_id, video_series_name, video_series_source_key,
  created_at, updated_at
)
values (%s, %s, %s, now(), now())
on conflict (video_id) do update set
  video_series_name = excluded.video_series_name,
  video_series_source_key = excluded.video_series_source_key,
  updated_at = now()
                """,
                (video_id, series_name, video_series_source_key),
            )
        conn.commit()


def upsert_video_taxonomy(db: Db, *, video_id: str, node_id: int, source: str) -> None:
    """
    建立视频与分类节点的关联（去重插入）。

    Args:
        db: Db 配置
        video_id: 视频 id
        node_id: taxonomy_nodes.id
        source: 关联来源（如 'list_page' / 'breadcrumb'）
    """

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
insert into video_taxonomy (video_id, node_id, source)
values (%s, %s, %s)
on conflict (video_id, node_id, source) do nothing
                """,
                (video_id, node_id, source),
            )
        conn.commit()


def insert_list_observation(
    db: Db,
    *,
    video_id: str,
    node_id: int | None,
    list_title: str | None,
    list_cover_url: str | None,
    list_badge: str | None,
    list_duration_text: str | None,
    list_duration_seconds: int | None,
    list_comment_count: int | None,
    list_flags: dict[str, Any],
    page_url: str | None,
) -> None:
    """
    插入列表页观测记录（每看到一次插一条）。

    该表设计用于后期清洗/对比（例如标题变更、卡片时长变更等）。

    Args:
        db: Db 配置
        video_id: 视频 id
        node_id: 当前所在的 series 节点 id
        list_title: 列表页标题
        list_cover_url: 列表页封面 URL
        list_badge: 列表页徽标（tags 第一项）
        list_duration_text: 列表页时长文本
        list_duration_seconds: 列表页时长（秒）
        list_comment_count: 列表页评论数
        list_flags: 其它标记（jsonb）
        page_url: 观测来源页 URL
    """

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
insert into video_list_observations (
  video_id, node_id,
  list_title, list_cover_url, list_badge,
  list_duration_text, list_duration_seconds, list_comment_count,
  list_flags, page_url
)
values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    video_id,
                    node_id,
                    list_title,
                    list_cover_url,
                    list_badge,
                    list_duration_text,
                    list_duration_seconds,
                    list_comment_count,
                    json.dumps(list_flags, ensure_ascii=False),
                    page_url,
                ),
            )
        conn.commit()


def upsert_download_page(
    db: Db,
    *,
    video_id: str,
    page_url: str | None,
    canonical_url: str | None,
    title: str | None,
    has_download_section: bool,
    has_magnet_button: bool,
    has_torrent_button: bool,
    external_link_domains: list[str],
    extract: dict[str, Any] | None,
) -> None:
    """
    upsert 下载页解析结果（video_download_pages）。

    Args:
        db: Db 配置
        video_id: 视频 id
        page_url: 下载页实际 URL
        canonical_url: 下载页 canonical
        title: 下载页 title
        has_download_section: 是否存在下载区块
        has_magnet_button: 是否存在 magnet 按钮
        has_torrent_button: 是否存在 torrent 按钮
        external_link_domains: 外链域名集合（jsonb）
        extract: 额外解析数据（jsonb），例如 breadcrumbs、下载页所有链接信息等
    """

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
insert into video_download_pages (
  video_id, page_url, canonical_url, title,
  has_download_section, has_magnet_button, has_torrent_button,
  external_link_domains, extract,
  created_at, updated_at, last_crawled_at
)
values (
  %s, %s, %s, %s,
  %s, %s, %s,
  %s, %s,
  now(), now(), now()
)
on conflict (video_id) do update set
  page_url = coalesce(excluded.page_url, video_download_pages.page_url),
  canonical_url = coalesce(excluded.canonical_url, video_download_pages.canonical_url),
  title = coalesce(excluded.title, video_download_pages.title),
  has_download_section = excluded.has_download_section,
  has_magnet_button = excluded.has_magnet_button,
  has_torrent_button = excluded.has_torrent_button,
  external_link_domains = excluded.external_link_domains,
  extract = excluded.extract,
  updated_at = now(),
  last_crawled_at = now()
                """,
                (
                    video_id,
                    page_url,
                    canonical_url,
                    title,
                    has_download_section,
                    has_magnet_button,
                    has_torrent_button,
                    json.dumps(external_link_domains, ensure_ascii=False),
                    json.dumps(extract, ensure_ascii=False) if extract else None,
                ),
            )
        conn.commit()


def insert_raw_page(
    db: Db,
    *,
    page_kind: str,
    source_key: str,
    url: str | None,
    html_sanitized: str,
    sha256: str,
    redactions: dict[str, Any] | None,
) -> None:
    """
    插入原始页面快照（已脱敏的 HTML）。

    说明：
    - html_sanitized 通常是对可直接下载/播放的直链做替换后的 HTML
    - sha256 用于后续去重/对比

    Args:
        db: Db 配置
        page_kind: 页面类型（'series' | 'video' | 'download'）
        source_key: 关联键（例如 video_id 或 series_id:page）
        url: 实际 URL
        html_sanitized: 处理后的 HTML 文本
        sha256: 处理后 HTML 的 sha256
        redactions: 替换统计（jsonb）
    """

    with db.connect() as conn:
        with conn.cursor() as cur:
            # URL 优先：如果同一 kind+source_key+url 的最新快照 sha256 一样，则跳过写入
            if url:
                cur.execute(
                    """
select sha256
from raw_pages
where page_kind=%s and source_key=%s and url=%s
order by fetched_at desc
limit 1
                    """,
                    (page_kind, source_key, url),
                )
                row = cur.fetchone()
                if row and str(row[0]) == sha256:
                    return

            # sha256 去重：如果同一 kind+source_key 已存在相同 sha256，则跳过写入
            cur.execute(
                """
select 1
from raw_pages
where page_kind=%s and source_key=%s and sha256=%s
limit 1
                """,
                (page_kind, source_key, sha256),
            )
            if cur.fetchone():
                return

            cur.execute(
                """
insert into raw_pages (page_kind, source_key, url, html_sanitized, redactions, sha256)
values (%s, %s, %s, %s, %s, %s)
                """,
                (
                    page_kind,
                    source_key,
                    url,
                    html_sanitized,
                    json.dumps(redactions, ensure_ascii=False) if redactions else None,
                    sha256,
                ),
            )
        conn.commit()


def get_latest_raw_page_by_url(
    db: Db,
    *,
    page_kind: str,
    url: str,
) -> tuple[str, str, datetime] | None:
    """
    按 URL 获取某类页面的最新 raw 快照（脱敏 HTML）。

    Returns:
        (html_sanitized, sha256, fetched_at) 或 None
    """

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
select html_sanitized, sha256, fetched_at
from raw_pages
where page_kind=%s and url=%s
order by fetched_at desc
limit 1
                """,
                (page_kind, url),
            )
            row = cur.fetchone()
            if not row:
                return None
            html_sanitized, sha256, fetched_at = row
            return str(html_sanitized), str(sha256), fetched_at
