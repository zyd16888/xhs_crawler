from __future__ import annotations

"""
命令行入口（CLI）。

提供的子命令：
- init-db：初始化 Postgres 表结构（schema.sql）
- discover-series：对“板块 series”的第 1 页进行解析，发现子分类并写入 taxonomy_nodes
- crawl-series：爬单个 series 的列表分页（可选抓详情页、下载页、原始快照）
- crawl-board：先解析板块（父级）并发现子分类，再逐个爬取子分类 series

设计要点：
- 先抓列表页快速落库，再按需补全详情/下载页，便于增量与断点续跑
- 分类树以 taxonomy_nodes 存储，视频与分类关系存于 video_taxonomy
"""

import argparse
import re
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from typing import Any
from urllib.parse import urljoin

from .config import load_config
from .db import (
    Db,
    get_taxonomy_node_id,
    get_latest_raw_page_by_url,
    init_db,
    insert_list_observation,
    insert_raw_page,
    upsert_download_page,
    update_video_download_links,
    update_video_series,
    update_video_detail,
    update_video_stream_links,
    upsert_taxonomy_node,
    upsert_video_base,
    upsert_video_taxonomy,
)
from .http_client import HttpClient
from .parsers import parse_download_page, parse_series_page, parse_video_page
from .sanitize import sanitize_html


NODE_TYPE_SERIES = "video_series"

LOG = logging.getLogger("xchina_crawler")

_RE_BREADCRUMB_SERIES = re.compile(r"/videos/series-([a-f0-9]{8,})(?:/\d+\.html|\.html)", re.I)


def _pick_specific_series_from_breadcrumbs(*, current_series_id: str, breadcrumbs: list[Any]) -> tuple[str, str] | None:
    """
    从 BreadcrumbList 中挑出“最具体”的 series（通常是最后一个 series 链接）。

    返回值用于纠正 crawl-board 聚合列表页下的 video_series_* 字段：
    - 当 breadcrumbs 里有多个 series（板块 + 子分类）时，取最后一个（子分类）
    - 当只有一个 series 时，仅在它等于当前列表页 series_id 时才采纳（避免误覆盖）
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

    if not matches:
        return None

    sid, name = matches[-1]
    if len(matches) >= 2 or sid == current_series_id:
        return sid, name
    return None


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
    )


def _fetch_path_logged(client: HttpClient, *, path: str, what: str, key: str | None = None) -> tuple[str, str]:
    """
    抓取一个 path，并输出可观测性日志（开始/结束/耗时）。

    Returns:
        (final_url, html_text)
    """

    msg = f"fetch start what={what} path={path}"
    if key:
        msg += f" key={key}"
    LOG.info(msg)
    t0 = time.perf_counter()
    res = client.fetch_path(path)
    dt_ms = int((time.perf_counter() - t0) * 1000)
    LOG.info(f"fetch done what={what} status={res.status_code} ms={dt_ms} url={res.url}")
    return res.url, res.text


def _maybe_load_cached_html(
    *,
    db: Db,
    client: HttpClient,
    page_kind: str,
    path: str,
    max_age_seconds: int | None,
) -> tuple[str, str] | None:
    """
    URL 优先的 raw_pages 快取：按候选 URL 顺序查找最新快照。

    Returns:
        (url, html_sanitized) 或 None

    注意：
        raw_pages 存的是脱敏 HTML（直链可能被替换），仅适合不依赖直链字段的解析场景。
    """

    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc)
    for url in client.candidate_urls(path):
        hit = get_latest_raw_page_by_url(db, page_kind=page_kind, url=url)
        if not hit:
            continue
        html_sanitized, _sha256, fetched_at = hit

        if max_age_seconds is not None:
            fa = fetched_at
            if fa.tzinfo is None:
                fa = fa.replace(tzinfo=timezone.utc)
            age = (now - fa).total_seconds()
            if age > max_age_seconds:
                continue

        LOG.info(f"raw cache hit kind={page_kind} url={url}")
        return url, html_sanitized
    return None


def _make_client(cfg) -> HttpClient:
    """
    根据配置构造 HttpClient。

    Args:
        cfg: Config

    Returns:
        HttpClient
    """

    return HttpClient(
        cfg.base_urls,
        user_agent=cfg.user_agent,
        referer=cfg.referer,
        timeout_seconds=cfg.timeout_seconds,
        retries=cfg.retries,
        sleep_seconds=cfg.sleep_seconds,
    )


def cmd_init_db(args: argparse.Namespace) -> int:
    """
    子命令：init-db

    功能：执行 schema.sql 创建/更新表结构。
    """

    cfg = load_config(args.config)
    init_db(Db(cfg.database_url), schema_path=args.schema)
    print("db initialized")
    return 0


def cmd_discover_series(args: argparse.Namespace) -> int:
    """
    子命令：discover-series

    功能：
    - 抓取板块 series 的第 1 页
    - 解析侧边栏“影片分类”区块，发现子分类（子 series）
    - 写入 taxonomy_nodes（建立父子关系）

    注意：
    - 该命令需要传入“板块父级”的 series_id（页面 h1 与侧边栏 parent 一致）
    - 如果传入的是子分类页，会直接报错，避免写坏分类树
    """

    cfg = load_config(args.config)
    db = Db(cfg.database_url)
    client = _make_client(cfg)

    path = f"/videos/series-{args.series_id}/1.html"
    res = client.fetch_path(path)
    parsed = parse_series_page(path, res.text)

    if parsed.board_name and parsed.series_name and parsed.board_name != parsed.series_name:
        raise RuntimeError(
            f"series_id={args.series_id} 看起来是子分类页：h1={parsed.series_name!r}，影片分类父级={parsed.board_name!r}；"
            "discover-series 需要传入板块（父级）对应的 series_id"
        )

    parent_name = parsed.series_name or f"series-{args.series_id}"
    parent_url = res.url
    parent_id = upsert_taxonomy_node(
        db,
        node_type=NODE_TYPE_SERIES,
        source_key=args.series_id,
        name=parent_name,
        url=parent_url,
        parent_id=None,
        item_count_hint=None,
    )
    print(f"parent: {args.series_id} -> node_id={parent_id} name={parent_name}")

    for child in parsed.children:
        child_abs_url = None
        # keep absolute url using same base as fetched
        if child.url_path.startswith("/"):
            child_abs_url = res.url.split("/videos/")[0] + child.url_path
        else:
            child_abs_url = child.url_path

        child_id = upsert_taxonomy_node(
            db,
            node_type=NODE_TYPE_SERIES,
            source_key=child.series_id,
            name=child.name,
            url=child_abs_url,
            parent_id=parent_id,
            item_count_hint=child.item_count_hint,
        )
        print(f" child: {child.series_id} -> node_id={child_id} name={child.name} count={child.item_count_hint}")

    print(f"children: {len(parsed.children)}")
    return 0


def _crawl_series_pages(
    *,
    db: Db,
    client: HttpClient,
    series_id: str,
    node_id: int,
    start_page: int,
    end_page: int | None,
    max_pages: int | None,
    crawl_video_detail: bool,
    crawl_video_download: bool,
    store_raw: bool,
    use_raw_cache: bool,
    raw_cache_max_age_seconds: int,
    workers: int,
) -> None:
    """
    爬取某个 series 的列表分页，并可选补全详情页/下载页。

    Args:
        db: 数据库配置
        client: HTTP 客户端
        series_id: series id
        node_id: taxonomy_nodes.id（该 series 对应的节点）
        start_page: 起始页（从 1 开始）
        end_page: 结束页（可选）
        max_pages: 最大抓取页数（可选，用于限制跑量）
        crawl_video_detail: 是否抓取详情页补全 videos 表
        crawl_video_download: 是否抓取下载页补全 video_download_pages 表（需要先有 video_id）
        store_raw: 是否存储 raw_pages（会对直链做替换脱敏）
    """

    current = start_page
    pages_done = 0
    while True:
        if end_page is not None and current > end_page:
            break
        if max_pages is not None and pages_done >= max_pages:
            break

        path = f"/videos/series-{series_id}/{current}.html"
        LOG.info(f"page start series_id={series_id} page={current} path={path}")

        cached = None
        if store_raw and use_raw_cache:
            cached = _maybe_load_cached_html(
                db=db,
                client=client,
                page_kind="series",
                path=path,
                max_age_seconds=(raw_cache_max_age_seconds if raw_cache_max_age_seconds > 0 else None),
            )

        if cached:
            page_url, html = cached
        else:
            page_url, html = _fetch_path_logged(client, path=path, what="series_list", key=f"{series_id}:{current}")

        parsed = parse_series_page(path, html)
        LOG.info(
            f"page parsed series_id={series_id} page={current} videos={len(parsed.video_cards)} last_page={parsed.last_page_number}"
        )

        if store_raw:
            san = sanitize_html(html)
            insert_raw_page(
                db,
                page_kind="series",
                source_key=f"{series_id}:{current}",
                url=page_url,
                html_sanitized=san.html,
                sha256=san.sha256,
                redactions=san.redactions,
            )

        # update node name from the page's h1 (real series name)
        if parsed.series_name:
            upsert_taxonomy_node(
                db,
                node_type=NODE_TYPE_SERIES,
                source_key=series_id,
                name=parsed.series_name,
                url=page_url,
                parent_id=None,
                item_count_hint=None,
            )

        def process_card(card) -> None:
            # allow crawling download page without detail page
            if crawl_video_download and not crawl_video_detail:
                download_path = f"/download/id-{card.video_id}.html"
                dl_cached = None
                if store_raw and use_raw_cache:
                    dl_cached = _maybe_load_cached_html(
                        db=db,
                        client=client,
                        page_kind="download",
                        path=download_path,
                        max_age_seconds=(raw_cache_max_age_seconds if raw_cache_max_age_seconds > 0 else None),
                    )
                if dl_cached:
                    dl_url, dl_html = dl_cached
                else:
                    dl_url, dl_html = _fetch_path_logged(client, path=download_path, what="download", key=card.video_id)

                dl_parsed = parse_download_page(dl_html)
                if store_raw:
                    san = sanitize_html(dl_html)
                    insert_raw_page(
                        db,
                        page_kind="download",
                        source_key=card.video_id,
                        url=dl_url,
                        html_sanitized=san.html,
                        sha256=san.sha256,
                        redactions=san.redactions,
                    )
                upsert_download_page(
                    db,
                    video_id=card.video_id,
                    page_url=dl_url,
                    canonical_url=dl_parsed.canonical_url,
                    title=dl_parsed.title,
                    has_download_section=dl_parsed.has_download_section,
                    has_magnet_button=dl_parsed.has_magnet_button,
                    has_torrent_button=dl_parsed.has_torrent_button,
                    external_link_domains=dl_parsed.external_link_domains,
                    extract={
                        "breadcrumbs": [asdict(b) for b in dl_parsed.breadcrumbs],
                        "links": dl_parsed.links,
                    },
                )
                update_video_stream_links(
                    db,
                    video_id=card.video_id,
                    m3u8_url=dl_parsed.m3u8_url,
                    poster_url=dl_parsed.poster_url,
                )
                update_video_download_links(
                    db,
                    video_id=card.video_id,
                    magnet_uri=dl_parsed.magnet_uri,
                    torrent_url=(urljoin(dl_url, dl_parsed.torrent_url) if dl_parsed.torrent_url else None),
                )
                picked = _pick_specific_series_from_breadcrumbs(
                    current_series_id=series_id, breadcrumbs=dl_parsed.breadcrumbs
                )
                if picked:
                    sid, sname = picked
                    update_video_series(
                        db, video_id=card.video_id, video_series_name=sname, video_series_source_key=sid
                    )

            if crawl_video_detail:
                detail_cached = None
                if store_raw and use_raw_cache:
                    detail_cached = _maybe_load_cached_html(
                        db=db,
                        client=client,
                        page_kind="video",
                        path=card.page_path,
                        max_age_seconds=(raw_cache_max_age_seconds if raw_cache_max_age_seconds > 0 else None),
                    )
                if detail_cached:
                    detail_url, detail_html = detail_cached
                else:
                    detail_url, detail_html = _fetch_path_logged(client, path=card.page_path, what="video", key=card.video_id)

                vparsed = parse_video_page(detail_html)
                if not vparsed.video_id:
                    return

                if store_raw:
                    san = sanitize_html(detail_html)
                    insert_raw_page(
                        db,
                        page_kind="video",
                        source_key=vparsed.video_id,
                        url=detail_url,
                        html_sanitized=san.html,
                        sha256=san.sha256,
                        redactions=san.redactions,
                    )

                ss_url = urljoin(detail_url, vparsed.screenshot_url) if vparsed.screenshot_url else None
                ss_urls = [urljoin(detail_url, u) for u in (vparsed.screenshot_urls or [])]
                update_video_detail(
                    db,
                    video_id=vparsed.video_id,
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
                    extract={"breadcrumbs": [asdict(b) for b in vparsed.breadcrumbs]},
                )

                # taxonomy from breadcrumb: try to map /videos/series-<id>.html items
                for bc in vparsed.breadcrumbs:
                    m = _RE_BREADCRUMB_SERIES.search(bc.item or "")
                    if m:
                        sid = m.group(1)
                        tid = get_taxonomy_node_id(db, node_type=NODE_TYPE_SERIES, source_key=sid)
                        if tid is None:
                            tid = upsert_taxonomy_node(
                                db,
                                node_type=NODE_TYPE_SERIES,
                                source_key=sid,
                                name=bc.name or f"series-{sid}",
                                url=bc.item,
                                parent_id=None,
                                item_count_hint=None,
                            )
                        upsert_video_taxonomy(db, video_id=vparsed.video_id, node_id=tid, source="breadcrumb")
                picked = _pick_specific_series_from_breadcrumbs(
                    current_series_id=series_id, breadcrumbs=vparsed.breadcrumbs
                )
                if picked:
                    sid, sname = picked
                    update_video_series(
                        db, video_id=vparsed.video_id, video_series_name=sname, video_series_source_key=sid
                    )

                if crawl_video_download:
                    download_path = f"/download/id-{vparsed.video_id}.html"
                    dl_cached = None
                    if store_raw and use_raw_cache:
                        dl_cached = _maybe_load_cached_html(
                            db=db,
                            client=client,
                            page_kind="download",
                            path=download_path,
                            max_age_seconds=(raw_cache_max_age_seconds if raw_cache_max_age_seconds > 0 else None),
                        )
                    if dl_cached:
                        dl_url, dl_html = dl_cached
                    else:
                        dl_url, dl_html = _fetch_path_logged(client, path=download_path, what="download", key=vparsed.video_id)

                    dl_parsed = parse_download_page(dl_html)
                    if store_raw:
                        san = sanitize_html(dl_html)
                        insert_raw_page(
                            db,
                            page_kind="download",
                            source_key=vparsed.video_id,
                            url=dl_url,
                            html_sanitized=san.html,
                            sha256=san.sha256,
                            redactions=san.redactions,
                        )
                    upsert_download_page(
                        db,
                        video_id=vparsed.video_id,
                        page_url=dl_url,
                        canonical_url=dl_parsed.canonical_url,
                        title=dl_parsed.title,
                        has_download_section=dl_parsed.has_download_section,
                        has_magnet_button=dl_parsed.has_magnet_button,
                        has_torrent_button=dl_parsed.has_torrent_button,
                        external_link_domains=dl_parsed.external_link_domains,
                        extract={
                            "breadcrumbs": [asdict(b) for b in dl_parsed.breadcrumbs],
                            "links": dl_parsed.links,
                        },
                    )
                    update_video_stream_links(
                        db,
                        video_id=vparsed.video_id,
                        m3u8_url=dl_parsed.m3u8_url,
                        poster_url=dl_parsed.poster_url,
                    )
                    update_video_download_links(
                        db,
                        video_id=vparsed.video_id,
                        magnet_uri=dl_parsed.magnet_uri,
                        torrent_url=(urljoin(dl_url, dl_parsed.torrent_url) if dl_parsed.torrent_url else None),
                    )

        futures = []
        executor = None
        if workers and workers > 1 and (crawl_video_detail or crawl_video_download):
            executor = ThreadPoolExecutor(max_workers=workers)

        try:
            base = page_url.split("/videos/")[0]
            for card in parsed.video_cards:
                page_url_video = base + card.page_path
                upsert_video_base(
                    db,
                    video_id=card.video_id,
                    title=card.title,
                    page_url=page_url_video,
                    canonical_url=None,
                    cover_url=card.cover_url,
                    video_series_name=parsed.series_name,
                    video_series_source_key=series_id,
                    video_tags=card.tags,
                )
                upsert_video_taxonomy(db, video_id=card.video_id, node_id=node_id, source="list_page")
                insert_list_observation(
                    db,
                    video_id=card.video_id,
                    node_id=node_id,
                    list_title=card.title,
                    list_cover_url=card.cover_url,
                    list_badge=card.badge,
                    list_duration_text=card.duration_text,
                    list_duration_seconds=card.duration_seconds,
                    list_comment_count=card.comment_count,
                    list_flags={"has_magnet_icon": card.has_magnet_icon},
                    page_url=page_url,
                )

                if executor:
                    futures.append(executor.submit(process_card, card))
                else:
                    process_card(card)

            if futures:
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as exc:  # noqa: BLE001
                        print(f"[warn] card processing failed: {exc}")
        finally:
            if executor:
                executor.shutdown(wait=True)

        pages_done += 1
        LOG.info(f"page done series_id={series_id} page={current} videos={len(parsed.video_cards)}")

        if parsed.last_page_number is not None and current >= parsed.last_page_number:
            break
        current += 1


def cmd_crawl_series(args: argparse.Namespace) -> int:
    """
    子命令：crawl-series

    功能：抓取单个 series 的列表分页，并按需抓详情/下载页。
    """

    cfg = load_config(args.config)
    db = Db(cfg.database_url)
    client = _make_client(cfg)
    workers = args.workers if args.workers is not None else cfg.workers
    with_detail = args.with_detail if args.with_detail is not None else cfg.crawl_with_detail
    with_download = args.with_download if args.with_download is not None else cfg.crawl_with_download
    store_raw = args.store_raw if args.store_raw is not None else cfg.crawl_store_raw
    use_raw_cache = args.use_raw_cache if args.use_raw_cache is not None else cfg.crawl_use_raw_cache
    raw_cache_max_age_seconds = (
        args.raw_cache_max_age_seconds
        if args.raw_cache_max_age_seconds is not None
        else cfg.crawl_raw_cache_max_age_seconds
    )
    all_pages = args.all_pages if args.all_pages is not None else cfg.crawl_all_pages

    if all_pages:
        args.end_page = None
        args.max_pages = None

    node_id = get_taxonomy_node_id(db, node_type=NODE_TYPE_SERIES, source_key=args.series_id)
    if node_id is None:
        # create minimal node; name will be filled by first page
        node_id = upsert_taxonomy_node(
            db,
            node_type=NODE_TYPE_SERIES,
            source_key=args.series_id,
            name=f"series-{args.series_id}",
            url=None,
            parent_id=None,
            item_count_hint=None,
        )

    _crawl_series_pages(
        db=db,
        client=client,
        series_id=args.series_id,
        node_id=node_id,
        start_page=args.start_page,
        end_page=args.end_page,
        max_pages=args.max_pages,
        crawl_video_detail=bool(with_detail),
        crawl_video_download=bool(with_download),
        store_raw=bool(store_raw),
        use_raw_cache=bool(use_raw_cache),
        raw_cache_max_age_seconds=int(raw_cache_max_age_seconds),
        workers=max(1, int(workers)),
    )
    return 0


def cmd_crawl_board(args: argparse.Namespace) -> int:
    """
    子命令：crawl-board

    功能：
    - 抓取板块（父级）series 第 1 页，发现子分类
    - （可选）爬取板块自身分页
    - 逐个爬取子分类 series

    注意：
    - 需要传入板块父级的 series_id（页面 h1 与侧边栏 parent 一致）
    """

    cfg = load_config(args.config)
    db = Db(cfg.database_url)
    client = _make_client(cfg)
    workers = args.workers if args.workers is not None else cfg.workers
    with_detail = args.with_detail if args.with_detail is not None else cfg.crawl_with_detail
    with_download = args.with_download if args.with_download is not None else cfg.crawl_with_download
    store_raw = args.store_raw if args.store_raw is not None else cfg.crawl_store_raw
    use_raw_cache = args.use_raw_cache if args.use_raw_cache is not None else cfg.crawl_use_raw_cache
    raw_cache_max_age_seconds = (
        args.raw_cache_max_age_seconds
        if args.raw_cache_max_age_seconds is not None
        else cfg.crawl_raw_cache_max_age_seconds
    )
    all_pages = args.all_pages if args.all_pages is not None else cfg.crawl_all_pages

    if all_pages:
        args.board_end_page = None
        args.board_max_pages = None
        args.child_end_page = None
        args.child_max_pages = None

    board_first_path = f"/videos/series-{args.series_id}/1.html"
    board_res = client.fetch_path(board_first_path)
    board_parsed = parse_series_page(board_first_path, board_res.text)
    board_name = board_parsed.series_name or f"series-{args.series_id}"

    if board_parsed.board_name and board_parsed.series_name and board_parsed.board_name != board_parsed.series_name:
        raise RuntimeError(
            f"series_id={args.series_id} 看起来是子分类页：h1={board_parsed.series_name!r}，影片分类父级={board_parsed.board_name!r}；"
            "crawl-board 需要传入板块（父级）对应的 series_id"
        )

    board_node_id = upsert_taxonomy_node(
        db,
        node_type=NODE_TYPE_SERIES,
        source_key=args.series_id,
        name=board_name,
        url=board_res.url,
        parent_id=None,
        item_count_hint=None,
    )

    # upsert children, then crawl
    children = board_parsed.children
    print(f"board: {args.series_id} name={board_name} children={len(children)}")

    if args.board_max_pages is None or args.board_max_pages > 0:
        _crawl_series_pages(
            db=db,
            client=client,
            series_id=args.series_id,
            node_id=board_node_id,
            start_page=1,
            end_page=args.board_end_page,
            max_pages=args.board_max_pages,
            crawl_video_detail=bool(with_detail),
            crawl_video_download=bool(with_download),
            store_raw=bool(store_raw),
            use_raw_cache=bool(use_raw_cache),
            raw_cache_max_age_seconds=int(raw_cache_max_age_seconds),
            workers=max(1, int(workers)),
        )

    for child in children:
        child_node_id = upsert_taxonomy_node(
            db,
            node_type=NODE_TYPE_SERIES,
            source_key=child.series_id,
            name=child.name,
            url=(board_res.url.split('/videos/')[0] + child.url_path) if child.url_path.startswith('/') else child.url_path,
            parent_id=board_node_id,
            item_count_hint=child.item_count_hint,
        )
        _crawl_series_pages(
            db=db,
            client=client,
            series_id=child.series_id,
            node_id=child_node_id,
            start_page=1,
            end_page=args.child_end_page,
            max_pages=args.child_max_pages,
            crawl_video_detail=bool(with_detail),
            crawl_video_download=bool(with_download),
            store_raw=bool(store_raw),
            use_raw_cache=bool(use_raw_cache),
            raw_cache_max_age_seconds=int(raw_cache_max_age_seconds),
            workers=max(1, int(workers)),
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    """
    构建 argparse 命令行解析器。

    Returns:
        ArgumentParser
    """

    p = argparse.ArgumentParser(prog="xchina_crawler")
    p.add_argument("--config", default="config.yaml", help="YAML 配置文件路径（默认：config.yaml）")
    p.add_argument("--log-level", help="日志级别（DEBUG/INFO/WARNING/ERROR；默认取配置 logging.level）")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-db", help="create tables")
    p_init.add_argument("--schema", default="schema.sql")
    p_init.set_defaults(func=cmd_init_db)

    p_discover = sub.add_parser("discover-series", help="discover child series from a board series page 1")
    p_discover.add_argument("--series-id", required=True)
    p_discover.set_defaults(func=cmd_discover_series)

    p_crawl_series = sub.add_parser("crawl-series", help="crawl a single series (list pages; optional details)")
    p_crawl_series.add_argument("--series-id", required=True)
    p_crawl_series.add_argument("--start-page", type=int, default=1)
    p_crawl_series.add_argument("--end-page", type=int)
    p_crawl_series.add_argument("--max-pages", type=int)
    p_crawl_series.add_argument(
        "--all-pages",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="一直翻页直到 last_page（会忽略 --end-page/--max-pages；默认取配置 crawl.all_pages）",
    )
    p_crawl_series.add_argument(
        "--workers",
        type=int,
        help="并发线程数（用于详情/下载页抓取；默认取配置 crawl.workers）",
    )
    p_crawl_series.add_argument(
        "--with-detail",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="抓取详情页补全 videos（默认取配置 crawl.with_detail）",
    )
    p_crawl_series.add_argument(
        "--with-download",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="抓取下载页（默认取配置 crawl.with_download）",
    )
    p_crawl_series.add_argument(
        "--store-raw",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="存储 HTML 快照（会对直链进行脱敏；默认取配置 crawl.store_raw）",
    )
    p_crawl_series.add_argument(
        "--use-raw-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="优先复用 raw_pages 里的脱敏 HTML（默认取配置 crawl.use_raw_cache）",
    )
    p_crawl_series.add_argument(
        "--raw-cache-max-age-seconds",
        type=int,
        help="raw cache 最大可复用秒数（<=0 表示不限制；默认取配置 crawl.raw_cache_max_age_seconds）",
    )
    p_crawl_series.set_defaults(func=cmd_crawl_series)

    p_crawl_board = sub.add_parser("crawl-board", help="discover child series then crawl each child series")
    p_crawl_board.add_argument("--series-id", required=True)
    p_crawl_board.add_argument("--board-end-page", type=int)
    p_crawl_board.add_argument("--board-max-pages", type=int, default=1)
    p_crawl_board.add_argument("--child-end-page", type=int)
    p_crawl_board.add_argument("--child-max-pages", type=int, default=2)
    p_crawl_board.add_argument(
        "--all-pages",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="板块/子分类都一直翻页直到 last_page（默认取配置 crawl.all_pages）",
    )
    p_crawl_board.add_argument(
        "--workers",
        type=int,
        help="并发线程数（用于详情/下载页抓取；默认取配置 crawl.workers）",
    )
    p_crawl_board.add_argument(
        "--with-detail",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="抓取详情页补全 videos（默认取配置 crawl.with_detail）",
    )
    p_crawl_board.add_argument(
        "--with-download",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="抓取下载页（默认取配置 crawl.with_download）",
    )
    p_crawl_board.add_argument(
        "--store-raw",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="存储 HTML 快照（会对直链进行脱敏；默认取配置 crawl.store_raw）",
    )
    p_crawl_board.add_argument(
        "--use-raw-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="优先复用 raw_pages 里的脱敏 HTML（默认取配置 crawl.use_raw_cache）",
    )
    p_crawl_board.add_argument(
        "--raw-cache-max-age-seconds",
        type=int,
        help="raw cache 最大可复用秒数（<=0 表示不限制；默认取配置 crawl.raw_cache_max_age_seconds）",
    )
    p_crawl_board.set_defaults(func=cmd_crawl_board)

    return p


def main(argv: list[str] | None = None) -> int:
    """
    CLI 主入口。

    Args:
        argv: 参数列表（None 表示使用 sys.argv）

    Returns:
        进程退出码（0 表示成功）
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    _setup_logging(str(args.log_level or cfg.log_level))
    return int(args.func(args))
