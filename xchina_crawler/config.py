from __future__ import annotations

"""
配置加载模块（YAML）。

本项目不使用环境变量作为主要配置来源，而是通过 YAML 文件集中管理：
- 数据库 DSN
- 抓取域名列表（按顺序 failover）
- HTTP 相关参数（UA、Referer、超时、重试、限速）
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    """
    爬虫运行配置。

    Attributes:
        database_url: Postgres DSN，例如 `postgresql://user:pass@host:5432/dbname`
        base_urls: 抓取 HTML 的域名列表（会按顺序尝试，失败自动切换）
        user_agent: HTTP User-Agent
        referer: HTTP Referer
        proxy_url: 代理 URL（可选；同时作用于 http/https；例如 socks5h://127.0.0.1:7890）
        proxy_http: HTTP 代理 URL（可选；优先级高于 proxy_url）
        proxy_https: HTTPS 代理 URL（可选；优先级高于 proxy_url）
        trust_env: requests 是否读取环境变量代理（HTTP_PROXY/HTTPS_PROXY 等；默认 false）
        timeout_seconds: 单次请求超时时间（秒）
        retries: 单个 URL 的重试次数
        sleep_seconds: 每次请求后的固定 sleep（秒），用于降速/减少触发风控
        workers: 并发线程数（用于抓取详情/下载页等 I/O 密集步骤）
        log_level: 日志级别（DEBUG/INFO/WARNING/ERROR）
        crawl_with_detail: crawl-* 默认是否抓详情页
        crawl_with_download: crawl-* 默认是否抓下载页
        crawl_store_raw: crawl-* 默认是否存 raw_pages
        crawl_use_raw_cache: crawl-* 默认是否优先复用 raw_pages（URL 命中则不再发请求）
        crawl_raw_cache_max_age_seconds: raw cache 最大复用秒数（<=0 不限制）
        crawl_all_pages: crawl-* 默认是否一直翻页直到 last_page
        download_out_dir: Emby 下载输出根目录（可为空；也可用命令行 --out 覆盖）
        download_limit: 每次运行最多处理多少条
        download_refresh_video_page: 下载前是否刷新 /video 页面（更新 m3u8/图片等）
        download_move_to_complete: 完成后是否移动到 complete 目录
        download_work_subdir: 工作目录子目录名（默认：_working）
        download_complete_subdir: 完成目录子目录名（默认：complete）
        download_include_downloaded: 是否也处理已标记 downloaded 的记录（默认：false）
        download_workers: 下载并发数（按视频维度；默认 1）
        download_show_progress: 是否显示下载进度（默认 true）
        download_progress_interval_seconds: 进度输出最小间隔秒（默认 2.0）
        download_dynamic_progress: 是否用同一屏动态刷新进度（默认 true；非 TTY 自动降级为普通输出）
        download_name_max: 进度显示名称最大长度（默认 28）
        download_engine: 下载引擎（'ffmpeg' | 'aria2'；默认 'ffmpeg'）
        download_concurrent_segments: aria2 单视频分片并发数（默认 16）
        download_aria2c_path: aria2c 可执行文件名或路径（默认 'aria2c'）
        download_max_missing_segments: aria2 模式允许缺失的 ts 分片数量（默认 0）
    """

    database_url: str
    base_urls: list[str]
    user_agent: str
    referer: str
    proxy_url: str | None
    proxy_http: str | None
    proxy_https: str | None
    trust_env: bool
    timeout_seconds: int
    retries: int
    sleep_seconds: float
    workers: int
    log_level: str
    crawl_with_detail: bool
    crawl_with_download: bool
    crawl_store_raw: bool
    crawl_use_raw_cache: bool
    crawl_raw_cache_max_age_seconds: int
    crawl_all_pages: bool
    download_out_dir: str | None
    download_limit: int
    download_refresh_video_page: bool
    download_move_to_complete: bool
    download_work_subdir: str
    download_complete_subdir: str
    download_include_downloaded: bool
    download_workers: int
    download_show_progress: bool
    download_progress_interval_seconds: float
    download_dynamic_progress: bool
    download_name_max: int
    download_engine: str
    download_concurrent_segments: int
    download_aria2c_path: str
    download_max_missing_segments: int


def _require_str(obj: dict[str, Any], key: str) -> str:
    """
    从 dict 中读取必填字符串配置项。

    Args:
        obj: YAML 根对象或其子对象（dict）
        key: 字段名

    Returns:
        去除首尾空白后的字符串

    Raises:
        RuntimeError: 字段缺失、类型错误或为空字符串
    """

    v = obj.get(key)
    if not isinstance(v, str) or not v.strip():
        raise RuntimeError(f"配置项缺失或类型错误：{key}")
    return v.strip()


def _optional_str(obj: dict[str, Any], key: str, default: str) -> str:
    """
    从 dict 中读取可选字符串配置项。

    Args:
        obj: YAML 对象（dict）
        key: 字段名
        default: 缺省值

    Returns:
        字符串（为空时回退 default）

    Raises:
        RuntimeError: 字段类型错误
    """

    v = obj.get(key)
    if v is None:
        return default
    if not isinstance(v, str):
        raise RuntimeError(f"配置项类型错误：{key} 需要 string")
    return v.strip() or default


def _optional_str_or_none(obj: dict[str, Any], key: str) -> str | None:
    """
    从 dict 中读取可选字符串配置项（允许空字符串 -> None）。

    Args:
        obj: YAML 对象（dict）
        key: 字段名

    Returns:
        str | None

    Raises:
        RuntimeError: 字段类型错误
    """

    v = obj.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise RuntimeError(f"配置项类型错误：{key} 需要 string")
    vv = v.strip()
    return vv or None


def _optional_int(obj: dict[str, Any], key: str, default: int) -> int:
    """
    从 dict 中读取可选 int 配置项。

    Args:
        obj: YAML 对象（dict）
        key: 字段名
        default: 缺省值

    Returns:
        int

    Raises:
        RuntimeError: 字段类型错误
    """

    v = obj.get(key)
    if v is None:
        return default
    if not isinstance(v, int):
        raise RuntimeError(f"配置项类型错误：{key} 需要 int")
    return v


def _optional_float(obj: dict[str, Any], key: str, default: float) -> float:
    """
    从 dict 中读取可选 float 配置项。

    允许 YAML 用整数写小数配置（例如 `sleep_seconds: 1`），会自动转为 float。

    Args:
        obj: YAML 对象（dict）
        key: 字段名
        default: 缺省值

    Returns:
        float

    Raises:
        RuntimeError: 字段类型错误
    """

    v = obj.get(key)
    if v is None:
        return default
    if isinstance(v, int):
        return float(v)
    if not isinstance(v, float):
        raise RuntimeError(f"配置项类型错误：{key} 需要 float")
    return v


def _optional_bool(obj: dict[str, Any], key: str, default: bool) -> bool:
    v = obj.get(key)
    if v is None:
        return default
    if not isinstance(v, bool):
        raise RuntimeError(f"配置项类型错误：{key} 需要 bool")
    return v


def load_config(path: str) -> Config:
    """
    从 YAML 文件读取并校验配置，生成 `Config`。

    Args:
        path: 配置文件路径（例如 `config.yaml`）

    Returns:
        Config: 解析后的配置对象

    Raises:
        RuntimeError: 配置文件格式/类型错误或缺少必填项
        FileNotFoundError: 配置文件不存在
    """

    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise RuntimeError("配置文件格式错误：根节点必须是 YAML map/object")

    database_url = _require_str(data, "database_url")

    base_urls_v = data.get("base_urls")
    if not isinstance(base_urls_v, list) or not base_urls_v:
        raise RuntimeError("配置项缺失或类型错误：base_urls 需要 non-empty list")
    base_urls: list[str] = []
    for u in base_urls_v:
        if not isinstance(u, str) or not u.strip():
            raise RuntimeError("配置项类型错误：base_urls 每个元素需要 string")
        base_urls.append(u.strip().rstrip("/"))

    http = data.get("http") or {}
    if not isinstance(http, dict):
        raise RuntimeError("配置项类型错误：http 需要 map/object")

    logging_cfg = data.get("logging") or {}
    if not isinstance(logging_cfg, dict):
        raise RuntimeError("配置项类型错误：logging 需要 map/object")

    crawl = data.get("crawl") or {}
    if not isinstance(crawl, dict):
        raise RuntimeError("配置项类型错误：crawl 需要 map/object")

    download = data.get("download") or {}
    if not isinstance(download, dict):
        raise RuntimeError("配置项类型错误：download 需要 map/object")

    user_agent = _optional_str(
        http,
        "user_agent",
        "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Mobile Safari/537.36",
    )
    referer = _optional_str(http, "referer", base_urls[0] + "/")
    proxy_url = _optional_str_or_none(http, "proxy_url")
    proxy_http = _optional_str_or_none(http, "proxy_http")
    proxy_https = _optional_str_or_none(http, "proxy_https")
    trust_env = _optional_bool(http, "trust_env", False)

    # backwards compatibility: allow workers under http.*, but prefer crawl.workers
    http_workers = max(1, _optional_int(http, "workers", 1))
    crawl_workers = max(1, _optional_int(crawl, "workers", http_workers))

    download_out_dir_raw = download.get("out_dir")
    download_out_dir: str | None
    if download_out_dir_raw is None:
        download_out_dir = None
    elif not isinstance(download_out_dir_raw, str):
        raise RuntimeError("配置项类型错误：download.out_dir 需要 string")
    else:
        download_out_dir = download_out_dir_raw.strip() or None

    return Config(
        database_url=database_url,
        base_urls=base_urls,
        user_agent=user_agent,
        referer=referer,
        proxy_url=proxy_url,
        proxy_http=proxy_http,
        proxy_https=proxy_https,
        trust_env=bool(trust_env),
        timeout_seconds=_optional_int(http, "timeout_seconds", 30),
        retries=_optional_int(http, "retries", 3),
        sleep_seconds=_optional_float(http, "sleep_seconds", 0.3),
        workers=crawl_workers,
        log_level=_optional_str(logging_cfg, "level", "INFO"),
        crawl_with_detail=_optional_bool(crawl, "with_detail", False),
        crawl_with_download=_optional_bool(crawl, "with_download", False),
        crawl_store_raw=_optional_bool(crawl, "store_raw", False),
        crawl_use_raw_cache=_optional_bool(crawl, "use_raw_cache", False),
        crawl_raw_cache_max_age_seconds=_optional_int(crawl, "raw_cache_max_age_seconds", 0),
        crawl_all_pages=_optional_bool(crawl, "all_pages", False),
        download_out_dir=download_out_dir,
        download_limit=max(1, _optional_int(download, "limit", 50)),
        download_refresh_video_page=_optional_bool(download, "refresh_video_page", True),
        download_move_to_complete=_optional_bool(download, "move_to_complete", True),
        download_work_subdir=_optional_str(download, "work_subdir", "_working"),
        download_complete_subdir=_optional_str(download, "complete_subdir", "complete"),
        download_include_downloaded=_optional_bool(download, "include_downloaded", False),
        download_workers=max(1, _optional_int(download, "workers", 1)),
        download_show_progress=_optional_bool(download, "show_progress", True),
        download_progress_interval_seconds=_optional_float(download, "progress_interval_seconds", 2.0),
        download_dynamic_progress=_optional_bool(download, "dynamic_progress", True),
        download_name_max=max(8, _optional_int(download, "name_max", 28)),
        download_engine=_optional_str(download, "engine", "ffmpeg").lower(),
        download_concurrent_segments=max(1, _optional_int(download, "concurrent_segments", 16)),
        download_aria2c_path=_optional_str(download, "aria2c_path", "aria2c"),
        download_max_missing_segments=max(0, _optional_int(download, "max_missing_segments", 0)),
    )
