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
    """

    database_url: str
    base_urls: list[str]
    user_agent: str
    referer: str
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

    user_agent = _optional_str(
        http,
        "user_agent",
        "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Mobile Safari/537.36",
    )
    referer = _optional_str(http, "referer", base_urls[0] + "/")

    # backwards compatibility: allow workers under http.*, but prefer crawl.workers
    http_workers = max(1, _optional_int(http, "workers", 1))
    crawl_workers = max(1, _optional_int(crawl, "workers", http_workers))

    return Config(
        database_url=database_url,
        base_urls=base_urls,
        user_agent=user_agent,
        referer=referer,
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
    )
