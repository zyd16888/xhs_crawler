from __future__ import annotations

"""
HTTP 请求封装（requests）。

职责：
- 统一 UA/Referer 等请求头
- 按 base_urls 顺序 failover
- 基础重试与简单退避
- 控制请求间隔（sleep）
"""

import time
from dataclasses import dataclass
import threading
from urllib.parse import urljoin

import random
import requests


@dataclass(frozen=True)
class FetchResult:
    """
    单次抓取结果。

    Attributes:
        url: 实际请求的最终 URL（包含 base_url 拼接后的绝对地址）
        status_code: HTTP 状态码
        text: 响应体文本（requests 基于响应头推断编码并解码）
    """

    url: str
    status_code: int
    text: str


class HttpClient:
    """
    简单的 HTTP 客户端（带 failover + 重试 + 限速）。

    注意：
    - 本项目主要抓取 HTML 文本；不处理二进制资源下载。
    - 为降低因 brotli 依赖导致的解码问题，默认只声明 `Accept-Encoding: gzip`。
    """

    def __init__(
        self,
        base_urls: list[str],
        *,
        user_agent: str,
        referer: str,
        proxies: dict[str, str] | None = None,
        trust_env: bool = False,
        timeout_seconds: int,
        retries: int,
        sleep_seconds: float,
    ) -> None:
        self._base_urls = base_urls
        self._timeout_seconds = timeout_seconds
        self._retries = retries
        self._sleep_seconds = sleep_seconds
        self._proxies = dict(proxies) if proxies else None
        self._trust_env = bool(trust_env)
        self._headers = {
            "User-Agent": user_agent,
            "Referer": referer,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            # 优先 gzip，避免某些环境缺少 brotli 依赖导致的解码问题。
            "Accept-Encoding": "gzip",
        }
        self._tls = threading.local()

    def _session(self) -> requests.Session:
        """
        获取当前线程绑定的 requests.Session。

        requests.Session 不保证线程安全；多线程场景需要每线程一个 session。
        """

        sess = getattr(self._tls, "session", None)
        if sess is None:
            sess = requests.Session()
            sess.headers.update(self._headers)
            sess.trust_env = self._trust_env
            if self._proxies:
                sess.proxies.update(self._proxies)
            self._tls.session = sess
        return sess

    def fetch_path(self, path: str) -> FetchResult:
        """
        抓取站内路径（相对路径）并返回文本结果。

        Args:
            path: 站内路径，例如 `/videos/series-xxx/1.html`

        Returns:
            FetchResult: 包含最终 URL、状态码与响应文本

        Raises:
            RuntimeError: 多域名 + 多次重试仍失败时抛出
        """

        last_exc: Exception | None = None
        last_url: str | None = None

        retryable_status_codes = {408, 425, 429, 500, 502, 503, 504}
        base_backoff_seconds = max(0.5, float(self._sleep_seconds))
        max_backoff_seconds = 8.0

        def parse_retry_after_seconds(value: str | None) -> float | None:
            if not value:
                return None
            v = value.strip()
            if not v:
                return None
            try:
                return max(0.0, float(int(v)))
            except Exception:
                return None

        for base_url in self._base_urls:
            url = urljoin(base_url + "/", path.lstrip("/"))
            last_url = url
            for attempt in range(1, self._retries + 1):
                try:
                    resp = self._session().get(url, timeout=self._timeout_seconds)
                    if resp.status_code >= 400:
                        retry_after = parse_retry_after_seconds(resp.headers.get("Retry-After"))
                        if resp.status_code in retryable_status_codes and attempt < self._retries:
                            # Retryable HTTP status; backoff with jitter.
                            delay = min(max_backoff_seconds, base_backoff_seconds * (2 ** (attempt - 1)))
                            jitter = random.uniform(0.0, delay * 0.2)
                            sleep_for = min(max_backoff_seconds, delay + jitter)
                            if retry_after is not None:
                                sleep_for = max(sleep_for, min(max_backoff_seconds, retry_after))
                            time.sleep(sleep_for)
                            continue
                    resp.raise_for_status()
                    time.sleep(self._sleep_seconds)
                    return FetchResult(url=url, status_code=resp.status_code, text=resp.text)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if isinstance(exc, requests.exceptions.HTTPError):
                        status_code = getattr(getattr(exc, "response", None), "status_code", None)
                        # Non-retryable HTTP errors: don't retry the same base_url repeatedly.
                        # Still allow failover to the next base_url.
                        if status_code is not None and int(status_code) not in retryable_status_codes:
                            break
                    if attempt < self._retries:
                        # 连接抖动/临时错误：指数退避 + 抖动（避免多线程齐刷刷重试）
                        delay = min(max_backoff_seconds, base_backoff_seconds * (2 ** (attempt - 1)))
                        jitter = random.uniform(0.0, delay * 0.2)
                        time.sleep(min(max_backoff_seconds, delay + jitter))
                    continue

        raise RuntimeError(f"抓取失败：{last_url}") from last_exc

    def candidate_urls(self, path: str) -> list[str]:
        """
        将站内相对路径展开成按 base_urls 顺序排列的候选绝对 URL 列表。

        用途：
        - raw_pages URL 快取命中检查（无需实际发请求即可判断可能的 URL）
        """

        return [urljoin(base_url + "/", path.lstrip("/")) for base_url in self._base_urls]
