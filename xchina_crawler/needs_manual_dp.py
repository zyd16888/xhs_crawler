from __future__ import annotations

"""
辅助脚本：用 DrissionPage 处理 needs_manual 目录中的资源下载（解决 CF 盾导致的图片 403 等问题）。

用法示例：
    python3 -m xchina_crawler.needs_manual_dp --config config.yaml --needs-manual-dir "/path/to/MediaRoot/needs_manual"

流程：
1) 读取 needs_manual 下每个影片目录的 `_NEEDS_MANUAL.txt`
2) 根据策略用浏览器访问资源 URL（触发/刷新 CF 盾），并同步 cookies 到 session
3) 用内置 download() 下载缺失的图片到正确位置（失败时以“文件是否落地”为准）
4) 成功的条目会从 `_NEEDS_MANUAL.txt` 移除；全部成功则删除该文件
"""

import argparse
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from .config import load_config


_RE_ISSUE = re.compile(r"^(poster|screenshot(?P<idx>\d+))\s+下载失败\s+url=(?P<url>\S+?)\s+err=", re.I)
_RE_DIR_VIDEO_ID = re.compile(r"\[(?:xchina-)?(?P<id>[0-9a-f]{6,}|\d+)\]", re.I)


@dataclass(frozen=True)
class ManualIssue:
    raw_line: str
    kind: str  # 'poster' | 'screenshot'
    idx: int | None
    url: str
    dest_path: Path


def _url_ext(url: str) -> str:
    try:
        p = urlparse(url).path
    except Exception:
        p = ""
    _root, ext = os.path.splitext(p)
    ext = (ext or "").lower().strip()
    if not ext or len(ext) > 6:
        return ".jpg"
    return ext


def _parse_needs_manual_file(movie_dir: Path, p: Path) -> list[ManualIssue]:
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[ManualIssue] = []
    for line in lines:
        m = _RE_ISSUE.search(line.strip())
        if not m:
            continue
        url = m.group("url").strip()
        kind = m.group(1).lower()
        idx_s = m.group("idx")
        idx = int(idx_s) if idx_s else None

        ext = _url_ext(url)
        if kind == "poster":
            dest = movie_dir / f"poster{ext}"
        else:
            n = idx or 1
            dest = movie_dir / "extrafanart" / f"fanart{n}{ext}"
        out.append(ManualIssue(raw_line=line, kind=("poster" if kind == "poster" else "screenshot"), idx=idx, url=url, dest_path=dest))
    return out


def _copy_poster_variants(poster_path: Path, *, force: bool) -> None:
    if not poster_path.exists():
        return
    ext = poster_path.suffix
    for name in (f"fanart{ext}", f"thumb{ext}"):
        dst = poster_path.parent / name
        if force or not dst.exists():
            shutil.copy2(poster_path, dst)


def _ensure_dest_exists(dest_path: Path) -> Path:
    """
    DrissionPage 的 download() 有些版本会对 suffix 额外补点，导致目标文件名出现双点（例如 poster..webp）。
    这里做一次兜底：如果发现 dest 不存在，但同目录下存在 “多一个点”的同名文件，则改名回标准文件名。
    """

    if dest_path.exists():
        return dest_path

    # poster.webp -> poster..webp
    if dest_path.suffix:
        alt = dest_path.with_name(dest_path.stem + "." + dest_path.suffix)
        if alt.exists():
            alt.replace(dest_path)
            return dest_path

    return dest_path


def _iter_issue_files(needs_manual_dir: Path) -> Iterable[Path]:
    yield from needs_manual_dir.rglob("_NEEDS_MANUAL.txt")


def _infer_referer(cfg, movie_dir: Path) -> str:
    """
    给 upload.* 这类静态资源一个更“像浏览器”的 Referer。

    经验上，部分站点对图片直链有防盗链/风控：
    - 仅设置根 Referer（例如 https://xchina.co/）可能仍返回 403
    - 使用实际的视频页 Referer（/video/<id>）命中率更高
    """

    base = ""
    try:
        base_urls = getattr(cfg, "base_urls", None) or []
        if base_urls:
            base = str(base_urls[0] or "").strip()
    except Exception:
        base = ""

    if not base:
        base = str(getattr(cfg, "referer", "") or "").strip()
    base = base.rstrip("/")
    if not base:
        return ""

    m = _RE_DIR_VIDEO_ID.search(movie_dir.name)
    if not m:
        return base + "/"
    vid = (m.group("id") or "").strip()
    if not vid:
        return base + "/"
    return f"{base}/video/{vid}"


def _default_download_headers(*, user_agent: str, referer: str) -> dict[str, str]:
    h: dict[str, str] = {
        "User-Agent": user_agent,
        # 尽量模拟图片请求
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    if referer:
        h["Referer"] = referer
    return h


def _browser_visit_and_sync(page, url: str, *, new_tab: bool, wait_seconds: float) -> None:
    """
    用浏览器模式访问一次 URL（触发/刷新 CF 盾、更新 cookie），然后把 cookie 同步到 session。

    说明：
    - 某些情况下第一次浏览器访问会自动过盾，但后续 session 仍会 403
    - 因此可以选择：每个资源都浏览器访问一次（all），或仅在 403 时访问（on-403）
    """

    try:
        page.change_mode("d", go=False)
    except Exception:
        try:
            page.change_mode("d")
        except Exception:
            pass

    visited = False
    if new_tab:
        # best-effort: 不同版本 DrissionPage 的 new_tab API 可能不同，失败则回退同标签访问
        try:
            nt = getattr(page, "new_tab", None)
            if callable(nt):
                try:
                    tab = nt(url)
                    visited = True
                except TypeError:
                    tab = nt()
                    try:
                        getattr(tab, "get")(url)  # type: ignore[attr-defined]
                        visited = True
                    except Exception:
                        visited = False
                except Exception:
                    visited = False
                else:
                    try:
                        getattr(tab, "close")()  # type: ignore[attr-defined]
                    except Exception:
                        pass
        except Exception:
            visited = False

    if not visited:
        try:
            page.get(url, show_errmsg=False)
        except Exception:
            pass

    try:
        w = float(wait_seconds)
    except Exception:
        w = 0.0
    if w > 0:
        time.sleep(w)

    _sync_to_session(page)


def _sync_to_session(page) -> None:
    try:
        page.cookies_to_session(copy_user_agent=True)
    except Exception:
        pass
    try:
        page.change_mode("s", go=False, copy_cookies=True)
    except Exception:
        try:
            page.change_mode("s")
        except Exception:
            pass
    try:
        page.cookies_to_session(copy_user_agent=True)
    except Exception:
        pass


def _browser_warmup_and_prompt(page, url: str, *, wait_seconds: float, sync_session: bool) -> None:
    """
    首次 warmup：打开一个资源 URL，让用户有机会处理/等待 CF 盾，然后按回车继续。
    """

    try:
        page.change_mode("d", go=False)
    except Exception:
        try:
            page.change_mode("d")
        except Exception:
            pass

    try:
        page.get(url, show_errmsg=False)
    except Exception:
        pass

    try:
        w = float(wait_seconds)
    except Exception:
        w = 0.0
    if w > 0:
        time.sleep(w)

    print("如果出现 Cloudflare 验证，请在浏览器里手动点击/完成验证；或等待其自动跳过。完成后回到终端按回车继续。")
    input("继续（回车）> ")
    if sync_session:
        _sync_to_session(page)


def _download_one(
    page,
    *,
    url: str,
    dest_path: Path,
    overwrite: bool,
    headers: dict[str, str],
    timeout_seconds: float,
) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    rename = dest_path.stem
    # DrissionPage expects suffix without leading dot; otherwise it may generate double dots (poster..webp).
    suffix = dest_path.suffix.lstrip(".")
    try:
        page.download(
            url,
            save_path=str(dest_path.parent),
            rename=rename,
            suffix=suffix,
            file_exists=("overwrite" if overwrite else "skip"),
            show_msg=True,
            headers=headers,
            timeout=timeout_seconds,
        )
    except Exception:
        return False

    _ensure_dest_exists(dest_path)
    try:
        return dest_path.exists() and dest_path.stat().st_size > 0
    except Exception:
        return dest_path.exists()


def _download_via_browser_response(
    tab,
    *,
    url: str,
    dest_path: Path,
    overwrite: bool,
    wait_seconds: float,
) -> bool:
    """
    用“浏览器自身的网络栈”获取响应并落盘（绕开 requests/DownloadKit 的指纹/403 问题）。

    实现方式：使用 DrissionPage 的网络监听器抓取数据包，直接读取 response.body（bytes）并落盘。
    """

    if dest_path.exists() and not overwrite:
        return True
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        listener = getattr(tab, "listen", None)
        if listener is None:
            print("[browser-resp] 当前 DrissionPage 版本不支持 listen 监听器，无法用浏览器响应落盘。")
            return False

        try:
            listener.start(url)
        except Exception:
            # start() 参数不兼容时退化为“监听所有”，再在结果里匹配
            try:
                listener.start(True)
            except Exception as exc:  # noqa: BLE001
                print(f"[browser-resp] 启动监听失败：{exc}")
                return False

        try:
            getattr(tab, "get")(url, show_errmsg=False)  # type: ignore[misc]
        except TypeError:
            try:
                getattr(tab, "get")(url)  # type: ignore[misc]
            except Exception:
                pass
        except Exception:
            pass

        # 给 CF 自动跳过/重定向留一点时间
        try:
            w = float(wait_seconds)
        except Exception:
            w = 0.0
        if w > 0:
            time.sleep(w)

        try:
            timeout = max(10.0, float(wait_seconds) + 20.0)
        except Exception:
            timeout = 30.0
        try:
            pkt = listener.wait(timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"[browser-resp] 等待数据包失败：{exc}")
            pkt = False

        if not pkt:
            print(f"[browser-resp] 获取失败（未捕获到数据包）：{url}")
            return False

        # 如果监听所有，可能拿到的不是目标 URL，做一次兜底匹配
        if getattr(pkt, "url", "") != url:
            try:
                if hasattr(listener, "steps"):
                    for more in listener.steps(count=10, timeout=1.0):
                        if getattr(more, "url", "") == url:
                            pkt = more
                            break
            except Exception:
                pass

        resp = getattr(pkt, "response", None)
        status = getattr(resp, "status", None) if resp is not None else None
        body = getattr(resp, "body", None) if resp is not None else None
        headers = getattr(resp, "headers", None) if resp is not None else None
        if status is not None and int(status) >= 400:
            print(f"[browser-resp] 获取失败：status={status} url={url}")
            return False
        if not isinstance(body, (bytes, bytearray)):
            print(f"[browser-resp] 获取失败：body 类型异常 {type(body).__name__} url={url}")
            return False
        data = bytes(body)
        if not data:
            print(f"[browser-resp] 获取失败：空 body url={url}")
            return False
        # 兜底：若拿到的是 HTML（常见 CF/跳转页），直接视为失败
        ctype = ""
        try:
            if isinstance(headers, dict):
                ctype = str(headers.get("content-type") or headers.get("Content-Type") or "")
        except Exception:
            ctype = ""
        if ("text/html" in ctype.lower()) or data[:20].lstrip().lower().startswith(b"<"):
            print(f"[browser-resp] 获取失败（疑似 HTML/盾页）：url={url}")
            return False

    finally:
        # best-effort stop listener
        try:
            if "listener" in locals() and listener is not None:
                listener.stop()
        except Exception:
            pass

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        with tmp_path.open("wb") as f:
            f.write(data)
        tmp_path.replace(dest_path)
        _ensure_dest_exists(dest_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[browser-resp] 落盘失败：{url} -> {dest_path} err={exc}")
        try:
            if tmp_path.exists() and not dest_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return False

    try:
        return dest_path.exists() and dest_path.stat().st_size > 0
    except Exception:
        return dest_path.exists()


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="xchina_needs_manual_dp")
    ap.add_argument("--config", default="config.yaml", help="配置文件（用于 UA/Referer/代理等；默认 config.yaml）")
    ap.add_argument("--needs-manual-dir", required=True, help="needs_manual 目录路径（即包含各影片目录的那个）")
    ap.add_argument("--dry-run", action="store_true", help="只列出要处理的条目，不实际下载")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已有文件")
    g_warmup = ap.add_mutually_exclusive_group()
    g_warmup.add_argument(
        "--warmup",
        dest="warmup",
        action="store_true",
        default=True,
        help="启动时先打开 1 个资源 URL，手动/自动过一次 CF 盾后回车继续（默认）",
    )
    g_warmup.add_argument("--no-warmup", dest="warmup", action="store_false", help="不进行首次 warmup 交互")
    ap.add_argument(
        "--browser-visit",
        choices=["none", "all", "on-403"],
        default="on-403",
        help="浏览器访问策略：none=不访问；all=每个资源先用浏览器访问；on-403=仅当下载失败（常见 403）时用浏览器访问再重试（默认）",
    )
    ap.add_argument("--browser-new-tab", action="store_true", help="浏览器访问时尽量用新标签页（best-effort；失败会回退同标签）")
    ap.add_argument("--browser-wait-seconds", type=float, default=2.0, help="每次浏览器访问后等待秒数（给 CF 自动跳过留时间；默认 2.0）")
    ap.add_argument(
        "--download-via",
        choices=["auto", "session", "browser"],
        default="auto",
        help="下载方式：auto=先 session(download) 失败则用浏览器响应落盘；session=仅用 download；browser=仅用浏览器响应（默认 auto）",
    )
    ap.add_argument("--tabs", type=int, default=1, help="并发标签页数量（仅 download-via=browser 或 auto->browser 时有效；默认 1）")
    ap.add_argument("--retry-times", type=int, default=2, help="下载失败后最多重试次数（配合 browser-visit；默认 2）")
    ap.add_argument(
        "--move-to-complete",
        action="store_true",
        help="当某影片目录全部问题修复后，将其从 needs_manual 移动回 complete（目标目录从配置 download.complete_subdir 推导）",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    needs_manual_dir = Path(str(args.needs_manual_dir)).expanduser().resolve()
    if not needs_manual_dir.exists() or not needs_manual_dir.is_dir():
        raise RuntimeError(f"needs_manual_dir 不存在或不是目录：{needs_manual_dir}")

    issue_files = sorted(_iter_issue_files(needs_manual_dir))
    if not issue_files:
        print("未找到任何 _NEEDS_MANUAL.txt")
        return 0

    # Collect issues first (and pick a warmup URL).
    all_issues: list[tuple[Path, list[ManualIssue]]] = []
    warmup_url: str | None = None
    for issue_file in issue_files:
        movie_dir = issue_file.parent
        issues = _parse_needs_manual_file(movie_dir, issue_file)
        if not issues:
            continue
        all_issues.append((issue_file, issues))
        if warmup_url is None:
            warmup_url = issues[0].url

    if not all_issues:
        print("未解析到任何可处理的条目（_NEEDS_MANUAL.txt 可能为空或格式不匹配）")
        return 0

    if args.dry_run:
        for issue_file, issues in all_issues:
            print(f"== {issue_file}")
            for it in issues:
                print(f"- {it.kind}{'' if it.idx is None else it.idx}: {it.url} -> {it.dest_path}")
        return 0

    try:
        from DrissionPage import ChromiumOptions, SessionOptions, WebPage  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("缺少依赖 DrissionPage：请先安装（例如 pip install DrissionPage）") from exc

    # Build proxy config (best-effort, aligned with project config semantics).
    proxy_http = (cfg.proxy_http or cfg.proxy_url or "").strip() or None
    proxy_https = (cfg.proxy_https or cfg.proxy_url or "").strip() or None
    proxy_browser = (cfg.proxy_url or proxy_http or proxy_https or "").strip() or None

    co = ChromiumOptions(read_file=False)
    if proxy_browser:
        co.set_proxy(str(proxy_browser))
    # prefer visible browser for manual CF click
    co.headless(False)
    # keep UA aligned with downloader
    if getattr(co, "set_user_agent", None):
        try:
            co.set_user_agent(cfg.user_agent)
        except Exception:
            pass

    so = SessionOptions(read_file=False)
    so.set_headers({"User-Agent": cfg.user_agent, "Referer": cfg.referer})
    if proxy_http:
        so.set_proxies(str(proxy_http), str(proxy_https or proxy_http))
    if getattr(so, "trust_env", None) is not None:
        try:
            so.trust_env = bool(cfg.trust_env)
        except Exception:
            pass

    page = WebPage(mode="d", chromium_options=co, session_or_options=so, timeout=float(cfg.timeout_seconds))
    try:
        tabs = max(1, int(args.tabs) if args.tabs is not None else 1)
        download_via = str(args.download_via or "auto").strip()
        if tabs > 1:
            if download_via == "session":
                print("[提示] --download-via=session 不支持多标签并发，已忽略 --tabs（降为 1）。")
                tabs = 1
            elif download_via == "auto":
                print("[提示] --download-via=auto 且 --tabs>1 时，将直接使用浏览器响应落盘（等价于 --download-via=browser）。")
                download_via = "browser"

        sync_session = (download_via in ("session", "auto")) and tabs <= 1

        if args.warmup and warmup_url:
            print(f"将打开浏览器访问（warmup）：{warmup_url}")
            _browser_warmup_and_prompt(
                page,
                warmup_url,
                wait_seconds=float(args.browser_wait_seconds),
                sync_session=bool(sync_session),
            )
        else:
            if sync_session:
                _sync_to_session(page)

        # Slightly higher retry tolerance in session mode.
        try:
            page.set.retry_times(5)
            page.set.retry_interval(2)
        except Exception:
            pass

        out_root = needs_manual_dir.parent
        complete_subdir = (cfg.download_complete_subdir or "complete").strip() or "complete"
        complete_dir = out_root / complete_subdir

        use_parallel_tabs = (download_via == "browser") and tabs > 1
        tab_pool = None
        if use_parallel_tabs:
            try:
                from queue import Queue
                from concurrent.futures import ThreadPoolExecutor, as_completed
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("并发下载需要 queue/concurrent.futures（标准库）。") from exc

            if args.browser_new_tab:
                print("[提示] 已启用 --tabs，多标签并发场景下会复用标签页进行下载；--browser-new-tab 将被忽略。")

            tab_pool = Queue()
            tabs_list = [page]
            for _ in range(tabs - 1):
                try:
                    tabs_list.append(page.new_tab())
                except Exception as exc:  # noqa: BLE001
                    print(f"[警告] 创建新标签页失败，将降级并发：{exc}")
                    break
            for t in tabs_list:
                tab_pool.put(t)

            def _download_issue_in_pool(it: ManualIssue) -> bool:
                tab = tab_pool.get()
                try:
                    attempts = max(1, int(args.retry_times) if args.retry_times is not None else 1)
                    for attempt in range(attempts):
                        if attempt > 0:
                            print(f"[重试] 第 {attempt + 1}/{attempts} 次：{it.url}")
                        ok = _download_via_browser_response(
                            tab,
                            url=it.url,
                            dest_path=it.dest_path,
                            overwrite=bool(args.overwrite),
                            wait_seconds=float(args.browser_wait_seconds),
                        )
                        if ok:
                            return True
                    return False
                except Exception as exc:  # noqa: BLE001
                    print(f"[失败] {it.url} -> {it.dest_path} err={exc}")
                    return False
                finally:
                    tab_pool.put(tab)

        for issue_file, issues in all_issues:
            movie_dir = issue_file.parent
            remaining_lines = issue_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            keep_lines: list[str] = []

            # Build a map of raw_line -> issue for filtering.
            issue_by_line = {it.raw_line: it for it in issues}

            if use_parallel_tabs and tab_pool is not None:
                # 并发模式：只走浏览器响应落盘
                todo: list[ManualIssue] = []
                for line in remaining_lines:
                    it = issue_by_line.get(line)
                    if not it:
                        continue
                    if it.dest_path.exists() and not args.overwrite:
                        continue
                    todo.append(it)

                results: dict[str, bool] = {}
                if todo:
                    max_workers = min(tabs, len(todo))
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        fut_map = {ex.submit(_download_issue_in_pool, it): it for it in todo}
                        for fut in as_completed(fut_map):
                            it = fut_map[fut]
                            try:
                                results[it.raw_line] = bool(fut.result())
                            except Exception:
                                results[it.raw_line] = False

                # 回写 keep_lines + poster variants
                for line in remaining_lines:
                    it = issue_by_line.get(line)
                    if not it:
                        keep_lines.append(line)
                        continue
                    if it.dest_path.exists() and not args.overwrite:
                        continue
                    ok = bool(results.get(it.raw_line))
                    if not ok:
                        keep_lines.append(line)
                        continue
                    _ensure_dest_exists(it.dest_path)
                    if it.kind == "poster":
                        _copy_poster_variants(it.dest_path, force=bool(args.overwrite))
            else:
                # 顺序模式：支持 session/auto/browser
                for line in remaining_lines:
                    it = issue_by_line.get(line)
                    if not it:
                        keep_lines.append(line)
                        continue

                    if it.dest_path.exists() and not args.overwrite:
                        continue

                    referer = _infer_referer(cfg, movie_dir)
                    headers = _default_download_headers(user_agent=cfg.user_agent, referer=(referer or cfg.referer))

                    attempts = max(1, int(args.retry_times) if args.retry_times is not None else 1)
                    ok = False
                    for attempt in range(attempts):
                        if attempt > 0:
                            print(f"[重试] 第 {attempt + 1}/{attempts} 次：{it.url}")

                        if args.browser_visit == "all" and sync_session:
                            _browser_visit_and_sync(
                                page,
                                it.url,
                                new_tab=bool(args.browser_new_tab),
                                wait_seconds=float(args.browser_wait_seconds),
                            )

                        if download_via in ("session", "auto"):
                            ok = _download_one(
                                page,
                                url=it.url,
                                dest_path=it.dest_path,
                                overwrite=bool(args.overwrite),
                                headers=headers,
                                timeout_seconds=float(cfg.timeout_seconds),
                            )
                            if ok:
                                break

                        if download_via in ("browser", "auto"):
                            ok = _download_via_browser_response(
                                page,
                                url=it.url,
                                dest_path=it.dest_path,
                                overwrite=bool(args.overwrite),
                                wait_seconds=float(args.browser_wait_seconds),
                            )
                            if ok:
                                break

                    if not ok:
                        keep_lines.append(line)
                        continue

                    _ensure_dest_exists(it.dest_path)
                    if it.kind == "poster":
                        _copy_poster_variants(it.dest_path, force=bool(args.overwrite))

            if keep_lines:
                issue_file.write_text("\n".join(keep_lines) + "\n", encoding="utf-8")
            else:
                issue_file.unlink(missing_ok=True)

                if args.move_to_complete:
                    try:
                        rel = movie_dir.relative_to(needs_manual_dir)
                    except Exception:
                        rel = None
                    if rel is not None:
                        target = complete_dir / rel
                        if target.exists():
                            print(f"[跳过移动] 目标已存在：{target}")
                        else:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(movie_dir), str(target))
                            print(f"[已移动] {movie_dir} -> {target}")

        return 0
    finally:
        try:
            page.quit()
        except Exception:
            pass


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
