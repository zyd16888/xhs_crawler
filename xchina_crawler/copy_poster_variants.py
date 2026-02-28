from __future__ import annotations

"""
独立脚本：为 Emby/Kodi 补齐常见海报派生图。

把每个影片目录中的 poster.* 复制两份，生成：
- fanart.*
- thumb.*

用法：
    python -m xchina_crawler.copy_poster_variants --root "/path/to/MediaRoot"
"""

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Stats:
    scanned: int = 0
    created: int = 0
    skipped: int = 0
    missing_poster: int = 0


def _copy(src: Path, dst: Path, *, overwrite: bool, dry_run: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    if dry_run:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _pick_poster_in_dir(d: Path) -> Path | None:
    """
    优先选择常见格式，避免同目录存在多个 poster.* 时不稳定。
    """

    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = d / f"poster{ext}"
        if p.exists() and p.is_file():
            return p

    # 兜底：任意 poster.*（或无后缀 poster）
    p0 = d / "poster"
    if p0.exists() and p0.is_file():
        return p0

    hits = sorted(d.glob("poster.*"))
    for p in hits:
        if p.is_file():
            return p
    return None


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="xchina_copy_poster_variants")
    ap.add_argument("--root", required=True, help="媒体库根目录（递归扫描所有影片目录）")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的 fanart/thumb 文件")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要创建的文件，不实际写入")
    args = ap.parse_args(argv)

    root = Path(str(args.root)).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise RuntimeError(f"root 不存在或不是目录：{root}")

    stats = Stats()

    # 以目录为粒度：找到 poster.* 后补齐 fanart/thumb
    dirs = set()
    for p in root.rglob("poster.*"):
        if p.is_file():
            dirs.add(p.parent)
    p0 = root.rglob("poster")
    for p in p0:
        if p.is_file():
            dirs.add(p.parent)

    for d in sorted(dirs):
        stats.scanned += 1
        poster = _pick_poster_in_dir(d)
        if not poster:
            stats.missing_poster += 1
            continue

        ext = poster.suffix
        fanart = d / f"fanart{ext}"
        thumb = d / f"thumb{ext}"

        created_any = False
        for dst in (fanart, thumb):
            ok = _copy(poster, dst, overwrite=bool(args.overwrite), dry_run=bool(args.dry_run))
            if ok:
                created_any = True
                stats.created += 1
                print(f"[OK] {poster} -> {dst}")
            else:
                stats.skipped += 1

        if not created_any and args.dry_run:
            print(f"[SKIP] {d}（fanart/thumb 已存在）")

    print(
        f"完成：扫描目录 {stats.scanned}，创建 {stats.created}，跳过 {stats.skipped}，缺少 poster {stats.missing_poster}"
    )
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())

