# xchina 元数据爬虫

## 解析方式

HTML 解析使用 `BeautifulSoup (bs4)`。

## 存储内容（当前版本）

- 分类树：板块 series → 子分类 series（含数量提示）
- 视频元数据：`title/h1/canonical/page_url/cover_url/screenshot_url/upload_date/duration_seconds` 等
- videos 表补充字段：`video_series_name/video_series_source_key`、以及可选的 `m3u8_url/poster_url`
- 列表页观测：徽标（如“杏吧原版”）、时长文本、评论数、是否有磁铁图标等

说明：
- `raw_pages` 作为 HTML 快照仍会对直链类 URL 做替换脱敏（占位符：`M3U8_REDACTED` / `MAGNET_REDACTED` / `TORRENT_REDACTED`）。
- `videos.m3u8_url/poster_url` 若能从页面解析到，会落库；其中 m3u8 通常带短期签名参数，建议按需定期刷新。
- `raw_pages` 写入会基于 `url -> sha256` 做去重：同一页面内容未变化时，不会重复插入快照。

## 配置

复制 `config.example.yaml` 为 `config.yaml`，然后按需修改：

- `database_url`：Postgres DSN
- `base_urls`：HTML 页面抓取的域名列表（按顺序 failover）
- `http.*`：UA/Referer/超时/重试/限速等
- `http.proxy_*`：可选代理（支持 socks5/socks5h；建议 socks5h）
- `crawl.*`：抓取策略默认值（用配置尽量减少命令行参数）
- `download.*`：下载器默认参数（输出目录、是否移动到 complete 等）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 初始化数据库

```bash
python3 -m xchina_crawler --config config.yaml init-db
```

## 运行

1) 只发现子分类（从板块第 1 页的“影片分类”区块解析）：

```bash
python3 -m xchina_crawler --config config.yaml discover-series --series-id 63824a975d8ae
```

日志级别（可观测性）：默认取配置 `logging.level`；也可用 `--log-level DEBUG` 临时覆盖，输出更多“开始抓取/完成抓取/耗时/命中 raw cache”等信息：

```bash
python3 -m xchina_crawler --config config.yaml --log-level DEBUG crawl-series --series-id 5f904550b8fcc --max-pages 2 --with-detail
```

2) 跑板块：先抓板块本身，再抓每个子分类（默认子分类每个抓 2 页；加 `--with-detail` 会抓每条视频的详情页补全元数据；加 `--with-download` 会抓取下载页并写入 `video_download_pages` 的元数据）：

```bash
python3 -m xchina_crawler --config config.yaml crawl-board --series-id 63824a975d8ae --board-max-pages 1 --child-max-pages 2 --with-detail --with-download --store-raw
```

断点续跑：只想从板块列表第 N 页开始抓（仍会先抓第 1 页用于发现子分类），可用 `--board-start-page`：

```bash
python3 -m xchina_crawler --config config.yaml crawl-board --series-id 63824a975d8ae --board-start-page 10 --board-max-pages 2
```

也可以写到配置里（`config.yaml` 的 `crawl.board_start_page`），日常就不用每次传参数。

如果希望一直翻页直到板块/子分类全部抓完（直到解析到的 `last_page`），加 `--all-pages`：

```bash
python3 -m xchina_crawler --config config.yaml crawl-board --series-id 63824a975d8ae --all-pages --with-detail --with-download --store-raw
```

多线程抓取（并发详情/下载页；线程数可用 `--workers` 覆盖配置）：

```bash
python3 -m xchina_crawler --config config.yaml crawl-board --series-id 63824a975d8ae --all-pages --with-detail --with-download --workers 8
```

可选：复用 `raw_pages` 的快照以减少重复请求（URL 命中则不再发起抓取；可用 `--raw-cache-max-age-seconds` 限制复用时长）：

```bash
python3 -m xchina_crawler --config config.yaml crawl-board --series-id 63824a975d8ae --store-raw --use-raw-cache --raw-cache-max-age-seconds 86400
```

3) 只跑某个子分类 series：

```bash
python3 -m xchina_crawler --config config.yaml crawl-series --series-id 5f904550b8fcc --max-pages 2 --with-detail
```

## 下载到 Emby（电影库）

该工具会从数据库 `videos` 表读取待下载记录，按 Emby/Kodi 电影库结构落盘，并生成 `movie.nfo`：

- 默认会先刷新 `/video` 页面以获取最新 `m3u8_url/cover_url/screenshot_url`（避免过期），并回写数据库。
- 下载完成后会在数据库标记 `downloaded_at/download_status/downloaded_path`，用于避免重复下载（即使你后续把文件上传网盘并删除本地）。
- 默认会先下载到 `--out/_working/...`，完成后整体移动到 `--out/complete/...`（避免 Emby 扫到半成品）。
- 若下载过程中出现可忽略的问题（例如部分图片 403），仍会继续生成 NFO 并标记已下载，但会移动到 `download.needs_manual_subdir`（默认 `needs_manual`）等待手工处理；影片目录内会生成 `_NEEDS_MANUAL.txt` 记录失败项。
- 视频文件名优先使用 `videos.h1`（更易读）；Windows 下会自动截断以降低路径过长风险。
- NFO 文件名与视频同名（例如 `xxx.mp4` 对应 `xxx.nfo`）。
- screenshot 会写到 `extrafanart/fanart1.*`；不在影片目录下写 `fanart.*`/`thumb.*`。
- 可在 `download.workers` 开启多线程并发下载（按视频维度）；进度输出由 `download.show_progress` 控制。
- 进度输出优先使用数据库/JSON-LD 的 `duration_seconds`；若缺失，会尝试从 m3u8 清单的 `#EXTINF` 估算总时长来显示百分比。
- NFO 的 `<plot>` 仅使用 JSON-LD 的 `description`（不再回退写入 URL）。
- 进度里 `net=...MiB/s` 为脚本按输出字节增量估算的吞吐（更接近“网速”）；不再使用 ffmpeg 的 `speed=3.2x`（那是相对实时播放倍速）。
- `download.engine` 可选 `ffmpeg` 或 `aria2`；`aria2` 会按单视频分片并发（`download.concurrent_segments`，默认 16）下载后再本地合并。
- `download.max_missing_segments` 允许 aria2 模式缺失少量 ts 分片并继续合并（会造成画面/音频跳跃，默认 0）。

示例：

```bash
python3 -m xchina_crawler.emby_downloader --config config.yaml --out "/path/to/MediaRoot" --limit 50
```

指定单个视频：

```bash
python3 -m xchina_crawler.emby_downloader --config config.yaml --out "/path/to/MediaRoot" --video-id 699ae9ca4d1d8
```

强制重下（忽略 DB 标记/已有文件）：

```bash
python3 -m xchina_crawler.emby_downloader --config config.yaml --out "/path/to/MediaRoot" --video-id 699ae9ca4d1d8 --force
```

不移动到 complete（直接落在 `--out` 下）：

```bash
python3 -m xchina_crawler.emby_downloader --config config.yaml --out "/path/to/MediaRoot" --video-id 699ae9ca4d1d8 --no-move
```

也可以把 `download.out_dir / download.limit / download.move_to_complete` 等写到 `config.yaml`，这样日常跑的时候只需要：

```bash
python3 -m xchina_crawler.emby_downloader --config config.yaml
```
