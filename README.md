# xchina 元数据爬虫（先做：`series-63824a975d8ae` / 中文AV）

这个项目用于：
- 从“板块（series）列表页”自动发现子分类（如：麻豆传媒(3549)、独立创作者(1838)…）
- 抓取视频**元数据**并存入 Postgres，便于你后续清洗/增强

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
- `crawl.*`：抓取策略默认值（用配置尽量减少命令行参数）

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
