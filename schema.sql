-- xchina 元数据爬虫核心表结构（Postgres）。

create table if not exists taxonomy_nodes (
  id bigserial primary key,
  source text not null default 'xchina',
  node_type text not null, -- 例如：'video_series'

  -- 对于 series 节点，此字段为 series id（例如 '63824a975d8ae'）。
  source_key text not null,

  name text not null,
  url text,
  parent_id bigint references taxonomy_nodes(id) on delete set null,
  item_count_hint integer,

  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  last_seen_at timestamptz
);

create unique index if not exists taxonomy_nodes_uniq
  on taxonomy_nodes(source, node_type, source_key);

create index if not exists taxonomy_nodes_parent_idx
  on taxonomy_nodes(parent_id);

create table if not exists videos (
  video_id text primary key,
  title text,
  h1 text,

  -- “当前抓取上下文”下的所属 series（通常来自列表页）。
  -- 注意：该信息可能并非唯一归类，完整关系请以 video_taxonomy 为准。
  video_series_name text,
  video_series_source_key text,

  -- 列表页 tags 的纯文本项（过滤掉评论/时长/磁链等图标项）。
  video_tags jsonb,

  -- download 页解析到的 magnet/torrent（可能包含短期参数，建议定期刷新）。
  video_magnet_uri text,
  video_torrent_url text,

  -- 下载落盘状态（用于避免重复下载；即使你后续把文件上传网盘并删除本地，也能跳过）。
  download_status text, -- 'done' | 'error'
  downloaded_path text,
  downloaded_at timestamptz,
  download_attempts integer not null default 0,
  download_last_attempt_at timestamptz,
  download_last_error text,

  canonical_url text,
  page_url text,

  cover_url text,
  screenshot_url text,
  screenshot_urls jsonb,

  -- 可直接用于播放的资源链接（可能包含短期签名参数，建议定期刷新）。
  m3u8_url text,
  poster_url text,

  upload_date timestamptz,
  duration_seconds integer,

  content_rating text,
  is_family_friendly boolean,

  jsonld jsonb,
  extract jsonb,

  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  last_crawled_at timestamptz
);

-- 兼容已存在的 videos 表：增量补齐新列（可重复执行）。
alter table videos add column if not exists video_series_name text;
alter table videos add column if not exists video_series_source_key text;
alter table videos add column if not exists video_tags jsonb;
alter table videos add column if not exists video_magnet_uri text;
alter table videos add column if not exists video_torrent_url text;
alter table videos add column if not exists download_status text;
alter table videos add column if not exists downloaded_path text;
alter table videos add column if not exists downloaded_at timestamptz;
alter table videos add column if not exists download_attempts integer not null default 0;
alter table videos add column if not exists download_last_attempt_at timestamptz;
alter table videos add column if not exists download_last_error text;
alter table videos add column if not exists m3u8_url text;
alter table videos add column if not exists poster_url text;
alter table videos add column if not exists screenshot_urls jsonb;

create table if not exists video_taxonomy (
  video_id text not null references videos(video_id) on delete cascade,
  node_id bigint not null references taxonomy_nodes(id) on delete cascade,
  source text not null, -- 'list_page' | 'breadcrumb'
  created_at timestamptz not null default now(),
  primary key (video_id, node_id, source)
);

create table if not exists video_list_observations (
  id bigserial primary key,
  video_id text not null references videos(video_id) on delete cascade,
  node_id bigint references taxonomy_nodes(id) on delete set null,

  list_title text,
  list_cover_url text,
  list_badge text, -- e.g. '杏吧原版'
  list_duration_text text, -- e.g. '01:58:14'
  list_duration_seconds integer,
  list_comment_count integer,
  list_flags jsonb, -- e.g. {"has_magnet_icon": true}

  page_url text,
  seen_at timestamptz not null default now()
);

create index if not exists video_list_observations_video_idx
  on video_list_observations(video_id);

create index if not exists video_list_observations_node_idx
  on video_list_observations(node_id);

-- 下载页元数据：主要存放结构化信息与页面分析字段。
create table if not exists video_download_pages (
  video_id text primary key references videos(video_id) on delete cascade,
  page_url text,
  canonical_url text,
  title text,

  has_download_section boolean,
  has_magnet_button boolean,
  has_torrent_button boolean,

  external_link_domains jsonb,
  extract jsonb,

  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  last_crawled_at timestamptz
);

-- 可选：存储抓取到的 HTML 快照，用于分析/调试。
-- 该表存储的是“脱敏后的 HTML”：会对下载/播放直链做替换，避免直接落库可用链接。
create table if not exists raw_pages (
  id bigserial primary key,
  page_kind text not null, -- 'series' | 'video' | 'download'
  source_key text not null, -- 例如 series_id 或 video_id（series 支持 `series_id:page`）
  url text,
  html_sanitized text not null,
  redactions jsonb,
  sha256 text not null,
  fetched_at timestamptz not null default now()
);

create index if not exists raw_pages_kind_key_idx
  on raw_pages(page_kind, source_key, fetched_at desc);

-- 用于基于 URL 的快取/跳过抓取（同一 URL 不重复抓）。
create index if not exists raw_pages_kind_url_idx
  on raw_pages(page_kind, url, fetched_at desc);

-- 用于基于 sha256 的去重/判断页面是否相同。
create index if not exists raw_pages_kind_source_sha_idx
  on raw_pages(page_kind, source_key, sha256);

-- 最小化的爬取状态表：用于断点续跑/增量策略（当前预留）。
create table if not exists crawl_state (
  key text primary key,
  value jsonb,
  updated_at timestamptz not null default now()
);
