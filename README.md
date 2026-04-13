# classifyd

Tdarr-style **Go** daemon: SQLite job queue, worker pool, REST API, embedded web UI.

Each job runs **ffprobe** on the video path and stores JSON with:

- `video_stats` — duration, codecs, resolution, fps, container
- `metadata_tags` — flattened `format` / `video` / `audio` stream tags from ffprobe
- `frequency_tokens` — deduplicated strings used for **cross-library frequency** (which values are common vs rare)
- `probe` — raw ffprobe JSON

Replace or extend `internal/pipeline` to add ONNX or another tagger; `frequency_tokens` can be extended with ML tags the same way.

## Build

```bash
go build -o classifyd ./cmd/classifyd
```

Needs **ffprobe** on `PATH` (install **ffmpeg**).

## Run

```bash
export CLASSIFYD_LISTEN=:8080
export CLASSIFYD_DATA=./data
export CLASSIFYD_WORKERS=2
./classifyd
```

Open **http://127.0.0.1:8080/** — dashboard with progress, job list (ffprobe summary per row), **tag frequency** and **codec/container** analytics.

### Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `CLASSIFYD_LISTEN` | `:8080` | HTTP listen address |
| `CLASSIFYD_DATA` | `./data` | SQLite directory |
| `CLASSIFYD_WORKERS` | `1` | Concurrent workers |
| `FFPROBE_BIN` | `ffprobe` | ffprobe binary |

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | `{"ok":true}` |
| GET | `/api/v1/stats` | Queue totals (pending/processing/done/failed/total) |
| GET | `/api/v1/jobs?status=&limit=&offset=` | List jobs with optional status filter + pagination |
| GET | `/api/v1/jobs/{id}` | Single job detail |
| DELETE | `/api/v1/jobs/{id}` | Delete a job |
| POST | `/api/v1/scan` | `{"root":"/path"}` — scan dir, enqueue videos |
| GET | `/api/v1/library/roots` | List scanned library roots |
| DELETE | `/api/v1/library/roots` | `{"path":"/path"}` — remove a root |
| POST | `/api/v1/retry` | `{"id":"<uuid>"}` or `{"id":"all"}` — re-queue failed job(s) |
| POST | `/api/v1/purge` | `{"status":"done"}` or `{"status":"failed"}` — bulk delete |
| GET | `/api/v1/analytics/ffprobe` | Histograms: video/audio codec, container, avg duration (finished jobs) |
| GET | `/api/v1/analytics/tags?limit=&min_count=&prefix=` | Token frequency + `% of finished jobs` + heuristic notes |
| GET | `/api/v1/jobs/{id}/tags` | `frequency_tokens` stored for that job |
| POST | `/api/v1/analytics/rebuild-tags` | Rebuild `job_tags` from `result_json` (after upgrade / repair) |

## Docker

```bash
docker build -t classifyd .
docker run -p 8080:8080 -v classifyd-data:/data classifyd
```
