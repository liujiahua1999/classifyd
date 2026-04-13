# classifyd

Tdarr-style **Go** daemon: SQLite job queue, worker pool, REST API, embedded web UI.

Each job runs **ffprobe** on the video path and stores JSON (`tagger: ffprobe-metadata`). Replace or extend `internal/pipeline` to add ONNX or another tagger.

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

Open **http://127.0.0.1:8080/** — dashboard with progress bar, ETA, job filters, retry/purge controls.

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

## Docker

```bash
docker build -t classifyd .
docker run -p 8080:8080 -v classifyd-data:/data classifyd
```
