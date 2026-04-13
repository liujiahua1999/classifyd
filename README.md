# classifyd

Tdarr-style **Go** daemon for video tagging: SQLite job queue, worker pool, REST API, embedded web UI.

## Tagging Pipeline

**Default:** runs `ffprobe` only (stores codec/container/duration metadata).

**WD14 mode:** set `PYTHON_SCRIPT` to call `anime_video_classifier.py` as a subprocess. Each video is tagged with the **WD14 Danbooru tagger** (ONNX) — general tags, character tags, and rating — with character identification as the primary goal:

1. WD14 tags N sampled frames → aggregate max scores across frames
2. If no character tags survive the score cutoff → match `character_names.txt` against the filename
3. If still empty → optional LLM fallback (OpenAI-compatible API)

The Go service stores the full JSON result plus indexed columns (`character_tag_string`, `general_tag_string`, `dominant_rating`, etc.) for fast SQL lookups, and `job_tags` for cross-library frequency analysis.

## Build

```bash
go build -o classifyd ./cmd/classifyd
```

Needs **ffprobe** on `PATH`. For WD14 mode, also needs Python 3.10+ with `dghs-imgutils` and `onnxruntime`.

## Run

```bash
# ffprobe-only (default)
./classifyd

# WD14 mode
export PYTHON_SCRIPT=./anime_video_classifier.py
export CLASSIFYD_WORKERS=2
./classifyd
```

Open **http://127.0.0.1:8080/** — character-focused dashboard with tag analytics, rating breakdown, progress, and job management.

### Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `CLASSIFYD_LISTEN` | `:8080` | HTTP listen address |
| `CLASSIFYD_DATA` | `./data` | SQLite directory |
| `CLASSIFYD_WORKERS` | `1` | Concurrent workers |
| `FFPROBE_BIN` | `ffprobe` | ffprobe binary |
| `PYTHON_BIN` | `python3` | Python interpreter |
| `PYTHON_SCRIPT` | *(empty)* | Path to `anime_video_classifier.py` (enables WD14) |
| `PYTHON_EXTRA_ARGS` | *(empty)* | Extra flags forwarded to the script (e.g. `--character-llm --frames 16`) |

All Python-side env vars (`CHARACTER_LLM`, `OPENAI_API_KEY`, `WD14_MODEL`, etc.) are inherited by the subprocess.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | `{"ok":true}` |
| GET | `/api/v1/stats` | Queue totals |
| GET | `/api/v1/jobs?status=&limit=&offset=` | Jobs (includes `character_tag_string`, `dominant_rating`, etc.) |
| GET | `/api/v1/jobs/{id}` | Single job with full `result_json` |
| GET | `/api/v1/jobs/{id}/tags` | Frequency tokens for one job |
| DELETE | `/api/v1/jobs/{id}` | Delete a job |
| POST | `/api/v1/scan` | `{"root":"/path"}` — enqueue videos |
| GET | `/api/v1/library/roots` | Scanned roots |
| DELETE | `/api/v1/library/roots` | Remove a root |
| POST | `/api/v1/retry` | `{"id":"<uuid>"}` or `{"id":"all"}` |
| POST | `/api/v1/purge` | `{"status":"done\|failed"}` |
| GET | `/api/v1/analytics/ffprobe` | Codec/container/rating/character histograms |
| GET | `/api/v1/analytics/tags?prefix=&limit=&min_count=` | Tag frequency + notes |
| POST | `/api/v1/analytics/rebuild-tags` | Rebuild tag index from JSON |

## Docker

```bash
docker build -t classifyd .
docker run -p 8080:8080 -v classifyd-data:/data classifyd
```
