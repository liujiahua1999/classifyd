# classifyd

Tdarr-style **Go** daemon for video tagging with **WD14 (Danbooru-style)** character identification. SQLite job queue, worker pool, REST API, embedded web dashboard.

## Pipeline

For each video:

1. **ffprobe** — extract codec, container, duration, resolution metadata
2. **ffmpeg** — extract N sampled frames (JPEG) from the video
3. **WD14 ONNX** — tag each frame via `wd14_tagger.py` (SwinV2_v3); aggregate max scores across frames; output general tags, character tags, and dominant rating
4. **Character fallback** — if WD14 yields no character tags above the cutoff:
   - **Filename matching** — match known names from `character_names.txt` against the video filename (longest-first, span masking)
   - **LLM fallback** — call an OpenAI-compatible chat API to read the filename and extract Danbooru-style character tags

If `wd14_tagger.py` is not found, falls back to ffprobe-only mode (metadata + filename/LLM character identification, no visual tagging).

Indexed columns (`character_tag_string`, `general_tag_string`, `dominant_rating`, `tagger`, etc.) enable fast SQL lookups. A `job_tags` table stores per-job frequency tokens for cross-library analytics.

## Build

```bash
go build -o classifyd ./cmd/classifyd
```

Needs **ffprobe** and **ffmpeg** on `PATH`.

### WD14 Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "dghs-imgutils[onnxruntime]" onnxruntime
```

Set `PYTHON_BIN=.venv/bin/python3` to point at the venv Python.

## Run

```bash
PYTHON_BIN=.venv/bin/python3 ./classifyd
```

Place these files next to the binary (or in cwd) for auto-discovery:

- **`wd14_tagger.py`** — WD14 ONNX inference helper (ships with this repo). Auto-discovered; or set `WD14_TAGGER_SCRIPT`.
- **`api`** — two-line file: line 1 = API key, line 2 = base URL (e.g. `https://xiaoai.plus/v1`). Enables LLM character fallback.
- **`character_names.txt`** — one Danbooru name per line (`#` = comment). Enables filename matching.

Open **http://127.0.0.1:8080/** for the dashboard.

### Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `CLASSIFYD_LISTEN` | `:8080` | HTTP listen address |
| `CLASSIFYD_DATA` | `./data` | SQLite directory |
| `CLASSIFYD_WORKERS` | `1` | Concurrent workers |
| `FFPROBE_BIN` | `ffprobe` | ffprobe binary |
| `PYTHON_BIN` | `python3` | Python interpreter (use venv path) |
| `WD14_TAGGER_SCRIPT` | *(auto)* | Path to `wd14_tagger.py` (enables WD14) |
| `WD14_FRAMES` | `12` | Frames to sample per video |
| `WD14_MAX_SIDE` | `1024` | Max dimension for extracted frames |
| `CHARACTER_NAMES_FILE` | *(auto)* | Path to character names file |
| `LLM_API_FILE` | *(auto)* | Path to two-line API credential file |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model for character extraction |
| `LLM_B64_FILENAME` | `false` | Base64-encode filenames (avoids content filters) |

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
