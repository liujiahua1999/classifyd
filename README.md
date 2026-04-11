# Anime video classifier (WD14 only)

Samples frames with **ffmpeg**, runs the **WD14** Danbooru tagger **locally** (ONNX via [dghs-imgutils](https://github.com/deepghs/imgutils)), writes **JSON** (no cloud APIs).

## Setup

```bash
pip install -r requirements.txt
```

First inference downloads WD14 weights from Hugging Face (cache under `~/.cache/huggingface`). Optional: set `HF_TOKEN` for rate limits.

## Usage

```bash
python3 anime_video_classifier.py ./video.mkv -o out.json
python3 anime_video_classifier.py Video out.json          # same as -o out.json
python3 anime_video_classifier.py Video -o out.json --dry-run   # ffmpeg only, no tagging
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--frames` | 12 | Frames sampled per video |
| `--max-side` | 1024 | Longer JPEG side (px) |
| `--wd14-model` | `SwinV2_v3` | e.g. `ConvNext_v3`, `ViT_v3` |
| `--wd14-general-threshold` | 0.15 | Per-frame WD14 cutoff (low = more recall before merging) |
| `--wd14-character-threshold` | 0.45 | Per-frame character cutoff |
| `--min-score` | **0.9** | **Final** cutoff: only tags with merged score **>** this appear in `general_tags` / `character_tags` |
| `--character-names` | *(optional)* | Text file: one Danbooru-style name per line. If **`character_tags` are empty** after the score filter, names that appear in the **video file stem** are added (`score` 1.0, `source: filename`). If omitted, **`character_names.txt`** next to the script is used when that file exists. Env: `CHARACTER_NAMES_FILE`. |
| `--character-llm` | off | If tags are **still** empty after the steps above, call an **OpenAI-compatible** `chat/completions` API to infer names from the filename (`source: llm`). Requires credentials (see below). Env: `CHARACTER_LLM=1` also enables. |
| `--character-llm-api-file` | `api` if present | Two lines: **API key**, then **base URL** (must include `/v1`, e.g. `https://example.com/v1`). Env: `OPENAI_COMPAT_API_FILE`, or use **`OPENAI_API_KEY`** + **`OPENAI_BASE_URL`** instead. |
| `--character-llm-model` | `gpt-4o-mini` | Model id on your provider. Env: `CHARACTER_LLM_MODEL`. |
| `--workers` | 1 | Parallel videos (ONNX uses a lock if >1) |
| `--dry-run` | | Extract frames only; skip WD14 |

Each **`wd14`** object includes **`summary`**: `output_min_score`, inference thresholds, before/after tag counts, `dominant_rating_guess`, Danbooru-style **`general_tag_string`** / **`character_tag_string`**, **`character_filename_fallback`** (whitelist substring matches, if any), optional **`character_llm_names`** / **`character_llm_model`**, and **`combined_high_confidence`**. Only **`rating`** entries with score > `--min-score` are kept; `general_tags` / `character_tags` are likewise filtered. Filename and LLM fallbacks run only when **no** character tags remain above the cutoff after WD14 (whitelist first, then LLM).

**Git:** add your local `api` key file to `.gitignore` (already listed) and do not commit keys.

## Requirements

- Python 3.10+, **ffmpeg** / **ffprobe** on `PATH`
- Packages in `requirements.txt`
