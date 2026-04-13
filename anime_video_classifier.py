#!/usr/bin/env python3
"""
Sample video frames with ffmpeg, run local WD14 (Danbooru-style) tagging via ONNX.
Write results as JSON array, JSON Lines (jsonl), or SQLite (--output-format / file extension).

Outputs only tags with merged score above --min-score (default 0.9), with a summary block
(tag strings, counts, dominant rating). Uses low per-frame inference thresholds by default
so high-confidence tags are not missed before merging across frames. Optional
character_names.txt: if WD14 yields no character tags above the cutoff, match listed names
against the video file stem. Optional OpenAI-compatible API (--character-llm): small chat model
reads the filename when tags are still empty.

Requires: pip install -r requirements.txt  (dghs-imgutils, onnxruntime)
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import logging
import os
import sqlite3
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Single lowercase set: paths are matched with suffix.lower()
VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"})
# Progress logging while scanning (every N paths seen under a directory root)
_COLLECT_PROGRESS_INTERVAL = 10_000

DEFAULT_WD14_MODEL = "SwinV2_v3"
DEFAULT_CHARACTER_NAMES_PATH = Path(__file__).resolve().parent / "character_names.txt"
DEFAULT_OPENAI_COMPAT_API_FILE = Path(__file__).resolve().parent / "api"
DEFAULT_CHARACTER_LLM_MODEL = "gpt-4o-mini"
_wd14_lock = threading.Lock()
_character_llm_lock = threading.Lock()


def _normalize_name_match(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[_\s.\-+]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_character_names(path: Path | None) -> list[str]:
    """
    One Danbooru-style name per line (underscores ok); # starts a comment; blank lines skipped.
    """
    if path is None or not path.is_file():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def match_character_names_in_filename(stem: str, names: list[str]) -> list[str]:
    """
    Return names from `names` whose normalized form appears as a substring in the stem.
    Longer names are matched first; overlapping spans are masked so shorter tokens inside
    a longer matched name are not added twice.
    """
    if not names:
        return []
    hay = _normalize_name_match(stem)
    if not hay:
        return []
    # Longest first so e.g. "hatsune_miku" wins before "miku" in the same span.
    ordered = sorted(set(names), key=lambda n: (-len(_normalize_name_match(n)), n.lower()))
    matched: list[str] = []
    for name in ordered:
        nn = _normalize_name_match(name)
        if len(nn) < 2:
            continue
        pos = hay.find(nn)
        if pos < 0:
            continue
        matched.append(name)
        hay = hay[:pos] + " " * len(nn) + hay[pos + len(nn) :]
    return matched


@dataclass
class CharacterLLMConfig:
    """OpenAI-compatible /v1/chat/completions (Bearer auth)."""

    api_key: str
    base_url: str
    model: str
    timeout_sec: float = 45.0


def load_openai_compat_api_file(path: Path) -> tuple[str | None, str | None]:
    """
    Two-line file: line 1 = API key, line 2 = base URL (e.g. https://host/v1).
    Lines starting with # are ignored; blank lines skipped.
    """
    if not path.is_file():
        return None, None
    key: str | None = None
    base: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if key is None:
            key = s
        elif base is None:
            base = s
        else:
            break
    return key, base


def resolve_character_llm_config(
    *,
    enabled: bool,
    api_file: Path | None,
    model: str,
) -> CharacterLLMConfig | None:
    if not enabled:
        return None
    path = api_file
    if path is None:
        env_p = os.environ.get("OPENAI_COMPAT_API_FILE")
        path = Path(env_p).expanduser() if env_p else DEFAULT_OPENAI_COMPAT_API_FILE
    f_key, f_base = load_openai_compat_api_file(path) if path else (None, None)
    api_key = os.environ.get("OPENAI_API_KEY") or f_key
    base_url = (os.environ.get("OPENAI_BASE_URL") or f_base or "").strip().rstrip("/")
    if not api_key or not base_url:
        log.warning(
            "Character LLM enabled but missing OPENAI_API_KEY / OPENAI_BASE_URL "
            "(or two-line api file: key + base URL)."
        )
        return None
    return CharacterLLMConfig(api_key=api_key, base_url=base_url, model=model)


class CharacterLLMRequestError(RuntimeError):
    """OpenAI-compatible API error from chat/completions (parsed HTTP body when possible)."""

    def __init__(
        self,
        message: str,
        *,
        http_code: int | None = None,
        api_error_code: str | None = None,
        api_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.http_code = http_code
        self.api_error_code = api_error_code
        self.api_message = api_message


def _parse_chat_completions_http_error(e: urllib.error.HTTPError) -> CharacterLLMRequestError:
    body = e.read() if e.fp else b""
    text = body.decode("utf-8", errors="replace")
    api_code: str | None = None
    api_msg: str | None = None
    try:
        j = json.loads(text)
        err = j.get("error")
        if isinstance(err, dict):
            ac = err.get("code")
            api_code = str(ac) if ac is not None else None
            api_msg = err.get("message")
            if isinstance(api_msg, str):
                pass
            else:
                api_msg = None
        elif isinstance(err, str):
            api_msg = err
    except json.JSONDecodeError:
        pass
    if api_msg is None:
        api_msg = text.strip() or (e.reason or "")
    short = f"LLM HTTP {e.code}: {(api_msg or '')[:500]}"
    return CharacterLLMRequestError(
        short,
        http_code=e.code,
        api_error_code=api_code,
        api_message=api_msg,
    )


def _is_content_filter_error(e: CharacterLLMRequestError) -> bool:
    if (e.api_error_code or "").lower() == "content_filter":
        return True
    msg = (e.api_message or "").lower()
    return "content_filter" in msg or "content management policy" in msg


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def llm_extract_character_names_from_filename(
    *,
    stem: str,
    basename: str,
    cfg: CharacterLLMConfig,
    encode_filename_b64: bool = False,
) -> tuple[list[str], str | None]:
    """
    Call OpenAI-compatible chat completions. Returns (names, raw_assistant_text_or_none).

    encode_filename_b64: send the file label as Base64 so literal filenames are not in the
    request body (helps with Azure / OpenAI content filters on explicit filenames).
    """
    if encode_filename_b64:
        system = (
            "You extract anime or game character names that a file label likely refers to. "
            "The user message is UTF-8 text, Base64-encoded on a single line after the label. "
            "Decode the Base64 first, then read the decoded lines "
            '"File name (no path): ..." and "Stem only: ...". '
            "Use Danbooru-style tags: lowercase words joined by underscores, no spaces. "
            "Reply with ONLY a JSON object, no markdown: "
            '{"characters":["name_one","name_two"]}. '
            "Use an empty array if there are no recognizable character names."
        )
        plain = f"File name (no path): {basename}\nStem only: {stem}"
        b64 = base64.b64encode(plain.encode("utf-8")).decode("ascii")
        user = f"Encoded file label (Base64, UTF-8):\n{b64}"
    else:
        system = (
            "You extract anime or game character names that the filename likely refers to. "
            "Use Danbooru-style tags: lowercase words joined by underscores, no spaces. "
            "Reply with ONLY a JSON object, no markdown: "
            '{"characters":["name_one","name_two"]}. '
            "Use an empty array if the filename has no recognizable character names."
        )
        user = f"File name (no path): {basename}\nStem only: {stem}"
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "max_tokens": 256,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {cfg.api_key}")
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_sec) as resp:
            raw = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        raise _parse_chat_completions_http_error(e) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"LLM request failed: {e}") from e

    choices = raw.get("choices") or []
    if not choices:
        return [], None
    msg = (choices[0].get("message") or {}) if isinstance(choices[0], dict) else {}
    content = (msg.get("content") or "").strip()
    if not content:
        return [], None
    try:
        data = json.loads(_strip_json_fence(content))
    except json.JSONDecodeError:
        return [], content
    chars = data.get("characters")
    if not isinstance(chars, list):
        return [], content
    out: list[str] = []
    seen: set[str] = set()
    for x in chars:
        if not isinstance(x, str):
            continue
        name = x.strip()
        if not name:
            continue
        low = name.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(name)
    return out, content


def apply_llm_character_fallback(
    wd: dict[str, Any],
    *,
    video: Path,
    cfg: CharacterLLMConfig | None,
    use_lock: bool,
    encode_filename_b64: bool = False,
) -> dict[str, Any]:
    """
    If character_tags is still empty, ask a small LLM (OpenAI-compatible API) to read the filename.
    Mutates and returns wd (same dict).
    """
    if cfg is None:
        return wd
    tags = wd.get("character_tags") or []
    if tags:
        return wd
    summary = wd.setdefault("summary", {})
    ctx = _character_llm_lock if use_lock else contextlib.nullcontext()
    with ctx:
        try:
            names, raw_content = llm_extract_character_names_from_filename(
                stem=video.stem,
                basename=video.name,
                cfg=cfg,
                encode_filename_b64=encode_filename_b64,
            )
        except CharacterLLMRequestError as e:
            if _is_content_filter_error(e):
                log.warning(
                    "Character LLM blocked by content filter (often explicit words in the filename). "
                    "Retry with --character-llm-b64-filename, or adjust Azure OpenAI content filtering "
                    "for your deployment."
                )
                summary["character_llm_content_filter"] = True
                summary["character_llm_error"] = (
                    "content_filter: upstream blocked the request (prompt/filename)."
                )
            else:
                log.warning("Character LLM fallback failed: %s", e)
                summary["character_llm_error"] = (e.api_message or str(e))[:2000]
            return wd
        except Exception as e:
            log.warning("Character LLM fallback failed: %s", e)
            summary["character_llm_error"] = str(e)[:2000]
            return wd
    if raw_content is not None:
        summary["character_llm_raw_reply"] = raw_content[:2000]
    summary["character_llm_model"] = cfg.model
    summary["character_llm_filename_encoding"] = "base64" if encode_filename_b64 else "plain"
    if not names:
        summary["character_llm_names"] = []
        return wd
    summary["character_llm_names"] = names
    for nm in names:
        tags.append(
            {
                "name": nm,
                "score": 0.95,
                "source": "llm",
            }
        )
    wd["character_tags"] = tags

    def danbooru_join(tlist: list[dict[str, Any]]) -> str:
        return ", ".join(str(t["name"]).replace(" ", "_") for t in tlist)

    gen_hi = wd.get("general_tags") or []
    summary["character_tag_string"] = danbooru_join(tags)
    summary["character_tags_above_cutoff"] = len(tags)
    summary["combined_high_confidence"] = [
        *[f"general:{t['name']}" for t in gen_hi],
        *[f"character:{t['name']}" for t in tags],
    ]
    return wd


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def probe(path: Path) -> dict[str, Any]:
    r = run_cmd(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ]
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr or "ffprobe failed")
    data = json.loads(r.stdout)
    fmt = data.get("format") or {}
    vstreams = [s for s in (data.get("streams") or []) if s.get("codec_type") == "video"]
    v = vstreams[0] if vstreams else {}
    dur = None
    try:
        dur = float(fmt.get("duration") or 0) or None
    except (TypeError, ValueError):
        pass
    return {
        "duration_sec": dur,
        "format_name": (fmt.get("format_name") or "").split(",")[0] or None,
        "video_codec": v.get("codec_name"),
        "width": int(v.get("width") or 0) or None,
        "height": int(v.get("height") or 0) or None,
    }


def extract_frames(video: Path, out_dir: Path, n: int, max_side: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = probe(video)
    dur = float(meta.get("duration_sec") or 60.0)
    dur = max(dur, 1.0)
    interval = dur / max(n, 1)
    scale = f"scale='min({max_side},iw)':'-2':flags=lanczos"
    pattern = str(out_dir / "f_%04d.jpg")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
        "-vf", f"fps=1/{interval:.6f},{scale}",
        "-frames:v", str(n), "-q:v", "4", pattern,
    ]
    r = run_cmd(cmd)
    if r.returncode != 0:
        raise RuntimeError(r.stderr or "ffmpeg failed")
    frames = sorted(out_dir.glob("f_*.jpg"))
    if not frames:
        raise RuntimeError("No frames extracted")
    return frames


# ---------------------------------------------------------------------------
# WD14
# ---------------------------------------------------------------------------


def aggregate_wd14(
    frame_paths: list[Path],
    *,
    model_name: str,
    general_threshold: float,
    character_threshold: float,
    use_lock: bool,
) -> dict[str, Any]:
    try:
        from imgutils.tagging import get_wd14_tags
    except ImportError as e:
        raise RuntimeError(
            "WD14 deps missing: pip install -r requirements.txt"
        ) from e

    rating_max: dict[str, float] = {}
    gen_max: dict[str, float] = {}
    gen_hits: dict[str, int] = {}
    char_max: dict[str, float] = {}
    char_hits: dict[str, int] = {}

    def run_one(fp: Path):
        return get_wd14_tags(
            str(fp),
            model_name=model_name,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
        )

    for fp in frame_paths:
        if use_lock:
            with _wd14_lock:
                rating, general, character = run_one(fp)
        else:
            rating, general, character = run_one(fp)
        for k, v in (rating or {}).items():
            kk = str(k)
            rating_max[kk] = max(rating_max.get(kk, 0.0), float(v))
        for k, v in (general or {}).items():
            kk = str(k)
            fv = float(v)
            gen_max[kk] = max(gen_max.get(kk, 0.0), fv)
            gen_hits[kk] = gen_hits.get(kk, 0) + 1
        for k, v in (character or {}).items():
            kk = str(k)
            fv = float(v)
            char_max[kk] = max(char_max.get(kk, 0.0), fv)
            char_hits[kk] = char_hits.get(kk, 0) + 1

    def tag_list(scores: dict[str, float], hits: dict[str, int]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, score in sorted(scores.items(), key=lambda x: -x[1]):
            row: dict[str, Any] = {"name": name, "score": score}
            if name in hits:
                row["frames"] = hits[name]
            out.append(row)
        return out

    raw = {
        "model": model_name,
        "frames_tagged": len(frame_paths),
        "rating": rating_max,
        "general_tags": tag_list(gen_max, gen_hits),
        "character_tags": tag_list(char_max, char_hits),
    }
    return raw


def finalize_wd14_output(
    raw: dict[str, Any],
    *,
    min_score: float,
    inference_general: float,
    inference_character: float,
    filename_stem: str | None = None,
    character_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Keep only tags with score > min_score (default 0.9). Add a compact summary for downstream use.
    If character_tags are empty after filtering and `character_names` is non-empty, try matching
    names against `filename_stem` (not WD14 scores; tagged with source=filename).
    """
    gen_all = raw.get("general_tags") or []
    char_all = raw.get("character_tags") or []
    n_gen = len(gen_all)
    n_char = len(char_all)

    def filt(tags: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [t for t in tags if float(t.get("score") or 0) > min_score]

    gen_hi = filt(gen_all)
    char_hi_wd14 = filt(char_all)
    char_hi = list(char_hi_wd14)

    filename_fallback: list[str] = []
    if not char_hi and character_names and filename_stem:
        filename_fallback = match_character_names_in_filename(filename_stem, character_names)
        for nm in filename_fallback:
            char_hi.append(
                {
                    "name": nm,
                    "score": 1.0,
                    "source": "filename",
                }
            )

    rating_raw = raw.get("rating") or {}
    rating_hi = {k: float(v) for k, v in rating_raw.items() if float(v) > min_score}
    best_rating: str | None = None
    if rating_raw:
        best_rating = max(rating_raw.items(), key=lambda kv: float(kv[1]))[0]

    def danbooru_join(tags: list[dict[str, Any]]) -> str:
        return ", ".join(str(t["name"]).replace(" ", "_") for t in tags)

    summary: dict[str, Any] = {
        "output_min_score": min_score,
        "inference_thresholds": {
            "general": inference_general,
            "character": inference_character,
        },
        "counts": {
            "general_tags_above_cutoff": len(gen_hi),
            "general_tags_total_before_cutoff": n_gen,
            "character_tags_above_cutoff": len(char_hi),
            "character_tags_wd14_above_cutoff": len(char_hi_wd14),
            "character_tags_total_before_cutoff": n_char,
        },
        "dominant_rating_guess": best_rating,
        "rating_above_cutoff": rating_hi,
        "general_tag_string": danbooru_join(gen_hi),
        "character_tag_string": danbooru_join(char_hi),
        "character_filename_fallback": filename_fallback or None,
        "combined_high_confidence": [
            *[f"general:{t['name']}" for t in gen_hi],
            *[f"character:{t['name']}" for t in char_hi],
        ],
    }

    return {
        "model": raw.get("model"),
        "frames_tagged": raw.get("frames_tagged"),
        "summary": summary,
        "rating": rating_hi,
        "general_tags": gen_hi,
        "character_tags": char_hi,
    }


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class Row:
    video_path: str
    duration_sec: float | None
    container_format: str | None
    video_codec: str | None
    width: int | None
    height: int | None
    frames_sampled: int
    tagger: str
    wd14: dict[str, Any]
    error: str | None = None
    processed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        if d.get("error") is None:
            del d["error"]
        return d


def collect_videos(
    paths: list[Path],
    *,
    verbose_scan: bool = False,
) -> list[Path]:
    """
    Walk each path once (not once per extension). Large/complex trees are much faster
    than the previous 12× rglob approach.
    """
    out: set[Path] = set()

    def consider_file(f: Path) -> None:
        nonlocal scanned
        scanned += 1
        if verbose_scan and scanned % _COLLECT_PROGRESS_INTERVAL == 0:
            log.info(
                "Scan progress: %d paths examined, %d video(s) found so far",
                scanned,
                len(out),
            )
        try:
            if not f.is_file():
                return
        except OSError:
            return
        if f.suffix.lower() not in VIDEO_SUFFIXES:
            return
        try:
            out.add(f.resolve())
        except OSError:
            pass

    for raw in paths:
        # Per input path so a file listed after a large directory does not reuse that dir's count.
        scanned = 0
        p = raw.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        if p.is_dir():
            t0 = time.perf_counter()
            n_videos_before = len(out)
            if verbose_scan:
                log.info("Scanning for videos under %s ...", p)
            for f in p.rglob("*"):
                consider_file(f)
            if verbose_scan:
                log.info(
                    "Finished %s (%d paths in %.1fs, +%d video(s) under this root)",
                    p,
                    scanned,
                    time.perf_counter() - t0,
                    len(out) - n_videos_before,
                )
        elif p.is_file():
            consider_file(p)
        else:
            raise ValueError(f"Not a file or directory: {p}")
    return sorted(out)


def infer_output_format(path: Path, explicit: str) -> str:
    """
    explicit: auto | json | jsonl | sqlite
    """
    e = (explicit or "auto").strip().lower()
    if e != "auto":
        return e
    suf = path.suffix.lower()
    if suf == ".jsonl":
        return "jsonl"
    if suf in (".db", ".sqlite", ".sqlite3"):
        return "sqlite"
    return "json"


class JsonlResultSink:
    """One JSON object per line, flushed after each row so output is visible during long runs."""

    def __init__(self, path: Path, lock: threading.Lock | None = None) -> None:
        self._fp = path.open("w", encoding="utf-8")
        self._lock = lock

    def write_row(self, d: dict[str, Any]) -> None:
        line = json.dumps(d, ensure_ascii=False) + "\n"
        if self._lock is not None:
            with self._lock:
                self._fp.write(line)
                self._fp.flush()
        else:
            self._fp.write(line)
            self._fp.flush()

    def close(self) -> None:
        self._fp.close()


class SqliteResultSink:
    """Single table with full row JSON; each insert is committed so readers can see rows while the job runs."""

    def __init__(self, path: Path, lock: threading.Lock | None = None) -> None:
        self._conn = sqlite3.connect(str(path))
        self._lock = lock
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS video_tags (
                video_path TEXT PRIMARY KEY NOT NULL,
                payload TEXT NOT NULL
            )"""
        )
        self._conn.commit()

    def write_row(self, d: dict[str, Any]) -> None:
        payload = json.dumps(d, ensure_ascii=False)
        vp = str(d.get("video_path", ""))

        def ins() -> None:
            self._conn.execute(
                "INSERT OR REPLACE INTO video_tags (video_path, payload) VALUES (?, ?)",
                (vp, payload),
            )
            self._conn.commit()

        if self._lock is not None:
            with self._lock:
                ins()
        else:
            ins()

    def close(self) -> None:
        self._conn.close()


def process_one(
    video: Path,
    *,
    frames: int,
    max_side: int,
    wd14_model: str,
    general_threshold: float,
    character_threshold: float,
    min_output_score: float,
    character_names: list[str],
    character_llm: CharacterLLMConfig | None,
    character_llm_use_lock: bool,
    character_llm_b64_filename: bool,
    dry_run: bool,
    wd14_lock: bool,
) -> Row:
    try:
        meta = probe(video)
    except Exception as e:
        return Row(
            video_path=str(video.resolve()),
            duration_sec=None,
            container_format=None,
            video_codec=None,
            width=None,
            height=None,
            frames_sampled=0,
            tagger="wd14",
            wd14={},
            error=str(e),
        )
    tmp = Path(tempfile.mkdtemp(prefix="wd14_"))
    try:
        fpaths = extract_frames(video, tmp, frames, max_side)
        n = len(fpaths)
        if dry_run:
            return Row(
                video_path=str(video.resolve()),
                duration_sec=meta.get("duration_sec"),
                container_format=meta.get("format_name"),
                video_codec=meta.get("video_codec"),
                width=meta.get("width"),
                height=meta.get("height"),
                frames_sampled=n,
                tagger="none",
                wd14={"note": "dry_run: frames extracted, no WD14"},
            )
        raw = aggregate_wd14(
            fpaths,
            model_name=wd14_model,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            use_lock=wd14_lock,
        )
        wd = finalize_wd14_output(
            raw,
            min_score=min_output_score,
            inference_general=general_threshold,
            inference_character=character_threshold,
            filename_stem=video.stem,
            character_names=character_names,
        )
        wd = apply_llm_character_fallback(
            wd,
            video=video,
            cfg=character_llm,
            use_lock=character_llm_use_lock,
            encode_filename_b64=character_llm_b64_filename,
        )
        return Row(
            video_path=str(video.resolve()),
            duration_sec=meta.get("duration_sec"),
            container_format=meta.get("format_name"),
            video_codec=meta.get("video_codec"),
            width=meta.get("width"),
            height=meta.get("height"),
            frames_sampled=n,
            tagger=f"wd14:{wd14_model}",
            wd14=wd,
        )
    except Exception as e:
        log.exception("Failed: %s", video)
        return Row(
            video_path=str(video.resolve()),
            duration_sec=meta.get("duration_sec"),
            container_format=meta.get("format_name"),
            video_codec=meta.get("video_codec"),
            width=meta.get("width"),
            height=meta.get("height"),
            frames_sampled=0,
            tagger=f"wd14:{wd14_model}",
            wd14={},
            error=str(e),
        )
    finally:
        for x in tmp.glob("*"):
            try:
                x.unlink()
            except OSError:
                pass
        try:
            tmp.rmdir()
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract video frames with ffmpeg, tag with local WD14 (Danbooru ONNX)."
    )
    ap.add_argument("inputs", nargs="*", type=Path, help="Videos and/or directories")
    ap.add_argument("-d", "--dir", action="append", type=Path, dest="dirs")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file (format: --output-format, or from extension: .json / .jsonl / .db)",
    )
    ap.add_argument(
        "--output-format",
        choices=("auto", "json", "jsonl", "sqlite"),
        default=os.environ.get("OUTPUT_FORMAT", "auto"),
        help=(
            "auto: infer from path (.jsonl -> jsonl, .db/.sqlite -> sqlite, else json). "
            "json: one JSON array at end (high memory). "
            "jsonl: one object per line, written and flushed after each video. "
            "sqlite: one row committed after each video (WAL; readable while running). "
            "Env: OUTPUT_FORMAT."
        ),
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=int(os.environ.get("FRAMES", "12")),
        help="Frames per video (default 12)",
    )
    ap.add_argument(
        "--max-side",
        type=int,
        default=int(os.environ.get("MAX_SIDE", "1024")),
        help="Max JPEG dimension (default 1024)",
    )
    ap.add_argument(
        "--wd14-model",
        default=os.environ.get("WD14_MODEL", DEFAULT_WD14_MODEL),
        help=f"WD14 ONNX variant (default {DEFAULT_WD14_MODEL})",
    )
    ap.add_argument(
        "--wd14-general-threshold",
        type=float,
        default=float(os.environ.get("WD14_GENERAL_THRESHOLD", "0.15")),
        help="Low threshold per frame so WD14 returns many candidates; final list uses --min-score (default 0.15)",
    )
    ap.add_argument(
        "--wd14-character-threshold",
        type=float,
        default=float(os.environ.get("WD14_CHARACTER_THRESHOLD", "0.45")),
        help="Per-frame character cutoff before merging (default 0.45)",
    )
    ap.add_argument(
        "--min-score",
        type=float,
        default=float(os.environ.get("WD14_MIN_OUTPUT_SCORE", "0.9")),
        metavar="S",
        help="Keep only tags with merged score > S in JSON (default 0.9)",
    )
    ap.add_argument(
        "--character-names",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Optional list: one name per line (# comments ok). If WD14 yields no character "
            f"tags above --min-score, match these names against the video file stem. "
            f"Default: {DEFAULT_CHARACTER_NAMES_PATH.name} next to this script if that file exists. "
            "Env: CHARACTER_NAMES_FILE."
        ),
    )
    ap.add_argument(
        "--character-llm",
        action="store_true",
        help=(
            "If character tags are still empty after WD14 + optional name list, call an "
            "OpenAI-compatible chat API (see --character-llm-api-file / OPENAI_API_KEY). "
            "Env: CHARACTER_LLM=1 enables the same."
        ),
    )
    ap.add_argument(
        "--character-llm-api-file",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            f"Two lines: API key, then base URL (e.g. https://host/v1). "
            f"Default file if present: {DEFAULT_OPENAI_COMPAT_API_FILE.name} beside this script. "
            "Env: OPENAI_COMPAT_API_FILE, or set OPENAI_API_KEY + OPENAI_BASE_URL."
        ),
    )
    ap.add_argument(
        "--character-llm-model",
        default=os.environ.get("CHARACTER_LLM_MODEL", DEFAULT_CHARACTER_LLM_MODEL),
        metavar="NAME",
        help="Chat model id for filename parsing (default gpt-4o-mini or CHARACTER_LLM_MODEL).",
    )
    ap.add_argument(
        "--character-llm-b64-filename",
        action="store_true",
        help=(
            "Send the filename/stem as Base64 in the user message so the API body does not "
            "contain literal file names (often avoids Azure OpenAI content_filter on explicit names). "
            "Env: CHARACTER_LLM_B64_FILENAME=1."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="ffprobe + frame extract only; no WD14 (no model download)",
    )
    ap.add_argument(
        "--stdout-json-one",
        action="store_true",
        help=(
            "Process a single input video and print its JSON result to stdout (implies --quiet). "
            "Used by the Go daemon to call this script as a subprocess."
        ),
    )
    ap.add_argument("--workers", type=int, default=1, metavar="N")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=(
            "Enable debug logs and periodic directory-scan progress "
            f"(every {_COLLECT_PROGRESS_INTERVAL} paths) for large trees"
        ),
    )
    args = ap.parse_args()

    if args.stdout_json_one:
        args.quiet = True

    if args.verbose:
        log.setLevel(logging.DEBUG)

    if args.stdout_json_one:
        if not args.inputs or len(args.inputs) != 1:
            log.error("--stdout-json-one requires exactly one input video path.")
            return 1
        video = args.inputs[0].expanduser().resolve()
        if not video.is_file():
            log.error("Not a file: %s", video)
            return 1

        cn_path: Path | None = args.character_names
        if cn_path is None:
            cn_path_env = os.environ.get("CHARACTER_NAMES_FILE")
            if cn_path_env:
                cn_path = Path(cn_path_env).expanduser()
            elif DEFAULT_CHARACTER_NAMES_PATH.is_file():
                cn_path = DEFAULT_CHARACTER_NAMES_PATH
        character_names = load_character_names(cn_path)

        llm_on = args.character_llm or os.environ.get("CHARACTER_LLM", "").lower() in ("1", "true", "yes")
        character_llm_cfg = resolve_character_llm_config(
            enabled=llm_on, api_file=args.character_llm_api_file, model=args.character_llm_model,
        )
        character_llm_b64 = args.character_llm_b64_filename or os.environ.get(
            "CHARACTER_LLM_B64_FILENAME", "").lower() in ("1", "true", "yes", "on")

        r = process_one(
            video,
            frames=args.frames,
            max_side=args.max_side,
            wd14_model=args.wd14_model,
            general_threshold=args.wd14_general_threshold,
            character_threshold=args.wd14_character_threshold,
            min_output_score=args.min_score,
            character_names=character_names,
            character_llm=character_llm_cfg,
            character_llm_use_lock=False,
            character_llm_b64_filename=character_llm_b64,
            dry_run=args.dry_run,
            wd14_lock=False,
        )
        sys.stdout.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return 0

    if args.output is None and len(args.inputs) >= 2:
        last = args.inputs[-1]
        if str(last).lower().endswith(
            (".json", ".jsonl", ".db", ".sqlite", ".sqlite3")
        ):
            args.output = last
            args.inputs = list(args.inputs[:-1])

    all_in: list[Path] = list(args.inputs)
    if args.dirs:
        all_in.extend(args.dirs)
    if not all_in:
        log.error("No inputs.")
        return 1

    if run_cmd(["which", "ffmpeg"]).returncode or run_cmd(["which", "ffprobe"]).returncode:
        log.error("ffmpeg and ffprobe must be on PATH")
        return 1

    t_collect = time.perf_counter()
    try:
        videos = collect_videos(all_in, verbose_scan=args.verbose)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 1

    if not videos:
        log.error("No video files found.")
        return 1

    log.info(
        "Discovered %d video file(s) in %.1fs.",
        len(videos),
        time.perf_counter() - t_collect,
    )
    if not args.dry_run:
        log.info(
            "WD14 loads its ONNX model on the first tagged frames (one-time; can take a bit)."
        )

    cn_path: Path | None = args.character_names
    if cn_path is None:
        cn_path_env = os.environ.get("CHARACTER_NAMES_FILE")
        if cn_path_env:
            cn_path = Path(cn_path_env).expanduser()
        elif DEFAULT_CHARACTER_NAMES_PATH.is_file():
            cn_path = DEFAULT_CHARACTER_NAMES_PATH
    character_names = load_character_names(cn_path)
    if args.character_names is not None and not args.character_names.expanduser().is_file():
        log.warning("Character names file not found: %s", args.character_names)
    elif character_names and cn_path is not None:
        log.info("Using %d names from %s for filename character fallback", len(character_names), cn_path)

    llm_on = args.character_llm or os.environ.get("CHARACTER_LLM", "").lower() in (
        "1",
        "true",
        "yes",
    )
    character_llm_cfg = resolve_character_llm_config(
        enabled=llm_on,
        api_file=args.character_llm_api_file,
        model=args.character_llm_model,
    )
    character_llm_b64 = args.character_llm_b64_filename or os.environ.get(
        "CHARACTER_LLM_B64_FILENAME", ""
    ).lower() in ("1", "true", "yes", "on")

    if llm_on and character_llm_cfg:
        log.info(
            "Character LLM fallback enabled (model=%s, base=%s, filename_encoding=%s)",
            character_llm_cfg.model,
            character_llm_cfg.base_url,
            "base64" if character_llm_b64 else "plain",
        )

    workers = max(1, args.workers)
    use_wd14_lock = workers > 1 and not args.dry_run
    use_llm_lock = workers > 1

    out_path = args.output
    output_fmt = "json"
    row_sink: JsonlResultSink | SqliteResultSink | None = None
    accumulate_json_array = False
    if out_path is not None:
        output_fmt = infer_output_format(out_path, args.output_format)
        if output_fmt == "json":
            accumulate_json_array = True
        elif output_fmt == "jsonl":
            row_sink = JsonlResultSink(
                out_path, threading.Lock() if workers > 1 else None
            )
        else:
            row_sink = SqliteResultSink(
                out_path, threading.Lock() if workers > 1 else None
            )
        log.info("Writing results as %s to %s", output_fmt, out_path)

    rows: list[dict[str, Any]] = [] if accumulate_json_array else []

    def work(vp: Path) -> dict[str, Any]:
        log.info("%s", vp)
        r = process_one(
            vp,
            frames=args.frames,
            max_side=args.max_side,
            wd14_model=args.wd14_model,
            general_threshold=args.wd14_general_threshold,
            character_threshold=args.wd14_character_threshold,
            min_output_score=args.min_score,
            character_names=character_names,
            character_llm=character_llm_cfg,
            character_llm_use_lock=use_llm_lock,
            character_llm_b64_filename=character_llm_b64,
            dry_run=args.dry_run,
            wd14_lock=use_wd14_lock,
        )
        d = r.to_json()
        if not args.quiet:
            print(json.dumps(d, indent=2, ensure_ascii=False))
        if row_sink is not None:
            row_sink.write_row(d)
        return d

    try:
        if workers == 1:
            for v in videos:
                d = work(v)
                if accumulate_json_array:
                    rows.append(d)
        else:
            slots: list[dict[str, Any] | None] = [None] * len(videos)
            with ThreadPoolExecutor(max_workers=min(workers, len(videos))) as ex:
                futs = {ex.submit(work, videos[i]): i for i in range(len(videos))}
                for fut in as_completed(futs):
                    i = futs[fut]
                    slots[i] = fut.result()
            if accumulate_json_array:
                rows = [s for s in slots if s is not None]

        if out_path is not None and output_fmt == "json":
            out_path.write_text(
                json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            log.info("Wrote %s (%d rows)", out_path, len(rows))
        elif out_path is not None:
            log.info("Wrote %s (%d rows as %s)", out_path, len(videos), output_fmt)
    finally:
        if row_sink is not None:
            row_sink.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
