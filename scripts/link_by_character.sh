#!/usr/bin/env bash
# Create a virtual folder tree of symlinks, one subfolder per extracted character tag.
# Reads done jobs from classifyd SQLite (job_tags: character:*) and optional character_tag_string.
#
# Typical invocation (three paths):
#   ./scripts/link_by_character.sh \
#     -b /path/to/classifyd.db \
#     -m /original/emby/library/media \
#     -o /custom/new/linked-folder
#
# Shorter (DB defaults to \$CLASSIFYD_DATA/classifyd.db):
#   CLASSIFYD_DATA=/var/lib/classifyd ./scripts/link_by_character.sh -o /srv/emby/by-character
#
# Ubuntu / SMB notes:
#   - Prefer relative symlinks (default): works across layouts when both paths resolve on the client.
#   - On CIFS/SMB, mount with symlink support, e.g.:
#       //server/share /mnt/share cifs ...,mfsymlinks,uid=...,gid=...
#   - If symlinks are not created on the share, put this OUTPUT tree on local ext4 and point
#     links at SMB paths with -a (absolute targets).
#   - Requires: sqlite3, bash 4+, coreutils (ln with --relative on recent Ubuntu).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${CLASSIFYD_DATA:-${REPO_ROOT}/data}"
DB_FILE="${CLASSIFYD_DB:-}"
OUT_ROOT="${LINK_OUTPUT:-}"
MEDIA_ROOT="${EMBY_MEDIA_ROOT:-}"
DRY_RUN=0
CLEAN=0
ABSOLUTE=0
UNCAT="${LINK_UNCATEGORIZED:-}"

MEDIA_ROOT_CANON=""

abs_dir() {
	local p="$1"
	if [[ ! -d "$p" ]]; then
		echo "error: not a directory: $p" >&2
		exit 1
	fi
	(cd "$p" && pwd)
}

resolve_db_file() {
	local f="$1"
	if [[ "$f" != /* ]]; then
		f="$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
	fi
	if [[ ! -f "$f" ]]; then
		echo "error: database file not found: $f" >&2
		exit 1
	fi
	printf '%s' "$f"
}

path_under_media_root() {
	local f="$1"
	local rf rr
	if [[ -z "${MEDIA_ROOT_CANON:-}" ]]; then
		return 0
	fi
	rr="$MEDIA_ROOT_CANON"
	rf="$(realpath "$f" 2>/dev/null)" || rf="$f"
	[[ "$rf" == "$rr" ]] || [[ "$rf" == "$rr"/* ]]
}

usage() {
	sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
	echo ""
	echo "Main paths (recommended explicit):"
	echo "  -b FILE  classifyd.db location (\$CLASSIFYD_DB)"
	echo "  -m DIR   original Emby/media library root — only link jobs whose stored path"
	echo "           lies under this directory (\$EMBY_MEDIA_ROOT). Omit to include all jobs."
	echo "  -o DIR   new folder for the symlink tree (\$LINK_OUTPUT). Required unless \$LINK_OUTPUT set."
	echo ""
	echo "Other:"
	echo "  -d DIR   classifyd data dir if -b omitted (default: \$CLASSIFYD_DATA or repo/data)"
	echo "  -n       dry run (print actions only)"
	echo "  -c       remove existing symlinks under -o before linking (keeps dirs)"
	echo "  -a       use absolute symlink targets (helps if relative fails on some SMB setups)"
	echo "  -u NAME  folder for done jobs with no character (default: skip)"
	echo ""
	echo "Environment:"
	echo "  CLASSIFYD_DB      Explicit path to classifyd.db (same as -b)"
	echo "  CLASSIFYD_DATA    Data directory used only when -b / CLASSIFYD_DB is unset"
	echo "  EMBY_MEDIA_ROOT Same as -m"
	echo "  LINK_OUTPUT     Same as -o"
	echo "  LINK_UNCATEGORIZED  Same as -u"
	exit "${1:-0}"
}

while getopts "d:o:b:m:ncau:h" opt; do
	case "$opt" in
	d) DATA_DIR="$OPTARG" ;;
	o) OUT_ROOT="$OPTARG" ;;
	b) DB_FILE="$OPTARG" ;;
	m) MEDIA_ROOT="$OPTARG" ;;
	n) DRY_RUN=1 ;;
	c) CLEAN=1 ;;
	a) ABSOLUTE=1 ;;
	u) UNCAT="$OPTARG" ;;
	h) usage 0 ;;
	*) usage 1 ;;
	esac
done

if [[ -z "$OUT_ROOT" ]]; then
	echo "error: set linked folder with -o or LINK_OUTPUT" >&2
	usage 1
fi

if [[ -n "$DB_FILE" ]]; then
	DB_PATH="$(resolve_db_file "$DB_FILE")"
else
	# Relative CLASSIFYD_DATA / -d is resolved from the current working directory.
	if [[ "$DATA_DIR" != /* ]]; then
		DATA_DIR="$(cd "$(dirname "$DATA_DIR")" && pwd)/$(basename "$DATA_DIR")"
	fi
	DATA_DIR="$(abs_dir "$DATA_DIR")"
	DB_PATH="${DATA_DIR}/classifyd.db"
	if [[ ! -f "$DB_PATH" ]]; then
		echo "error: database not found: $DB_PATH (use -b / CLASSIFYD_DB for an explicit path)" >&2
		exit 1
	fi
fi

if [[ -n "$MEDIA_ROOT" ]]; then
	if [[ ! -d "$MEDIA_ROOT" ]]; then
		echo "error: -m / EMBY_MEDIA_ROOT must be an existing directory: $MEDIA_ROOT" >&2
		exit 1
	fi
	if [[ "$MEDIA_ROOT" != /* ]]; then
		MEDIA_ROOT="$(cd "$(dirname "$MEDIA_ROOT")" && pwd)/$(basename "$MEDIA_ROOT")"
	fi
	MEDIA_ROOT="$(abs_dir "$MEDIA_ROOT")"
	MEDIA_ROOT_CANON="$(realpath "$MEDIA_ROOT" 2>/dev/null || echo "$MEDIA_ROOT")"
fi

command -v sqlite3 >/dev/null 2>&1 || {
	echo "error: sqlite3 is required (apt install sqlite3)" >&2
	exit 1
}

mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"

sanitize_dirname() {
	# Safe single path segment for Linux + SMB; preserves UTF-8 except control chars and slashes.
	local s="$1"
	s="${s//$'\r'/}"
	s="${s//$'\n'/}"
	s="${s//\//_}"
	s="${s//:/_}"
	s="$(echo -n "$s" | tr -d '\000')"
	s="${s//\"/}"
	s="${s//#/?}"
	if [[ -z "${s// /}" ]]; then
		s="_unknown"
	fi
	printf '%s' "$s"
}

unique_link_path() {
	local dir="$1"
	local base="$2"
	local shortid="$3"
	local dst="${dir}/${base}"
	if [[ ! -e "$dst" ]]; then
		printf '%s' "$dst"
		return
	fi
	local stem ext
	stem="${base%.*}"
	ext=""
	if [[ "$base" == *.* ]]; then
		ext=".${base##*.}"
	fi
	printf '%s' "${dir}/${stem}__${shortid}${ext}"
}

mkrel() {
	local target="$1"
	local linkpath="$2"
	if [[ "$ABSOLUTE" -eq 1 ]]; then
		if command -v realpath >/dev/null 2>&1; then
			realpath -s "$target"
		else
			readlink -f "$target" 2>/dev/null || echo "$target"
		fi
		return
	fi
	local linkdir out
	linkdir="$(dirname "$linkpath")"
	if command -v realpath >/dev/null 2>&1; then
		out="$(realpath --relative-to="$linkdir" "$target" 2>/dev/null || true)"
	fi
	if [[ -n "${out:-}" ]]; then
		printf '%s' "$out"
		return
	fi
	# Fallback: Python (usually present where classifyd runs)
	python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$target" "$linkdir"
}

do_ln() {
	local target="$1"
	local linkpath="$2"
	local rel
	rel="$(mkrel "$target" "$linkpath")"
	if [[ "$DRY_RUN" -eq 1 ]]; then
		echo "ln -sfn $(printf '%q' "$rel") $(printf '%q' "$linkpath")"
		return
	fi
	mkdir -p "$(dirname "$linkpath")"
	ln -sfn "$rel" "$linkpath"
}

clean_tree() {
	if [[ "$DRY_RUN" -eq 1 ]]; then
		echo "would remove symlinks under $(printf '%q' "$OUT_ROOT")"
		return
	fi
	if [[ -d "$OUT_ROOT" ]]; then
		find "$OUT_ROOT" -type l -delete 2>/dev/null || true
	fi
}

mapfile -t rows < <(
	sqlite3 -batch -noheader -separator '	' "$DB_PATH" "
-- Primary: frequency token tags (character:name)
SELECT j.id, j.path, substr(t.tag, 11) AS cname
FROM jobs j
JOIN job_tags t ON t.job_id = j.id
WHERE j.status = 'done'
  AND t.tag LIKE 'character:%';
"
)

# Fallback: parse character_tag_string when tag rows missing (e.g. DB repair / old rows)
mapfile -t fallback < <(
	sqlite3 -batch -noheader -separator '	' "$DB_PATH" "
SELECT j.id, j.path, j.character_tag_string
FROM jobs j
WHERE j.status = 'done'
  AND j.character_tag_string IS NOT NULL
  AND TRIM(j.character_tag_string) != ''
  AND NOT EXISTS (
    SELECT 1 FROM job_tags t
    WHERE t.job_id = j.id AND t.tag LIKE 'character:%'
  );
"
)

declare -A SEEN
LINKS=0

process_row() {
	local job_id="$1"
	local fpath="$2"
	local cname="$3"
	local key="${job_id}|${fpath}|${cname}"
	if [[ -n "${SEEN[$key]:-}" ]]; then
		return
	fi
	SEEN[$key]=1

	if [[ -n "${MEDIA_ROOT_CANON:-}" ]] && ! path_under_media_root "$fpath"; then
		return
	fi

	if [[ ! -f "$fpath" ]] && [[ ! -L "$fpath" ]]; then
		echo "warn: missing file, skip: $fpath" >&2
		return
	fi

	local shortid="${job_id:0:8}"
	local base safe dir linkpath
	base="$(basename "$fpath")"
	safe="$(sanitize_dirname "$cname")"
	dir="${OUT_ROOT}/${safe}"
	linkpath="$(unique_link_path "$dir" "$base" "$shortid")"
	do_ln "$fpath" "$linkpath"
	LINKS=$((LINKS + 1))
}

process_fallback_line() {
	local job_id="$1" fpath="$2" ctagstr="$3"
	local IFS=,
	local -a PARTS
	read -ra PARTS <<<"$ctagstr"
	local p cleaned
	for p in "${PARTS[@]}"; do
		cleaned="$(echo "$p" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
		cleaned="${cleaned//_/ }"
		[[ -z "$cleaned" ]] && continue
		process_row "$job_id" "$fpath" "$cleaned"
	done
}

if [[ "$CLEAN" -eq 1 ]]; then
	clean_tree
fi

while IFS=$'\t' read -r job_id fpath cname; do
	[[ -z "${job_id:-}" ]] && continue
	process_row "$job_id" "$fpath" "$cname"
done < <(printf '%s\n' "${rows[@]:-}")

while IFS=$'\t' read -r job_id fpath ctagstr; do
	[[ -z "${job_id:-}" ]] && continue
	process_fallback_line "$job_id" "$fpath" "$ctagstr"
done < <(printf '%s\n' "${fallback[@]:-}")

if [[ -n "$UNCAT" ]]; then
	mapfile -t uncat_rows < <(
		sqlite3 -batch -noheader -separator '	' "$DB_PATH" "
SELECT id, path FROM jobs
WHERE status = 'done'
  AND (character_tag_string IS NULL OR TRIM(character_tag_string) = '')
  AND NOT EXISTS (SELECT 1 FROM job_tags t WHERE t.job_id = jobs.id AND t.tag LIKE 'character:%');
"
	)
	while IFS=$'\t' read -r job_id fpath; do
		[[ -z "${job_id:-}" ]] && continue
		process_row "$job_id" "$fpath" "$UNCAT"
	done < <(printf '%s\n' "${uncat_rows[@]:-}")
fi

echo "done: $LINKS symlink(s) under $(printf '%q' "$OUT_ROOT") (db $(printf '%q' "$DB_PATH")${MEDIA_ROOT_CANON:+; only under $(printf '%q' "$MEDIA_ROOT_CANON")})"
