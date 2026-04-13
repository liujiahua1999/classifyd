#!/usr/bin/env python3
"""
Minimal WD14 tagger helper for classifyd.
Reads frame image paths from stdin (one per line), runs WD14 ONNX inference,
prints JSON result to stdout.

Usage:
    echo -e "/tmp/f1.jpg\n/tmp/f2.jpg" | python3 wd14_tagger.py [--model SwinV2_v3] [--general-threshold 0.15] [--character-threshold 0.45]

Requires: pip install dghs-imgutils[onnxruntime] onnxruntime
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="SwinV2_v3")
    ap.add_argument("--general-threshold", type=float, default=0.15)
    ap.add_argument("--character-threshold", type=float, default=0.45)
    ap.add_argument("--min-score", type=float, default=0.9)
    args = ap.parse_args()

    paths = [line.strip() for line in sys.stdin if line.strip()]
    if not paths:
        json.dump({"error": "no frame paths on stdin"}, sys.stdout)
        sys.stdout.write("\n")
        return

    try:
        from imgutils.tagging import get_wd14_tags
    except ImportError:
        json.dump({"error": "imgutils not installed: pip install dghs-imgutils[onnxruntime]"}, sys.stdout)
        sys.stdout.write("\n")
        return

    rating_max = {}
    gen_max = {}
    gen_hits = {}
    char_max = {}
    char_hits = {}

    for fp in paths:
        try:
            rating, general, character = get_wd14_tags(
                fp,
                model_name=args.model,
                general_threshold=args.general_threshold,
                character_threshold=args.character_threshold,
            )
        except Exception as e:
            sys.stderr.write(f"wd14 frame error {fp}: {e}\n")
            continue
        for k, v in (rating or {}).items():
            rating_max[k] = max(rating_max.get(k, 0.0), float(v))
        for k, v in (general or {}).items():
            gen_max[k] = max(gen_max.get(k, 0.0), float(v))
            gen_hits[k] = gen_hits.get(k, 0) + 1
        for k, v in (character or {}).items():
            char_max[k] = max(char_max.get(k, 0.0), float(v))
            char_hits[k] = char_hits.get(k, 0) + 1

    ms = args.min_score

    def tag_list(scores, hits):
        out = []
        for name, score in sorted(scores.items(), key=lambda x: -x[1]):
            if score <= ms:
                continue
            row = {"name": name, "score": round(score, 4)}
            if name in hits:
                row["frames"] = hits[name]
            out.append(row)
        return out

    rating_hi = {k: round(v, 4) for k, v in rating_max.items() if v > ms}
    best_rating = max(rating_max.items(), key=lambda kv: kv[1])[0] if rating_max else None

    gen_tags = tag_list(gen_max, gen_hits)
    char_tags = tag_list(char_max, char_hits)

    def danbooru(tags):
        return ", ".join(t["name"].replace(" ", "_") for t in tags)

    result = {
        "model": args.model,
        "frames_tagged": len(paths),
        "dominant_rating": best_rating,
        "rating": rating_hi,
        "general_tag_string": danbooru(gen_tags),
        "character_tag_string": danbooru(char_tags),
        "general_tags": gen_tags,
        "character_tags": char_tags,
    }
    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
