#!/usr/bin/env python
"""Normalize caption fields in sAudioCaps-style metadata JSON files.

Transforms any JSON where `caption` or `spatialized_caption` are stored as a
single string into a single-element list. Backs up original content to
`caption_full` / `spatialized_caption_full` (if not already present).

Usage:
  python scripts/normalize_captions.py \
      --root /Users/you/Desktop/Dataset/spatial_audio_text/AudioCaps/metadata \
      --dry

Options:
  --root PATH     Root directory containing train/valid/test subfolders.
  --dry           Dry run; show what would change without writing.
  --overwrite     If backup keys already exist, still overwrite them.

Exit code 0 on success, >0 on errors.
"""

import argparse
import json
import sys
from pathlib import Path


def process_file(path: Path, dry: bool = False, overwrite: bool = False):
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return False, f"READ_FAIL:{path.name}:{e}"  # skip unreadable

    changed = False
    report = []

    # Helper to wrap a field
    def wrap_field(key: str, backup_key: str):
        nonlocal changed
        if key not in data:
            return
        val = data[key]
        # Already a list
        if isinstance(val, list):
            # Ensure backup exists
            if backup_key not in data or overwrite:
                if not dry:
                    data[backup_key] = val.copy()
                changed = True
                report.append(f"BACKUP_LIST:{key}")
            return
        # Not a list -> wrap if string
        if isinstance(val, str):
            if backup_key not in data or overwrite:
                if not dry:
                    data[backup_key] = val
            if not dry:
                data[key] = [val]
            changed = True
            report.append(f"WRAP_STR:{key}")
        else:
            report.append(f"SKIP_TYPE:{key}:{type(val).__name__}")

    wrap_field("caption", "caption_full")
    wrap_field("spatialized_caption", "spatialized_caption_full")

    if changed and not dry:
        try:
            path.write_text(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            return False, f"WRITE_FAIL:{path.name}:{e}"
    return True, ":".join(report) if report else "NO_CHANGE"


def main():
    parser = argparse.ArgumentParser(description="Normalize caption fields to lists.")
    parser.add_argument("--root", required=True, help="Root metadata dir containing train/valid/test")
    parser.add_argument("--dry", action="store_true", help="Dry run; do not modify files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing backup keys")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root not found: {root}", file=sys.stderr)
        return 2

    splits = ["train", "valid", "test"]
    total = 0
    changed = 0
    errors = 0

    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"[WARN] Split missing: {split_dir}")
            continue
        for json_file in split_dir.glob("*.json"):
            ok, msg = process_file(json_file, dry=args.dry, overwrite=args.overwrite)
            total += 1
            if not ok:
                errors += 1
                print(f"[ERROR] {json_file.name} {msg}")
            else:
                if msg.startswith("WRAP_STR") or "WRAP_STR" in msg or "BACKUP_LIST" in msg:
                    changed += 1
                if args.dry and ("WRAP_STR" in msg or "BACKUP_LIST" in msg):
                    print(f"[DRY] {json_file.name} {msg}")

    print(f"Processed: {total}, Changed: {changed}, Errors: {errors}, Dry: {args.dry}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
