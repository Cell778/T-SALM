import os
import json
import glob
import math
import argparse
from typing import Union

"""Add missing 'direction' field to spatial metadata JSON files.

Mapping uses azi (azimuth in degrees) to one of 8 directions matching
the order used in sCLAPDataset.direction_label_dict:
east, northeast, north, northwest, west, southwest, south, southeast.

Resulting text format must be exactly: 'The sound is coming from the <dir>.'
so downstream label lookup remains consistent.
"""


def normalize_azi(azi: float) -> float:
    """Wrap azimuth to [-180, 180)."""
    return ((azi + 180.0) % 360.0) - 180.0


def map_direction(azi_deg: float) -> str:
    a = normalize_azi(azi_deg)
    # Interval boundaries chosen to split circle into 8 equal 45° sectors
    if -22.5 < a <= 22.5:
        return 'east'
    if 22.5 < a <= 67.5:
        return 'northeast'
    if 67.5 < a <= 112.5:
        return 'north'
    if 112.5 < a <= 157.5:
        return 'northwest'
    if a > 157.5 or a <= -157.5:
        return 'west'
    if -157.5 < a <= -112.5:
        return 'southwest'
    if -112.5 < a <= -67.5:
        return 'south'
    # -67.5 < a <= -22.5
    return 'southeast'


def parse_numeric(v: Union[str, float, int, list, tuple]) -> float:
    """Best-effort conversion of various representations to float."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)) and v:
        return parse_numeric(v[0])
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return 0.0
        # Strip common bracket wrappers like '[30.0]'
        if s[0] == '[' and s[-1] == ']':
            s = s[1:-1].strip()
        try:
            return float(s)
        except ValueError:
            try:
                import ast
                obj = ast.literal_eval(s)
                return parse_numeric(obj)
            except Exception:
                raise ValueError(f'Cannot parse azimuth value: {v}')
    raise ValueError(f'Unsupported azimuth type: {type(v)}')


def gather_files(root: str, glob_pattern: str = None) -> list:
    """Collect metadata json files.
    If glob_pattern is provided, it is joined to root directly (supports '**').
    Otherwise default pattern searches any 'metadata/*.json' recursively under root.
    """
    if glob_pattern:
        pattern = os.path.join(root, glob_pattern)
    else:
        pattern = os.path.join(root, '**', 'metadata', '*.json')
    return glob.glob(pattern, recursive=True)


def extract_direction_from_caption(spatialized_caption) -> str | None:
    """Extract direction token from spatialized caption text.
    Prioritizes longer tokens (northeast) to avoid prefix collisions.
    Returns canonical direction word or None.
    """
    if spatialized_caption is None:
        return None
    if isinstance(spatialized_caption, list):
        texts = spatialized_caption
    else:
        texts = [spatialized_caption]
    tokens = [
        'northeast', 'northwest', 'southeast', 'southwest',
        'north', 'south', 'east', 'west'
    ]
    for t in texts:
        low = t.lower()
        for tok in tokens:
            if tok in low:
                return tok
    return None


def process(root: str, dry_run: bool = False, glob_pattern: str = None, force: bool = False) -> int:
    files = gather_files(root, glob_pattern)
    changed = 0
    for fp in files:
        try:
            with open(fp, 'r') as rf:
                data = json.load(rf)
        except Exception as e:
            print(f'[SKIP] {fp}: cannot load ({e})')
            continue

        if 'direction' in data and not force:
            continue  # Already has the field and we are not forcing overwrite
        # 1) Try from spatialized_caption
        dir_word = None
        if 'spatialized_caption' in data:
            dir_word = extract_direction_from_caption(data['spatialized_caption'])
        # 2) Fallback to azimuth
        if dir_word is None and 'azi' in data:
            try:
                azi_val = parse_numeric(data['azi'])
                dir_word = map_direction(azi_val)
            except Exception as e:
                print(f'[WARN] {fp}: cannot parse azi ({e})')
        if dir_word is None:
            print(f'[SKIP] {fp}: no direction source (caption/azi)')
            continue

        new_direction_text = f'The sound is coming from the {dir_word}.'
        if 'direction' in data and force and data['direction'] != new_direction_text:
            print(f'[FORCE] {fp}: overwrite direction "{data["direction"]}" -> "{new_direction_text}"')
        data['direction'] = new_direction_text
        if dry_run:
            print(f'[DRY] Would add direction={data["direction"]} to {fp}')
            changed += 1
            continue
        try:
            with open(fp, 'w') as wf:
                json.dump(data, wf, ensure_ascii=False)
            changed += 1
        except Exception as e:
            print(f'[SKIP] {fp}: cannot write ({e})')
            continue
    return changed


def main():
    ap = argparse.ArgumentParser(description='Add missing direction field to spatial metadata JSONs.')
    ap.add_argument('--root', default='datasets', help='Root datasets directory or absolute path to dataset')
    ap.add_argument('--glob', default=None, help="Optional glob pattern relative to root (e.g. 'audio_text/sAudioCaps/metadata/train/*.json')")
    ap.add_argument('--dry-run', action='store_true', help='Only report changes without writing')
    ap.add_argument('--force', action='store_true', help='Overwrite existing direction by re-extracting from caption/azi')
    args = ap.parse_args()
    total = process(args.root, dry_run=args.dry_run, glob_pattern=args.glob, force=args.force)
    print(f'[DONE] Added direction to {total} files.')


if __name__ == '__main__':
    main()
