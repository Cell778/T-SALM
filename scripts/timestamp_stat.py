from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa


def _iter_metadata_files(metadata_root: Path) -> list[tuple[Path, str, str]]:
    """Iterate metadata files and get corresponding audio filename and split."""
    results: list[tuple[Path, str, str]] = []
    for split in ("train", "test", "valid"):
        split_dir = metadata_root / split
        if not split_dir.exists():
            continue
        for path in split_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
                audio_name = path.stem
                results.append((path, audio_name, split))
    return results


def _load_metadata(path: Path) -> tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """Load metadata from JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "audio_segments" in data:
        return data.get("audio_segments"), data

    return None, None


def _extract_durations_and_audios(
    segments: Optional[List[Dict[str, Any]]],
) -> tuple[List[str], List[float]]:
    """Extract ori_audiofile names and their durations from segments."""
    audio_names: List[str] = []
    durations: List[float] = []

    if not segments:
        return audio_names, durations

    for segment in segments:
        metadata = segment.get("metadata")
        if not isinstance(metadata, dict):
            continue

        ori_audiofile = metadata.get("ori_audiofile")
        ori_audio_duration = metadata.get("ori_audio_duration")

        if ori_audiofile and ori_audio_duration is not None:
            audio_names.append(str(ori_audiofile))
            if isinstance(ori_audio_duration, list):
                durations.append(sum(ori_audio_duration))
            else:
                durations.append(float(ori_audio_duration))

    return audio_names, durations


def _get_spatial_audio_duration(audio_dir: Path, audio_name: str, meta_split: str) -> Optional[tuple[float, str, str]]:
    """Get duration of spatial audio file. Returns (duration, split, filename) or None.
    First tries to find in the same split as metadata, then in other splits.
    Tries both temporal/ and spatial/ subdirectories."""
    # Try the same split as metadata first
    for subdir in ("temporal", "spatial", ""):
        for ext in (".flac", ".wav"):
            if subdir:
                audio_path = audio_dir / meta_split / subdir / f"{audio_name}{ext}"
            else:
                audio_path = audio_dir / meta_split / f"{audio_name}{ext}"
            if audio_path.exists():
                return float(librosa.get_duration(path=str(audio_path))), meta_split, audio_path.name
    
    # Fall back to other splits
    for split in ("train", "test", "valid"):
        if split == meta_split:
            continue
        for subdir in ("temporal", "spatial", ""):
            for ext in (".flac", ".wav"):
                if subdir:
                    audio_path = audio_dir / split / subdir / f"{audio_name}{ext}"
                else:
                    audio_path = audio_dir / split / f"{audio_name}{ext}"
                if audio_path.exists():
                    return float(librosa.get_duration(path=str(audio_path))), split, audio_path.name
    return None


def generate_statistics(
    metadata_root: Path,
    audio_dir: Path,
    output_file: Path,
) -> None:
    """Generate statistics CSV."""
    rows: List[Dict[str, Any]] = []

    meta_files = _iter_metadata_files(metadata_root)
    for meta_path, audio_name, meta_split in meta_files:
        segments, _ = _load_metadata(meta_path)
        audio_names, durations = _extract_durations_and_audios(segments)

        if not durations or len(durations) != 2:
            continue

        # Use both audios' durations
        audio1_name = audio_names[0]
        audio1_duration = durations[0]
        audio2_name = audio_names[1]
        audio2_duration = durations[1]
        total_duration = audio1_duration + audio2_duration

        # Get spatial audio duration
        result = _get_spatial_audio_duration(audio_dir, audio_name, meta_split)
        if result is None:
            continue
        spatial_duration, found_split, spatial_filename = result

        # Calculate difference
        diff = abs(total_duration - spatial_duration)

        rows.append({
            "spatial_audio_filename": spatial_filename,
            "audio1_name": audio1_name,
            "audio1_duration": round(audio1_duration, 4),
            "audio2_name": audio2_name,
            "audio2_duration": round(audio2_duration, 4),
            "ori_total_duration": round(total_duration, 4),
            "spatial_audio_duration": round(spatial_duration, 4),
            "difference": round(diff, 4),
        })

    # Write CSV
    if rows:
        with output_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "spatial_audio_filename",
                    "audio1_name",
                    "audio1_duration",
                    "audio2_name",
                    "audio2_duration",
                    "ori_total_duration",
                    "spatial_audio_duration",
                    "difference",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Statistics saved to {output_file}")
        print(f"Total records: {len(rows)}")
    else:
        print("No valid records found")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate statistics for stClotho temporal audio.",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path("/Users/cellren/Desktop/datasets/temporal_spatial_audio_text/stClotho_negative/metadata"),
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("/Users/cellren/Desktop/datasets/temporal_spatial_audio_text/stClotho_negative/audio"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.output is None:
        args.output = args.metadata_root / "stClotho_statistics.csv"
    generate_statistics(args.metadata_root, args.audio_dir, args.output)
