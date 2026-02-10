from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import librosa


def _iter_metadata_files(metadata_root: Path) -> Iterable[Path]:
	for split in ("train", "test", "valid"):
		split_dir = metadata_root / split
		if not split_dir.exists():
			continue
		for path in split_dir.rglob("*"):
			if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
				yield path


def _load_metadata(path: Path) -> Tuple[List[Dict[str, Any]], str, Optional[Dict[str, Any]]]:
	if path.suffix.lower() == ".jsonl":
		items: List[Dict[str, Any]] = []
		with path.open("r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				items.append(json.loads(line))
		return items, "jsonl", None

	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	if isinstance(data, list):
		return data, "json", None

	if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
		return data["data"], "json:dict", data

	if isinstance(data, dict) and "audio_segments" in data and isinstance(data["audio_segments"], list):
		return data["audio_segments"], "json:audio_segments", data

	raise ValueError(f"Unsupported JSON structure in {path}")


def _save_metadata(
	path: Path,
	items: List[Dict[str, Any]],
	fmt: str,
	container: Optional[Dict[str, Any]] = None,
) -> None:
	backup = path.with_suffix(path.suffix + ".bak")
	if not backup.exists():
		backup.write_bytes(path.read_bytes())

	if fmt == "jsonl":
		with path.open("w", encoding="utf-8") as f:
			for item in items:
				f.write(json.dumps(item, ensure_ascii=False) + "\n")
		return

	if fmt == "json":
		with path.open("w", encoding="utf-8") as f:
			json.dump(items, f, ensure_ascii=False, indent=2)
		return

	if fmt == "json:dict":
		original = container or {}
		original["data"] = items
		with path.open("w", encoding="utf-8") as f:
			json.dump(original, f, ensure_ascii=False, indent=2)
		return

	if fmt == "json:audio_segments":
		original = container or {}
		original["audio_segments"] = items
		with path.open("w", encoding="utf-8") as f:
			json.dump(original, f, ensure_ascii=False, indent=2)
		return

	raise ValueError(f"Unsupported format {fmt}")


def _build_clotho_index(clotho_root: Path) -> Dict[str, Path]:
	index: Dict[str, Path] = {}
	for split in ("development", "evaluation", "validation"):
		split_dir = clotho_root / split
		if not split_dir.exists():
			continue
		for wav in split_dir.rglob("*.wav"):
			index[wav.name] = wav
	return index


def _get_duration(path: Path, cache: Dict[Path, float]) -> float:
	if path in cache:
		return cache[path]
	duration = float(librosa.get_duration(path=str(path)))
	cache[path] = duration
	return duration


def _resolve_audio_files(
	ori_audiofile: Any,
	clotho_index: Dict[str, Path],
	missing: List[str],
) -> List[Path]:
	names: List[str]
	if isinstance(ori_audiofile, list):
		names = [str(x) for x in ori_audiofile]
	else:
		names = [str(ori_audiofile)]

	paths: List[Path] = []
	for name in names:
		if name in clotho_index:
			paths.append(clotho_index[name])
		else:
			missing.append(name)
	return paths


def add_duration_to_metadata(metadata_root: Path, clotho_root: Path) -> None:
	clotho_index = _build_clotho_index(clotho_root)
	duration_cache: Dict[Path, float] = {}

	for meta_path in _iter_metadata_files(metadata_root):
		items, fmt, container = _load_metadata(meta_path)
		missing_names: List[str] = []

		for item in items:
			if fmt == "json:audio_segments":
				metadata = item.get("metadata") if isinstance(item, dict) else None
				if not isinstance(metadata, dict):
					continue
				ori_audiofile = metadata.get("ori_audiofile")
			else:
				ori_audiofile = item.get("ori_audiofile")

			if ori_audiofile is None:
				continue

			paths = _resolve_audio_files(ori_audiofile, clotho_index, missing_names)
			durations: List[Optional[float]] = []
			for p in paths:
				durations.append(_get_duration(p, duration_cache))

			if fmt == "json:audio_segments":
				if isinstance(ori_audiofile, list):
					metadata["ori_audio_duration"] = durations
				else:
					metadata["ori_audio_duration"] = durations[0] if durations else None
			else:
				if isinstance(ori_audiofile, list):
					item["ori_audio_duration"] = durations
				else:
					item["ori_audio_duration"] = durations[0] if durations else None

		_save_metadata(meta_path, items, fmt, container)

		if missing_names:
			miss_log = meta_path.with_suffix(meta_path.suffix + ".missing.txt")
			miss_log.write_text("\n".join(sorted(set(missing_names))), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Add original audio duration from Clotho to stClotho metadata.",
	)
	parser.add_argument(
		"--metadata-root",
		type=Path,
		default=Path("/Users/cellren/Desktop/datasets/temporal_spatial_audio_text/stClotho_negative/metadata"),
	)
	parser.add_argument(
		"--clotho-root",
		type=Path,
		default=Path("/Users/cellren/Desktop/datasets/Clotho"),
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = _parse_args()
	add_duration_to_metadata(args.metadata_root, args.clotho_root)
