from pathlib import Path
import json
from tqdm import tqdm
import re
import hashlib
import random

# --------------------------
# 配置
# --------------------------
DATA_ROOT = Path('datasets/temporal_spatial_audio_text/stClotho_negative/metadata')
OUT_ROOT = Path('datasets/temporal_spatial_audio_text/stClotho_negative/metadata_rule_concat')
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 批量设置
NUM_CAPTIONS = 5  # 每个事件有5组spatialized_caption

CONNECTORS = [
	"then",
	"next",
	"after that",
	"following that",
	"afterwards",
	"soon after",
	"shortly after",
	"soon thereafter",
	"moments later",
	"right after",
	"later",
	"a moment later",
	"a little later",
	"before long",
	"in the next moment",
	"in the following moment",
	"and then",
	"and later",
	"subsequently",
	"thereafter",
	"not long after",
	"immediately after",
	"directly after",
	"soon afterwards",
	"in quick succession",
]

DIRECTION_WORDS = {
	"north", "northeast", "east", "southeast",
	"south", "southwest", "west", "northwest",
}

BAD_WORD_STRINGS = [
	"Sound 1", "Sound 2", "sound 1", "sound 2",
	"Output", "Output:", "Example", "Now merge",
	"Okay", "OK", "Alright", "Let's", "I need", "Here is", "Here's",
	"merged:", "Combine:", "Description:", "Result:",
]


# --------------------------
# 辅助函数
# --------------------------
def _stable_rng(key: str) -> random.Random:
	seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
	return random.Random(seed)


def pick_connectors_for_file(filename: str, k: int) -> list[str]:
	rng = _stable_rng(filename)
	if len(CONNECTORS) >= k:
		return rng.sample(CONNECTORS, k=k)
	return [rng.choice(CONNECTORS) for _ in range(k)]



def fallback_template(s1: str, s2: str, connector: str) -> str:
	a = s1.strip().rstrip('.!?')
	b = s2.strip()
	out = f"{a}, {connector} {b}"
	out = re.sub(r"\s+", " ", out).strip()
	if out and out[-1] not in '.!?':
		out += '.'
	return out


def build_temporal_spatial_captions(metafile: Path) -> list[str] | None:
	"""从 metadata 里取两个事件的 spatialized_caption，按 i-to-i 用随机 connector 拼接生成 NUM_CAPTIONS 条。"""
	with open(metafile, 'r', encoding='utf-8') as f:
		data = json.load(f)

	if 'audio_segments' not in data or len(data['audio_segments']) < 2:
		return None

	first_seg = data['audio_segments'][0]['metadata']
	second_seg = data['audio_segments'][1]['metadata']

	first_spatialized = first_seg.get('spatialized_caption', [])
	second_spatialized = second_seg.get('spatialized_caption', [])
	if not isinstance(first_spatialized, list) or not isinstance(second_spatialized, list):
		return None
	if len(first_spatialized) == 0 or len(second_spatialized) == 0:
		return None
	if len(first_spatialized) < NUM_CAPTIONS or len(second_spatialized) < NUM_CAPTIONS:
		return None

	chosen_connectors = pick_connectors_for_file(metafile.name, k=NUM_CAPTIONS)
	outs: list[str] = []
	for i in range(NUM_CAPTIONS):
		s1 = str(first_spatialized[i]).strip()
		s2 = str(second_spatialized[i]).strip()
		connector = chosen_connectors[i]
		outs.append(fallback_template(s1, s2, connector))
	return outs


# --------------------------
# 批量处理
# --------------------------
splits = ['train', 'valid', 'test']

for split in splits:
	split_dir = DATA_ROOT / split
	if not split_dir.exists():
		print(f"Warning: {split} directory not found: {split_dir}")
		continue

	# IMPORTANT: negative metadata is under split/spatial/*.json and split/temporal/*.json
	metafiles = sorted([p for p in split_dir.rglob('*.json') if p.is_file()])
	print(f"\n{'='*70}")
	print(f"Processing {split} split (FULL): {len(metafiles)} files")
	print(f"{'='*70}")

	# Keep the original subfolder structure (e.g., test/spatial/*.json)
	file_pbar = tqdm(
		total=len(metafiles),
		desc=f"{split}: files",
		unit="file",
		dynamic_ncols=True,
		mininterval=0.5,
	)

	for mf in metafiles:
		outs = build_temporal_spatial_captions(mf)
		file_pbar.update(1)
		if outs is None:
			continue

		with open(mf, 'r', encoding='utf-8') as f:
			data = json.load(f)

		new_data = data.copy()
		new_data['temporal_spatial_caption'] = outs
		new_data['generation_info'] = {
			'method': 'rule_concat',
			'num_captions': len(outs),
			'connectors': CONNECTORS,
			'note': 'Per-file stable random connectors; Output is template concat of spatialized_caption[0] + connector + spatialized_caption[1] (i-to-i).'
		}

		rel = mf.relative_to(DATA_ROOT)
		out_file = OUT_ROOT / rel
		out_file.parent.mkdir(parents=True, exist_ok=True)
		with open(out_file, 'w', encoding='utf-8') as f:
			json.dump(new_data, f, indent=2, ensure_ascii=False)

	file_pbar.close()

print("\n" + "="*70)
print("All done!")
print(f"Output directory: {OUT_ROOT}")
print("="*70)

