from pathlib import Path
import json
import sys
import subprocess
from tqdm import tqdm
import torch
import re
import hashlib
import random

# --------------------------
# 检查 accelerate
# --------------------------
try:
    import accelerate
except ImportError:
    print("Installing accelerate...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate"])
    import accelerate

# --------------------------
# 选择 backend
# --------------------------
try:
    from modelscope import AutoModelForCausalLM, AutoTokenizer
    USE_MODELSCOPE = True
    print("Using ModelScope backend")
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    USE_MODELSCOPE = False
    print("Using Transformers backend")

# --------------------------
# 配置
# --------------------------
MODEL_ID = 'Qwen/Qwen3-8B'
DATA_ROOT = Path('datasets/temporal_spatial_audio_text/stClotho/metadata')
OUT_ROOT = Path('datasets/temporal_spatial_audio_text/stClotho/metadata_qwen3-8B')
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 批量设置（全量生成建议按批次处理，避免一次性把所有 prompts 放进显存/内存）
# FILE_BATCH_SIZE 控制每批处理多少个 metadata 文件（每个文件会产生 NUM_CAPTIONS 条 prompt）
# PROMPT_BATCH_SIZE 控制每次 model.generate 处理多少条 prompt
FILE_BATCH_SIZE = 200
PROMPT_BATCH_SIZE = 64
MAX_NEW_TOKENS = 60
NUM_CAPTIONS = 5  # 每个事件有5组spatialized_caption

# “略多样但仍稳定”：通过固定连接词轮换提供多样性；生成使用非采样beam search保证稳定
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

# 尽量阻止模型输出元文本/自言自语
BAD_WORD_STRINGS = [
    "Sound 1", "Sound 2", "sound 1", "sound 2",
    "Output", "Output:", "Example", "Now merge",
    "Okay", "OK", "Alright", "Let's", "I need", "Here is", "Here's",
    "merged:", "Combine:", "Description:", "Result:",
]

# --------------------------
# 辅助函数
# --------------------------
def extract_direction_words(text: str) -> set[str]:
    lower = (text or "").lower()
    found: set[str] = set()
    for d in DIRECTION_WORDS:
        if re.search(rf"\b{re.escape(d)}\b", lower):
            found.add(d)
    return found


def _stable_rng(key: str) -> random.Random:
    seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    return random.Random(seed)


def pick_connectors_for_file(filename: str, k: int) -> list[str]:
    rng = _stable_rng(filename)
    if len(CONNECTORS) >= k:
        return rng.sample(CONNECTORS, k=k)
    return [rng.choice(CONNECTORS) for _ in range(k)]


def build_prompt(s1: str, s2: str, connector: str) -> str:
    # 只给最小指令 + 原句，避免模型模仿“Example/Output/Sound 1”等元格式
    return (
        "Task: Write ONE natural English sentence combining two sequential sound descriptions.\n"
        "Rules:\n"
        "- Sound 1 happens first, Sound 2 happens next.\n"
        f"- Use the connector phrase exactly once: \"{connector}\".\n"
        "- Keep the direction words exactly as written (do not replace with synonyms).\n"
        "- Do NOT add labels, commentary, thoughts, or extra sentences.\n"
        "- Output ONLY the final sentence.\n\n"
        f"Sentence 1: {s1}\n"
        f"Sentence 2: {s2}\n"
        "Final sentence:"
    )

def prepare_prompts_paired(metafile):
    """从 stClotho 的 metadata 准备配对的 prompts (5组, 使用 spatialized_caption)"""
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

    # 期望每个事件都有 NUM_CAPTIONS 组（i 对 i）；不足则跳过，避免后续越界
    if len(first_spatialized) < NUM_CAPTIONS or len(second_spatialized) < NUM_CAPTIONS:
        return None
    
    chosen_connectors = pick_connectors_for_file(metafile.name, k=NUM_CAPTIONS)

    # 生成配对的prompts
    prompt_pairs = []
    for i in range(NUM_CAPTIONS):
        s1 = str(first_spatialized[i]).strip()
        s2 = str(second_spatialized[i]).strip()
        connector = chosen_connectors[i]
        needed_dirs = extract_direction_words(s1) | extract_direction_words(s2)

        prompt_pairs.append({
            'prompt': build_prompt(s1, s2, connector),
            'connector': connector,
            'needed_dirs': sorted(list(needed_dirs)),
            'first_spatialized': s1,
            'second_spatialized': s2,
        })
    
    return {
        'metafile': metafile,
        'prompt_pairs': prompt_pairs,
        'first_audio': data['audio_segments'][0]['filename'],
        'second_audio': data['audio_segments'][1]['filename'],
        'original_data': data
    }

def clean_generated(text):
    """清理生成的文本 - 更严格的清理"""
    text = (text or "").strip()

    # 只取第一行，减少跑偏内容
    text = text.splitlines()[0].strip() if text else text
    
    # 移除常见的前缀
    prefixes = ["Combined description:", "Description:", "Output:", "Answer:"]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    # 移除引号
    text = text.strip('"\'')
    
    # 移除括号内的内容（如模型的自我解释）
    text = re.sub(r'\([^)]*\)', '', text)
    
    # 移除换行符
    text = text.replace('\n', ' ')
    
    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 截断为第一句
    m = re.search(r'^(.+?[.!?])\s', text)
    if m:
        text = m.group(1).strip()
    else:
        if text and text[-1] not in '.!?':
            text = text + '.'
    
    # 移除"Okay"、"Let's see"等自言自语
    self_talk_patterns = [
        r'^(Okay|OK|Well|So|Now|Let\'s see|Here|Alright)[,\s]+',
        r'\s+(Okay|OK|Well|So|Now|Let\'s see)[,\.\s]*$'
    ]
    for pattern in self_talk_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    text = text.strip()
    
    return text


def build_bad_words_ids(tokenizer):
    bad = []
    for w in BAD_WORD_STRINGS:
        ids = tokenizer(w, add_special_tokens=False).input_ids
        if ids:
            bad.append(ids)
    return bad


def _connector_pattern(connector: str) -> re.Pattern:
    # 以“独立短语”的方式匹配连接词，避免子串误判（例如 connector=then 匹配到 thenx）
    c = (connector or "").strip().lower()
    return re.compile(rf"(?<!\\w){re.escape(c)}(?!\\w)")


def is_valid_output(text: str, connector: str, needed_dirs: list[str]) -> bool:
    if not text:
        return False

    low = text.lower()
    banned_re = r"\b(sound\s*\d+|output|example|now merge|i need|let's|okay|alright|here's|here is|merged)\b"
    if re.search(banned_re, low):
        return False

    pat = _connector_pattern(connector)
    if pat.search(low) is None:
        return False
    if len(pat.findall(low)) != 1:
        return False

    for d in needed_dirs:
        if re.search(rf"\b{re.escape(d)}\b", low) is None:
            return False

    if len(text.split()) > 40:
        return False

    return True


def fallback_template(s1: str, s2: str, connector: str) -> str:
    a = s1.strip().rstrip('.!?')
    b = s2.strip()
    out = f"{a}, {connector} {b}"
    out = re.sub(r"\s+", " ", out).strip()
    if out and out[-1] not in '.!?':
        out += '.'
    return out


def iter_batches(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

# --------------------------
# 加载模型
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model {MODEL_ID} on {device}...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=torch.bfloat16 if device=="cuda" else torch.float32,
    low_cpu_mem_usage=True
).eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bad_words_ids = build_bad_words_ids(tokenizer)

# --------------------------
# 批量处理
# --------------------------
splits = ['train', 'valid', 'test']

for split in splits:
    split_dir = DATA_ROOT / split
    if not split_dir.exists():
        print(f"Warning: {split} directory not found: {split_dir}")
        continue

    metafiles = sorted(list(split_dir.glob('*.json')))
    print(f"\n{'='*70}")
    print(f"Processing {split} split (FULL): {len(metafiles)} files")
    print(f"{'='*70}")

    out_split_dir = OUT_ROOT / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    # 分批处理文件，避免一次性构建巨量 prompts
    file_pbar = tqdm(
        total=len(metafiles),
        desc=f"{split}: files",
        unit="file",
        dynamic_ncols=True,
        mininterval=0.5,
    )

    for mf_batch in iter_batches(metafiles, FILE_BATCH_SIZE):
        # 该批次内扁平化 prompts
        all_prompts = []
        file_indices = []
        caption_indices = []
        all_file_data = []

        for mf in mf_batch:
            file_data = prepare_prompts_paired(mf)
            if file_data is None:
                file_pbar.update(1)
                continue
            file_idx = len(all_file_data)
            all_file_data.append(file_data)
            for caption_idx, pair in enumerate(file_data['prompt_pairs']):
                all_prompts.append(pair['prompt'])
                file_indices.append(file_idx)
                caption_indices.append(caption_idx)
            file_pbar.update(1)

        if not all_prompts:
            continue

        # 分批推理 prompts，避免 OOM
        all_generated = [""] * len(all_prompts)
        prompt_pbar = tqdm(
            total=len(all_prompts),
            desc=f"{split}: prompts",
            unit="prompt",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
        )

        for start in range(0, len(all_prompts), PROMPT_BATCH_SIZE):
            batch_prompts = all_prompts[start:start + PROMPT_BATCH_SIZE]
            inputs = tokenizer(
                batch_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=4,
                    repetition_penalty=1.05,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    bad_words_ids=bad_words_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_texts = tokenizer.batch_decode(
                outputs[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            for offset, raw in enumerate(generated_texts):
                idx = start + offset
                cleaned = clean_generated(raw)
                f_idx = file_indices[idx]
                c_idx = caption_indices[idx]
                pair = all_file_data[f_idx]['prompt_pairs'][c_idx]
                connector = pair['connector']
                needed_dirs = pair['needed_dirs']
                s1 = pair['first_spatialized']
                s2 = pair['second_spatialized']
                if is_valid_output(cleaned, connector=connector, needed_dirs=needed_dirs):
                    all_generated[idx] = cleaned
                else:
                    all_generated[idx] = fallback_template(s1, s2, connector)

            prompt_pbar.update(len(batch_prompts))

        prompt_pbar.close()

        # 写回该批次文件
        # 初始化每个文件的 5 条
        file_out_captions = [[None] * NUM_CAPTIONS for _ in range(len(all_file_data))]
        for idx, text in enumerate(all_generated):
            f_idx = file_indices[idx]
            c_idx = caption_indices[idx]
            file_out_captions[f_idx][c_idx] = text

        for f_idx, file_data in enumerate(all_file_data):
            temporal_spatial_captions = file_out_captions[f_idx]
            # 兜底：如果有 None，补模板
            for i in range(NUM_CAPTIONS):
                if temporal_spatial_captions[i] is None:
                    pair = file_data['prompt_pairs'][i]
                    temporal_spatial_captions[i] = fallback_template(
                        pair['first_spatialized'],
                        pair['second_spatialized'],
                        pair['connector'],
                    )

            new_data = file_data['original_data'].copy()
            new_data['temporal_spatial_caption'] = temporal_spatial_captions
            new_data['generation_info'] = {
                'model': MODEL_ID,
                'first_audio': file_data['first_audio'],
                'second_audio': file_data['second_audio'],
                'num_captions': len(temporal_spatial_captions),
                'decode': {
                    'do_sample': False,
                    'num_beams': 4,
                    'repetition_penalty': 1.05,
                    'no_repeat_ngram_size': 3,
                    'max_new_tokens': MAX_NEW_TOKENS,
                },
                'connectors': CONNECTORS,
                'note': 'Stable: deterministic decode + bad-words + direction check; Diverse: per-file stable random connectors; Source: spatialized_caption'
            }

            out_file = out_split_dir / file_data['metafile'].name
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)

    file_pbar.close()

print("\n" + "="*70)
print("All done!")
print(f"Output directory: {OUT_ROOT}")
print("="*70)