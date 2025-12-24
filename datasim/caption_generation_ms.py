from pathlib import Path
import json
from tqdm import tqdm
import torch
import sys
import re

# --------------------------
# 检查 accelerate
# --------------------------
try:
    import accelerate
except ImportError:
    print("Installing accelerate...")
    import subprocess
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
DATA_ROOT = Path('datasets/spatial_audio_text/Clotho/metadata')
OUT_ROOT = Path('datasets/spatial_audio_text/Clotho/metadata_qwen3-8B')
OUT_ROOT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
MAX_NEW_TOKENS = 70  # 允许稍微长一点的输出
INSTRUCTION = ("Write one natural, concise English sentence describing the sound "
               "and its direction. Avoid repeating words or giving multiple versions.")

# --------------------------
# 辅助函数
# --------------------------
def build_direction(azi: float) -> str:
    if -22.5 < azi <= 22.5: return 'south'
    elif 22.5 < azi <= 67.5: return 'southeast'
    elif 67.5 < azi <= 112.5: return 'east'
    elif 112.5 < azi <= 157.5: return 'northeast'
    elif -22.5 > azi >= -67.5: return 'southwest'
    elif -67.5 > azi >= -112.5: return 'west'
    elif -112.5 > azi >= -157.5: return 'northwest'
    else: return 'north'

def prepare_prompts(metafile):
    """为单个 JSON 生成 direction 和 prompts"""
    with open(metafile, "r", encoding="utf-8") as f:
        js = json.load(f)

    azi = float(js.get('azi', 0))
    direction = build_direction(azi)
    js['direction'] = f"The sound is coming from the {direction}."

    prompts = []
    for cap in js.get("caption", []):
        prompts.append(f"{INSTRUCTION}\nSound: \"{cap}\", direction: {direction}.")
    return metafile, js, prompts

def clean_generated(text):
    """只保留第一句话，去掉重复和多余换行"""
    sentences = re.split(r'[.?!]', text)
    if sentences:
        return sentences[0].strip() + '.'
    return text.strip()

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

# --------------------------
# 批量处理
# --------------------------
metafiles = list(DATA_ROOT.rglob("*.json"))
print(f"Total {len(metafiles)} files to process")

metadata = [prepare_prompts(mf) for mf in metafiles]

pbar = tqdm(total=len(metadata), desc="Generating")

for i in range(0, len(metadata), BATCH_SIZE):
    batch = metadata[i:i+BATCH_SIZE]
    all_prompts = []
    for _, _, prompts in batch:
        all_prompts.extend(prompts)

    # 批量编码
    inputs = tokenizer(
        all_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,           # 允许采样，提高自然度
            top_p=0.9,                # 核采样
            temperature=0.2,          # 控制多样性
            repetition_penalty=1.2,   # 避免重复
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # 分配结果
    idx = 0
    for metafile, js, prompts in batch:
        num_caps = len(prompts)
        results = decoded[idx:idx+num_caps]
        idx += num_caps

        # 清理输出，只保留一句自然文本
        cleaned = [clean_generated(res.replace(pr, "").strip()) for res, pr in zip(results, prompts)]
        js['spatialized_caption'] = cleaned

        # 构建输出路径，保持目录结构
        rel_path = metafile.relative_to(DATA_ROOT)
        out_path = OUT_ROOT / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入新文件
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(js, f, indent=4, ensure_ascii=False)

        pbar.update(1)
        tqdm.write(f"Processed {out_path}")

pbar.close()
print("All done!")