import argparse, json, shutil
from pathlib import Path
import soundfile as sf

SPLIT_MAP = {
    "train": ("AudioCaps_FOA_train", "train.json"),
    "valid": ("AudioCaps_FOA_val",   "val.json"),
    "test":  ("AudioCaps_FOA_test",  "test.json"),
}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_records(meta_path: Path):
    with open(meta_path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        for k in ["data","samples","items","records"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        # 若是 dict 但没有常见键，尝试把 dict 的 values 当作样本
        if all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())
        raise ValueError(f"无法识别 JSON 结构: {meta_path}")
    elif isinstance(obj, list):
        return obj
    else:
        raise ValueError(f"不支持的 JSON 顶层类型: {type(obj)}")

def pick_name(rec, name_key):
    # 优先用用户指定的 name_key
    if name_key and name_key in rec:
        return Path(str(rec[name_key])).name
    # 常见字段猜测
    for k in ["filename","file","audio","audio_name","audio_path","name","basename","utt_id","uid","id"]:
        if k in rec:
            return Path(str(rec[k])).name
    # 最后尝试从 caption 中提取（不推荐）
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="AudioCaps_FOA 根目录，例如 /Users/.../AudioCaps_FOA")
    ap.add_argument("--dst", required=True, help="作为 paths.dataset_dir 的根，例如 /Users/.../Dataset")
    ap.add_argument("--name-key", default=None, help="JSON 中指向音频文件名的键，例如 filename 或 audio_name")
    ap.add_argument("--copy-audio", action="store_true", help="默认转码为 flac；指定该项则直接复制已有的 .flac（若是 .wav 会转码）")
    args = ap.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()
    out_base = dst_root / "spatial_audio_text" / "AudioCaps"

    for split, (split_dirname, meta_name) in SPLIT_MAP.items():
        audio_src = src_root / split_dirname / "seld_audiocaps_foa" / "foa"
        meta_src  = src_root / split_dirname / "SpatialCaps" / meta_name

        assert audio_src.is_dir(), f"缺少目录: {audio_src}"
        assert meta_src.is_file(), f"缺少标注: {meta_src}"

        out_audio = out_base / "audio" / split
        out_meta  = out_base / "metadata" / split
        ensure_dir(out_audio); ensure_dir(out_meta)

        # 收集音频
        audio_files = sorted(list(audio_src.glob("*.flac")) + list(audio_src.glob("*.wav")))
        assert audio_files, f"{audio_src} 下未发现音频（支持 .flac/.wav）"
        stem2in = {p.stem: p for p in audio_files}

        # 读取并拆分元数据
        records = load_records(meta_src)
        print(f"[{split}] records: {len(records)}, audios: {len(audio_files)}")

        # 批量处理
        matched = 0
        for rec in records:
            name = pick_name(rec, args.name_key)
            if not name:
                raise ValueError(f"[{split}] 无法从记录中识别音频名，请用 --name-key 指定。示例记录键: {list(rec.keys())[:10]}")
            stem = Path(name).stem
            # 常见：AudioCaps 用 'Y' 前缀
            if stem not in stem2in and (not stem.startswith("Y")):
                if ("Y"+stem) in stem2in:
                    stem = "Y"+stem
            # 有时 json 里带扩展名不一致
            if stem not in stem2in:
                # 尝试模糊匹配
                candidates = [s for s in stem2in.keys() if s.endswith(stem) or stem.endswith(s)]
                if len(candidates) == 1:
                    stem = candidates[0]
            if stem not in stem2in:
                # 跳过无法匹配的项
                continue
            in_wav = stem2in[stem]
            out_flac = out_audio / f"{stem}.flac"

            # 音频：若已是 flac 且选择复制，则直接复制；否则用 soundfile 转为 flac
            if in_wav.suffix.lower() == ".flac" and args.copy_audio:
                if not out_flac.exists():
                    shutil.copy2(in_wav, out_flac)
            else:
                if not out_flac.exists():
                    data, sr = sf.read(in_wav, always_2d=False)
                    sf.write(out_flac, data, sr, format="FLAC")

            # 元数据：写入与音频同名的 JSON（只保留常用字段，存在才写）
            keep = {}
            for k in ["caption","captions","text","spatialized_caption","spatial_caption","azi","azimuth","ele","elevation","direction"]:
                if k in rec:
                    keep[k] = rec[k]
            # 规范化部分字段名
            if "captions" in keep and "caption" not in keep:
                keep["caption"] = keep.pop("captions")
            if "spatial_caption" in keep and "spatialized_caption" not in keep:
                keep["spatialized_caption"] = keep.pop("spatial_caption")
            if "azimuth" in keep and "azi" not in keep:
                keep["azi"] = keep.pop("azimuth")
            if "elevation" in keep and "ele" not in keep:
                keep["ele"] = keep.pop("elevation")
            out_meta_file = out_meta / f"{stem}.json"
            with open(out_meta_file, "w") as f:
                json.dump(keep, f, ensure_ascii=False)
            matched += 1

        print(f"[{split}] 成功配对并写出: {matched} 条")
        # 简单一致性检查
        out_audios = sorted(out_audio.glob("*.flac"))
        out_metas  = sorted(out_meta.glob("*.json"))
        assert len(out_audios) >= 1, f"{split} 输出无音频"
        assert len(out_metas) >= 1, f"{split} 输出无元数据"
        # 可不完全相等，但至少每个元数据都有音频
        a_stems = {p.stem for p in out_audios}
        m_stems = {p.stem for p in out_metas}
        inter = a_stems & m_stems
        assert len(inter) >= 1, f"{split} 音频与元数据未对齐"

    print("完成。目标数据根：", out_base)

if __name__ == "__main__":
    main()