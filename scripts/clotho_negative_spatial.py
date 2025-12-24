#!/usr/bin/env python3
"""
合并Clotho数据集中所有以_0结尾的空间音频文件，创建带有时序信息的音频序列。
同时合并对应的metadata。
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import soundfile as sf
import numpy as np
from tqdm import tqdm


def get_audio_files_ending_with_0(audio_dir: Path) -> List[Path]:
    """获取所有以_0.flac结尾的音频文件并排序"""
    audio_files = sorted([f for f in audio_dir.glob('*_0.flac')])
    return audio_files


def load_metadata(json_path: Path) -> Dict:
    """加载metadata JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def concatenate_audios(audio_files: List[Path]) -> Tuple[np.ndarray, int]:
    """
    按顺序连接音频文件，音频之间不重叠
    
    Returns:
        concatenated_audio: 合并后的音频数组
        sample_rate: 采样率
    """
    audio_segments = []
    sample_rate = None
    
    for audio_file in audio_files:
        audio, sr = sf.read(audio_file)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"采样率不一致: {audio_file} has {sr}, expected {sample_rate}")
        
        audio_segments.append(audio)
    
    # 连接所有音频段
    concatenated_audio = np.concatenate(audio_segments, axis=0)
    
    return concatenated_audio, sample_rate


def merge_metadata(json_files: List[Path]) -> Dict:
    """
    合并多个metadata文件
    
    Returns:
        merged_metadata: 包含所有音频metadata的字典
    """
    merged = {
        "audio_segments": [],
        "num_segments": len(json_files)
    }
    
    for json_file in json_files:
        metadata = load_metadata(json_file)
        merged["audio_segments"].append({
            "filename": json_file.stem,
            "metadata": metadata
        })
    
    return merged


def process_split(input_base: Path, output_base: Path, split: str, batch_size: int = 2):
    """
    处理一个split（train/test/valid）
    
    Args:
        input_base: 输入基础路径，例如 datasets/spatial_audio_text/Clotho
        output_base: 输出基础路径
        split: 'train', 'test', 或 'valid'
        batch_size: 每次合并的音频数量
    """
    # 设置路径
    audio_dir = input_base / 'audio' / split
    metadata_dir = input_base / 'metadata' / split
    
    output_audio_dir = output_base / 'audio' / split
    output_metadata_dir = output_base / 'metadata' / split
    
    # 创建输出目录
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有_0.flac文件
    audio_files = get_audio_files_ending_with_0(audio_dir)
    
    print(f"处理 {split} split: 找到 {len(audio_files)} 个以_0结尾的音频文件")
    
    # 按batch_size分组处理
    num_batches = len(audio_files) // batch_size
    output_count = 0

    for i in tqdm(range(num_batches), desc=f"Processing {split}"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_audio_0_files = audio_files[start_idx:end_idx]  # 全是 *_0.flac

        if len(batch_audio_0_files) != 2:
            raise ValueError(f"Batch {i} 尺寸异常，期望2，实际 {len(batch_audio_0_files)}")

        first_audio_0 = batch_audio_0_files[0]
        second_audio_0 = batch_audio_0_files[1]

        # 第一个音频改用其 *_2.flac
        if not first_audio_0.stem.endswith('_0'):
            raise ValueError(f"文件名不符合 *_0 约定: {first_audio_0}")
        first_stem_2 = first_audio_0.stem[:-1] + '2'  # 把 *_0 改成 *_2
        first_audio_2 = audio_dir / f"{first_stem_2}.flac"

        if not first_audio_2.exists():
            raise FileNotFoundError(f"Batch {i} 缺少第一个音频的 _2 文件: {first_audio_2}")

        # 第二个音频保持使用 *_0.flac (不需要改变)
        
        # 本批次的拼接音频顺序：[第一个的_2，第二个的_0]
        batch_audio_files = [first_audio_2, second_audio_0]

        # 对应的 metadata 也按 [*_2.json, *_0.json]
        batch_json_files = [
            metadata_dir / f"{first_stem_2}.json",
            metadata_dir / f"{second_audio_0.stem}.json",
        ]

        # 检查所有metadata文件是否存在（缺失则抛异常）
        if not all(json_file.exists() for json_file in batch_json_files):
            missing_files = [str(f) for f in batch_json_files if not f.exists()]
            raise FileNotFoundError(f"Batch {i} 中有metadata文件缺失: {missing_files}")

        # 合并音频（发生错误直接抛出，让外层感知失败）
        concatenated_audio, sample_rate = concatenate_audios(batch_audio_files)

        # 合并metadata
        merged_metadata = merge_metadata(batch_json_files)

        # 生成输出文件名
        output_name = f"spatial_negative_{output_count:06d}"
        output_audio_path = output_audio_dir / f"{output_name}.flac"
        output_json_path = output_metadata_dir / f"{output_name}.json"

        # 保存音频与metadata
        sf.write(output_audio_path, concatenated_audio, sample_rate)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(merged_metadata, f, indent=4, ensure_ascii=False)

        output_count += 1
    
    print(f"{split} split 完成: 生成了 {output_count} 个时序音频文件")


def main():
    """主函数"""
    # 输入路径
    clotho_base = Path('datasets/spatial_audio_text/Clotho')
    
    # 输出路径
    output_base = Path('/Users/cellren/Desktop/datasets/temporal_spatial_audio_text/stClotho_spatial_negative')
    
    # 确保输入路径存在
    if not clotho_base.exists():
        print(f"错误: 输入路径不存在: {clotho_base}")
        return
    
    # 处理所有splits
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        split_audio_dir = clotho_base / 'audio' / split
        if split_audio_dir.exists():
            print(f"\n{'='*60}")
            print(f"开始处理 {split} split")
            print(f"{'='*60}")
            process_split(clotho_base, output_base, split, batch_size=2)
        else:
            print(f"警告: {split} split 的音频目录不存在: {split_audio_dir}")
    
    print(f"\n{'='*60}")
    print("所有处理完成！")
    print(f"输出目录: {output_base}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()