#!/usr/bin/env python3
"""
随机合并Clotho数据集中的空间音频文件，创建带有时序信息的音频序列。
同时生成时序转换和空间转换的负样本。
"""

import json
import random
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


def save_audio_and_metadata(audio: np.ndarray, sample_rate: int, 
                            metadata: Dict, output_path: Path, 
                            json_path: Path):
    """保存音频和metadata"""
    sf.write(output_path, audio, sample_rate)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def process_split(input_base: Path, positive_output_base: Path, 
                 negative_output_base: Path, split: str, 
                 num_samples: int = 10000, seed: int = 42):
    """
    处理一个split（train/test/valid）
    
    Args:
        input_base: 输入基础路径
        positive_output_base: 正样本输出基础路径
        negative_output_base: 负样本输出基础路径
        split: 'train', 'test', 或 'valid'
        num_samples: 生成样本数量
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    
    # 设置路径
    audio_dir = input_base / 'audio' / split
    metadata_dir = input_base / 'metadata' / split
    
    # 正样本输出路径
    positive_audio_dir = positive_output_base / 'audio' / split
    positive_metadata_dir = positive_output_base / 'metadata' / split
    
    # 负样本输出路径
    neg_temporal_audio_dir = negative_output_base / 'audio' / split / 'temporal'
    neg_temporal_metadata_dir = negative_output_base / 'metadata' / split / 'temporal'
    neg_spatial_audio_dir = negative_output_base / 'audio' / split / 'spatial'
    neg_spatial_metadata_dir = negative_output_base / 'metadata' / split / 'spatial'
    
    # 创建所有输出目录
    positive_audio_dir.mkdir(parents=True, exist_ok=True)
    positive_metadata_dir.mkdir(parents=True, exist_ok=True)
    neg_temporal_audio_dir.mkdir(parents=True, exist_ok=True)
    neg_temporal_metadata_dir.mkdir(parents=True, exist_ok=True)
    neg_spatial_audio_dir.mkdir(parents=True, exist_ok=True)
    neg_spatial_metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有_0.flac文件
    audio_files = get_audio_files_ending_with_0(audio_dir)
    
    print(f"处理 {split} split: 找到 {len(audio_files)} 个以_0结尾的音频文件")
    print(f"将生成 {num_samples} 个正样本和 {num_samples * 2} 个负样本")
    
    success_count = 0
    attempts = 0
    max_attempts = num_samples * 10  # 最多尝试10倍
    
    with tqdm(total=num_samples, desc=f"Processing {split}") as pbar:
        while success_count < num_samples and attempts < max_attempts:
            attempts += 1
            
            # 随机选择两个不同的音频
            if len(audio_files) < 2:
                print("错误: 音频文件数量不足")
                break
            
            first_audio_0, second_audio_0 = random.sample(audio_files, 2)
            
            # 获取对应的 _2 文件
            first_stem_2 = first_audio_0.stem[:-1] + '2'
            second_stem_2 = second_audio_0.stem[:-1] + '2'
            
            first_audio_2 = audio_dir / f"{first_stem_2}.flac"
            second_audio_2 = audio_dir / f"{second_stem_2}.flac"
            
            # 检查所有需要的音频文件是否存在
            if not all([first_audio_2.exists(), second_audio_2.exists()]):
                continue
            
            # 检查所有metadata文件是否存在
            metadata_files = [
                metadata_dir / f"{first_audio_0.stem}.json",
                metadata_dir / f"{first_stem_2}.json",
                metadata_dir / f"{second_audio_0.stem}.json",
                metadata_dir / f"{second_stem_2}.json",
            ]
            
            if not all(f.exists() for f in metadata_files):
                continue
            
            try:
                # ========== 正样本: A_0 + B_2 ==========
                positive_audio_files = [first_audio_0, second_audio_2]
                positive_json_files = [
                    metadata_dir / f"{first_audio_0.stem}.json",
                    metadata_dir / f"{second_stem_2}.json",
                ]
                
                pos_audio, pos_sr = concatenate_audios(positive_audio_files)
                pos_metadata = merge_metadata(positive_json_files)
                
                pos_name = f"temporal_{success_count:06d}"
                save_audio_and_metadata(
                    pos_audio, pos_sr, pos_metadata,
                    positive_audio_dir / f"{pos_name}.flac",
                    positive_metadata_dir / f"{pos_name}.json"
                )
                
                # ========== 时序负样本: B_2 + A_0 (顺序反转) ==========
                temporal_neg_audio_files = [second_audio_2, first_audio_0]
                temporal_neg_json_files = [
                    metadata_dir / f"{second_stem_2}.json",
                    metadata_dir / f"{first_audio_0.stem}.json",
                ]
                
                temp_neg_audio, temp_neg_sr = concatenate_audios(temporal_neg_audio_files)
                temp_neg_metadata = merge_metadata(temporal_neg_json_files)
                
                temp_neg_name = f"temporal_negative_{success_count:06d}"
                save_audio_and_metadata(
                    temp_neg_audio, temp_neg_sr, temp_neg_metadata,
                    neg_temporal_audio_dir / f"{temp_neg_name}.flac",
                    neg_temporal_metadata_dir / f"{temp_neg_name}.json"
                )
                
                # ========== 空间负样本: A_2 + B_0 (空间转换) ==========
                spatial_neg_audio_files = [first_audio_2, second_audio_0]
                spatial_neg_json_files = [
                    metadata_dir / f"{first_stem_2}.json",
                    metadata_dir / f"{second_audio_0.stem}.json",
                ]
                
                spat_neg_audio, spat_neg_sr = concatenate_audios(spatial_neg_audio_files)
                spat_neg_metadata = merge_metadata(spatial_neg_json_files)
                
                spat_neg_name = f"spatial_negative_{success_count:06d}"
                save_audio_and_metadata(
                    spat_neg_audio, spat_neg_sr, spat_neg_metadata,
                    neg_spatial_audio_dir / f"{spat_neg_name}.flac",
                    neg_spatial_metadata_dir / f"{spat_neg_name}.json"
                )
                
                success_count += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"\n警告: 处理配对时出错: {e}")
                continue
    
    print(f"\n{split} split 完成:")
    print(f"  正样本: {success_count} 个")
    print(f"  时序负样本: {success_count} 个")
    print(f"  空间负样本: {success_count} 个")
    print(f"  总尝试次数: {attempts}")


def main():
    """主函数"""
    # 输入路径
    clotho_base = Path('datasets/spatial_audio_text/Clotho')
    
    # 输出路径
    positive_output_base = Path('/Users/cellren/Desktop/datasets/temporal_spatial_audio_text/stClotho')
    negative_output_base = Path('/Users/cellren/Desktop/datasets/temporal_spatial_audio_text/stClotho_negative')
    
    # 确保输入路径存在
    if not clotho_base.exists():
        print(f"错误: 输入路径不存在: {clotho_base}")
        return
    
    # 处理所有splits
    splits = ['train', 'valid', 'test']
    num_samples_per_split = {
        'train': 10000,
        'valid': 3000,
        'test': 3000
    }
    
    for split in splits:
        split_audio_dir = clotho_base / 'audio' / split
        if split_audio_dir.exists():
            print(f"\n{'='*70}")
            print(f"开始处理 {split} split")
            print(f"{'='*70}")
            process_split(
                clotho_base, 
                positive_output_base, 
                negative_output_base, 
                split, 
                num_samples=num_samples_per_split.get(split, 1000),
                seed=42 + splits.index(split)  # 不同split使用不同种子
            )
        else:
            print(f"警告: {split} split 的音频目录不存在: {split_audio_dir}")
    
    print(f"\n{'='*70}")
    print("所有处理完成！")
    print(f"正样本输出目录: {positive_output_base}")
    print(f"负样本输出目录: {negative_output_base}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()