#!/usr/bin/env python3
"""
分析 stClotho 数据集中时序音频的方向分布和时长统计
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List
import soundfile as sf


def analyze_temporal_metadata(metadata_dir: Path, audio_dir: Path) -> Dict:
    """
    分析时序metadata中的方向分布和时长
    
    Args:
        metadata_dir: metadata目录
        audio_dir: 音频文件目录
    
    Returns:
        统计信息字典
    """
    json_files = sorted(metadata_dir.glob('*.json'))
    
    if not json_files:
        print(f"警告: 在 {metadata_dir} 中没有找到JSON文件")
        return {}
    
    # 统计信息
    stats = {
        'total_files': len(json_files),
        'first_audio_directions': Counter(),
        'second_audio_directions': Counter(),
        'direction_pairs': Counter(),
        'total_duration': 0.0,  # 总时长（秒）
        'durations': [],  # 每个文件的时长
        'files_analyzed': []
    }
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'audio_segments' not in data or len(data['audio_segments']) < 2:
            print(f"警告: {json_file.name} 格式不正确，跳过")
            continue
        
        # 获取两个音频段的方向
        first_direction = data['audio_segments'][0]['metadata'].get('direction', 'Unknown')
        second_direction = data['audio_segments'][1]['metadata'].get('direction', 'Unknown')
        
        # 统计方向
        stats['first_audio_directions'][first_direction] += 1
        stats['second_audio_directions'][second_direction] += 1
        stats['direction_pairs'][(first_direction, second_direction)] += 1
        
        # 获取对应的音频文件并计算时长
        audio_file = audio_dir / f"{json_file.stem}.flac"
        duration = 0.0
        if audio_file.exists():
            try:
                info = sf.info(audio_file)
                duration = info.duration
                stats['total_duration'] += duration
                stats['durations'].append(duration)
            except Exception as e:
                print(f"警告: 无法读取音频文件 {audio_file}: {e}")
        else:
            print(f"警告: 音频文件不存在 {audio_file}")
        
        stats['files_analyzed'].append({
            'filename': json_file.stem,
            'first_audio': data['audio_segments'][0]['filename'],
            'first_direction': first_direction,
            'second_audio': data['audio_segments'][1]['filename'],
            'second_direction': second_direction,
            'duration': duration,
        })
    
    return stats


def format_duration(seconds: float) -> str:
    """将秒数格式化为可读的时长字符串"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs:.2f}s"


def print_statistics(stats: Dict, split_name: str):
    """打印统计信息"""
    print(f"\n{'='*70}")
    print(f"统计结果: {split_name}")
    print(f"{'='*70}")
    print(f"总文件数: {stats['total_files']}")
    print(f"成功分析: {len(stats['files_analyzed'])}")
    
    # 时长统计
    if stats['durations']:
        total_duration = stats['total_duration']
        avg_duration = total_duration / len(stats['durations'])
        min_duration = min(stats['durations'])
        max_duration = max(stats['durations'])
        
        print(f"\n{'='*70}")
        print("时长统计:")
        print(f"{'='*70}")
        print(f"  总时长:     {format_duration(total_duration)} ({total_duration:.2f} 秒)")
        print(f"  平均时长:   {avg_duration:.2f} 秒")
        print(f"  最短时长:   {min_duration:.2f} 秒")
        print(f"  最长时长:   {max_duration:.2f} 秒")
        print(f"  总时长(小时): {total_duration / 3600:.2f} 小时")
    
    print(f"\n{'='*70}")
    print("第一个音频段的方向分布:")
    print(f"{'='*70}")
    for direction, count in stats['first_audio_directions'].most_common():
        percentage = (count / len(stats['files_analyzed'])) * 100
        print(f"  {direction:45s}: {count:5d} ({percentage:5.2f}%)")
    
    print(f"\n{'='*70}")
    print("第二个音频段的方向分布:")
    print(f"{'='*70}")
    for direction, count in stats['second_audio_directions'].most_common():
        percentage = (count / len(stats['files_analyzed'])) * 100
        print(f"  {direction:45s}: {count:5d} ({percentage:5.2f}%)")
    
    print(f"\n{'='*70}")
    print("方向组合 TOP 20:")
    print(f"{'='*70}")
    print(f"{'第一段方向':25s} -> {'第二段方向':25s}   数量")
    print("-" * 70)
    for (dir1, dir2), count in stats['direction_pairs'].most_common(20):
        dir1_short = dir1.replace('The sound is coming from the ', '').rstrip('.')
        dir2_short = dir2.replace('The sound is coming from the ', '').rstrip('.')
        print(f"  {dir1_short:23s} -> {dir2_short:23s}   {count:5d}")


def main():
    """主函数"""
    base_path = Path('/Users/cellren/Desktop/datasets/temporal_spatial_audio_text/stClotho')
    
    # 检查路径是否存在
    if not base_path.exists():
        # 尝试相对路径
        base_path = Path('datasets/temporal_spatial_audio_text/stClotho')
        if not base_path.exists():
            print(f"错误: 找不到目录 {base_path}")
            return
    
    splits = ['train', 'valid', 'test']
    all_stats = {}
    total_duration_all = 0.0
    
    for split in splits:
        metadata_dir = base_path / 'metadata' / split
        audio_dir = base_path / 'audio' / split
        
        if not metadata_dir.exists():
            print(f"警告: {split} metadata目录不存在: {metadata_dir}")
            continue
        
        if not audio_dir.exists():
            print(f"警告: {split} audio目录不存在: {audio_dir}")
            continue
        
        print(f"\n正在分析 {split} split...")
        stats = analyze_temporal_metadata(metadata_dir, audio_dir)
        
        if stats:
            all_stats[split] = stats
            total_duration_all += stats['total_duration']
            print_statistics(stats, split.upper())
    
    # 汇总统计
    if all_stats:
        print(f"\n{'='*70}")
        print("汇总统计")
        print(f"{'='*70}")
        total_files = 0
        for split, stats in all_stats.items():
            file_count = len(stats['files_analyzed'])
            duration = stats['total_duration']
            total_files += file_count
            print(f"{split:10s}: {file_count:5d} 个文件, "
                  f"时长: {format_duration(duration)} ({duration/3600:.2f}h)")
        
        print(f"\n{'总计':10s}: {total_files:5d} 个文件, "
              f"时长: {format_duration(total_duration_all)} ({total_duration_all/3600:.2f}h)")


if __name__ == '__main__':
    main()