#!/usr/bin/env python3
"""
SafeTensor JSON Header Extractor

这个脚本用于从safetensor格式文件中提取JSON header部分。
SafeTensor文件格式：
- 前8字节：header长度n (little-endian uint64)
- 接下来n字节：UTF-8编码的JSON数据
- 之后：tensor数据
"""

import sys
import struct
import json
from pathlib import Path
from typing import Dict, Any


def read_safetensor_header(file_path: str) -> Dict[str, Any]:
    """
    读取safetensor文件的JSON header
    
    Args:
        file_path: safetensor文件路径
        
    Returns:
        解析后的JSON数据
    """
    with open(file_path, 'rb') as f:
        # 读取前8字节获取header长度
        header_size_bytes = f.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError("文件太小，无法读取header长度")
        
        # 解析header长度 (little-endian uint64)
        header_size = struct.unpack('<Q', header_size_bytes)[0]
        print(f"Header长度: {header_size} 字节")
        
        # 读取header内容
        header_bytes = f.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"无法读取完整的header，期望{header_size}字节，实际读取{len(header_bytes)}字节")
        
        # 解码UTF-8 JSON
        try:
            header_json = header_bytes.decode('utf-8')
            return json.loads(header_json)
        except UnicodeDecodeError as e:
            raise ValueError(f"Header不是有效的UTF-8编码: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Header不是有效的JSON格式: {e}")


def save_json_header(input_path: str, output_path: str = None) -> str:
    """
    提取并保存safetensor文件的JSON header
    
    Args:
        input_path: 输入的safetensor文件路径
        output_path: 输出JSON文件路径，如果为None则自动生成
        
    Returns:
        输出文件路径
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 如果没有指定输出路径，自动生成
    if output_path is None:
        output_path = input_file.with_suffix('.header.json')
    
    # 读取header
    print(f"正在读取文件: {input_path}")
    header_data = read_safetensor_header(input_path)
    
    # 保存为JSON文件
    output_file = Path(output_path)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(header_data, f, indent=2, ensure_ascii=False)
    
    print(f"JSON header已保存到: {output_path}")
    print(f"Header包含的键: {list(header_data.keys())}")
    
    return str(output_path)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python cut_safetensor.py <safetensor文件路径> [输出JSON文件路径]")
        print("示例: python cut_safetensor.py model.safetensors")
        print("示例: python cut_safetensor.py model.safetensors header.json")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        output_file = save_json_header(input_path, output_path)
        print(f"\n✓ 成功提取JSON header到: {output_file}")
    except Exception as e:
        print(f"✗ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
