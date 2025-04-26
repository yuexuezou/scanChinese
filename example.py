#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多引擎中文OCR处理系统使用示例
"""

import os
import sys
from advanced_chinese_ocr import MultiEngineOCR

def main():
    """示例脚本主函数"""
    # 初始化多引擎OCR系统
    print("初始化多引擎OCR系统...")
    ocr = MultiEngineOCR(use_gpu=False, confidence_threshold=0.6)
    
    # 判断是否提供了图像路径参数
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("请提供一个图像文件路径作为命令行参数")
        print("用法: python example.py path/to/image.jpg")
        return
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 文件 '{image_path}' 不存在")
        return
    
    # 创建输出目录
    output_dir = "ocr_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"处理图像: {image_path}")
    
    # 处理图像
    result = ocr.process_image(
        image_path,
        output_dir=output_dir,
        write_annotated=True,
        write_text=True,
        write_json=True
    )
    
    # 打印处理结果
    print(f"处理完成，耗时: {result['processing_time']:.2f} 秒")
    print(f"识别到 {len(result['results'])} 个文本区域")
    
    # 打印识别到的文本
    print("\n识别到的文本:")
    for i, text_result in enumerate(result["results"], 1):
        print(f"{i}. {text_result.text} (置信度: {text_result.confidence:.2f}, 引擎: {text_result.engine})")
    
    # 打印输出文件路径
    print("\n输出文件:")
    for file_type, file_path in result["output_files"].items():
        print(f"- {file_type}: {file_path}")


if __name__ == "__main__":
    main() 