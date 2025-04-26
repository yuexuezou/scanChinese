#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
单引擎OCR使用示例
演示如何单独使用各OCR引擎
"""

import os
import sys
import argparse
import logging

# 添加项目根目录到Python路径，以便直接导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.engines import get_available_engines, get_engine_by_name

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SingleEngineDemo")


def main():
    """单引擎OCR示例主函数"""
    # 获取系统中可用的OCR引擎
    available_engines = get_available_engines()
    engines_str = ", ".join(available_engines) if available_engines else "没有可用引擎"
    
    parser = argparse.ArgumentParser(
        description=f"单引擎OCR使用示例 (可用引擎: {engines_str})"
    )
    parser.add_argument("--input", "-i", required=True, help="输入图像文件")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--engine", "-e", required=True, choices=available_engines, 
                       help="使用的OCR引擎")
    parser.add_argument("--gpu", action="store_true", help="使用GPU加速")
    parser.add_argument("--lang", "-l", help="语言设置")
    parser.add_argument("--preprocess", "-p", help="预处理方法")
    parser.add_argument("--no-annotated", action="store_true", help="不输出标注图像")
    parser.add_argument("--no-text", action="store_true", help="不输出文本文件")
    parser.add_argument("--no-json", action="store_true", help="不输出JSON文件")
    
    args = parser.parse_args()
    
    try:
        # 检查系统中是否有可用引擎
        if not available_engines:
            logger.error("系统中没有可用的OCR引擎，请安装至少一个OCR引擎")
            return
        
        # 检查指定的引擎是否可用
        if args.engine not in available_engines:
            logger.error(f"引擎 {args.engine} 不可用，请安装相应的依赖")
            return
        
        # 根据不同引擎设置参数
        engine_kwargs = {}
        
        if args.engine == "tesseract":
            engine_kwargs["lang"] = args.lang or "chi_sim+chi_tra"
        elif args.engine == "paddle":
            engine_kwargs["use_gpu"] = args.gpu
            engine_kwargs["lang"] = args.lang or "ch"
        elif args.engine == "easyocr":
            langs = []
            if args.lang:
                langs = args.lang.split(",")
            else:
                langs = ["ch_sim", "en"]
            engine_kwargs["langs"] = langs
            engine_kwargs["gpu"] = args.gpu
        elif args.engine == "rapidocr":
            # RapidOCR没有额外参数
            pass
        
        # 初始化OCR引擎
        engine = get_engine_by_name(args.engine, **engine_kwargs)
        
        # 处理图像
        result = engine.process_image(
            args.input,
            args.output,
            write_annotated=not args.no_annotated,
            write_text=not args.no_text,
            write_json=not args.no_json
        )
        
        if "error" in result:
            logger.error(result["error"])
        else:
            logger.info(f"处理完成: {args.input}")
            logger.info(f"识别到 {len(result.get('results', []))} 个文本区域")
            
            # 打印输出文件路径
            for file_type, file_path in result.get("output_files", {}).items():
                if file_type == "annotated_image":
                    logger.info(f"标注图像: {file_path}")
                elif file_type == "text_file":
                    logger.info(f"文本结果: {file_path}")
                elif file_type == "json_file":
                    logger.info(f"JSON结果: {file_path}")
            
            # 打印识别结果
            print("\n识别结果:")
            for idx, res in enumerate(result.get("results", []), 1):
                print(f"{idx}. 文本: \"{res.text}\" (置信度: {res.confidence:.2f})")
    
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 