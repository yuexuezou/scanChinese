#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多引擎OCR使用示例
演示如何同时使用多个OCR引擎并整合结果
"""

import os
import sys
import argparse
import logging

# 添加项目根目录到Python路径，以便直接导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.multi_engine import MultiEngineOCR
from ocr.engines import get_available_engines

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MultiEngineDemo")


def main():
    """多引擎OCR示例主函数"""
    # 获取系统中可用的OCR引擎
    available_engines = get_available_engines()
    engines_str = ", ".join(available_engines) if available_engines else "没有可用引擎"
    
    parser = argparse.ArgumentParser(
        description=f"多引擎OCR使用示例 (可用引擎: {engines_str})"
    )
    parser.add_argument("--input", "-i", required=True, help="输入图像文件")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--engines", "-e", help="要使用的引擎，用逗号分隔，如 'tesseract,easyocr'")
    parser.add_argument("--gpu", action="store_true", help="使用GPU加速")
    parser.add_argument("--confidence", "-c", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--no-annotated", action="store_true", help="不输出标注图像")
    parser.add_argument("--no-text", action="store_true", help="不输出文本文件")
    parser.add_argument("--no-json", action="store_true", help="不输出JSON文件")
    parser.add_argument("--details", "-d", action="store_true", help="显示每个引擎的识别结果详情")
    
    args = parser.parse_args()
    
    try:
        # 检查系统中是否有可用引擎
        if not available_engines:
            logger.error("系统中没有可用的OCR引擎，请安装至少一个OCR引擎")
            return
        
        # 解析引擎列表
        engines = args.engines.split(",") if args.engines else None
        
        # 检查指定的引擎是否可用
        if engines:
            for engine in engines:
                if engine not in available_engines:
                    logger.error(f"引擎 {engine} 不可用，请安装相应的依赖")
                    return
        
        # 初始化OCR
        ocr = MultiEngineOCR(
            engines=engines,
            use_gpu=args.gpu, 
            confidence_threshold=args.confidence
        )
        
        if not ocr.engines:
            logger.error("没有可用的OCR引擎，请安装至少一个OCR引擎")
            return
        
        # 处理图像
        result = ocr.process_image(
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
                    logger.info(f"多引擎标注图像: {file_path}")
                elif file_type == "text_file":
                    logger.info(f"文本结果: {file_path}")
                elif file_type == "json_file":
                    logger.info(f"JSON结果: {file_path}")
            
            # 打印总体识别结果
            print("\n多引擎整合结果:")
            for idx, res in enumerate(result.get("results", []), 1):
                print(f"{idx}. 文本: \"{res.text}\" (置信度: {res.confidence:.2f}, 引擎: {res.engine}, 预处理: {res.preprocessor})")
            
            # 如果用户要求显示详情，则打印每个引擎的识别结果
            if args.details:
                results_by_engine = result.get("results_by_engine", {})
                for engine_name, engine_results in results_by_engine.items():
                    if engine_results:
                        print(f"\n{engine_name} 识别结果 ({len(engine_results)} 个):")
                        for idx, res in enumerate(engine_results, 1):
                            print(f"{idx}. 文本: \"{res.text}\" (置信度: {res.confidence:.2f}, 预处理: {res.preprocessor})")
                    else:
                        print(f"\n{engine_name} 未识别到任何文本")
    
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 