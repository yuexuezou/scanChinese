#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ScanChinese: 多引擎中文OCR处理系统
集成多个OCR引擎，优化中文文本识别
"""

import os
import sys
import argparse
import logging
from ocr.multi_engine import MultiEngineOCR
from ocr.engines import get_available_engines

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ScanChinese")


def main():
    """主函数"""
    # 获取系统中可用的OCR引擎
    available_engines = get_available_engines()
    engines_str = ", ".join(available_engines) if available_engines else "没有可用引擎"
    
    parser = argparse.ArgumentParser(
        description=f"ScanChinese: 多引擎中文OCR处理系统 (可用引擎: {engines_str})"
    )
    parser.add_argument("--input", "-i", required=True, help="输入图像文件或目录")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--engines", "-e", help="要使用的引擎，用逗号分隔，如 'tesseract,easyocr'")
    parser.add_argument("--gpu", action="store_true", help="使用GPU加速")
    parser.add_argument("--confidence", "-c", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--workers", "-w", type=int, default=4, help="并行处理的工作线程数")
    parser.add_argument("--no-annotated", action="store_true", help="不输出标注图像")
    parser.add_argument("--no-text", action="store_true", help="不输出文本文件")
    parser.add_argument("--no-json", action="store_true", help="不输出JSON文件")
    parser.add_argument("--version", "-v", action="store_true", help="显示版本信息")
    
    args = parser.parse_args()
    
    # 显示版本信息
    if args.version:
        from ocr import __version__
        print(f"ScanChinese v{__version__}")
        print(f"可用引擎: {engines_str}")
        return
    
    try:
        # 检查系统中是否有可用引擎
        if not available_engines:
            logger.error("系统中没有可用的OCR引擎，请安装至少一个OCR引擎")
            print(
                "请安装至少一个OCR引擎：\n"
                "- Tesseract: pip install pytesseract\n"
                "- PaddleOCR: pip install paddleocr\n"
                "- EasyOCR: pip install easyocr\n"
                "- RapidOCR: pip install rapidocr_onnxruntime"
            )
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
        
        # 处理输入
        if os.path.isfile(args.input):
            # 处理单个文件
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
                    if "annotated_image" in file_type and file_type == "annotated_image":
                        logger.info(f"标注图像: {file_path}")
                    elif file_type == "text_file":
                        logger.info(f"文本结果: {file_path}")
                    elif file_type == "json_file":
                        logger.info(f"JSON结果: {file_path}")
        else:
            # 处理目录
            if not os.path.isdir(args.input):
                logger.error(f"输入路径不存在或不是目录: {args.input}")
                return
                
            result = ocr.process_directory(
                args.input,
                args.output,
                max_workers=args.workers,
                write_annotated=not args.no_annotated,
                write_text=not args.no_text,
                write_json=not args.no_json
            )
            
            if "error" in result:
                logger.error(result["error"])
            else:
                logger.info(f"处理完成: {args.input}")
                summary = result.get("summary", {})
                logger.info(f"成功处理: {summary.get('successful', 0)}/{summary.get('total_files', 0)} 文件")
                logger.info(f"总处理时间: {summary.get('total_processing_time', 0):.2f} 秒")
                logger.info(f"汇总报告: {summary.get('output_directory', '')}/summary.json")
    
    except KeyboardInterrupt:
        logger.info("用户中断，正在退出...")
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 