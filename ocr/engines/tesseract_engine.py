#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tesseract OCR引擎模块
封装Tesseract OCR引擎的功能，支持中文识别
"""

import os
import cv2
import numpy as np
import argparse
import logging
from typing import List, Dict, Any, Optional
import warnings

from ..base import OCREngine, OCRResult
from ..preprocessor import ImagePreprocessor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TesseractEngine")

# 检查Tesseract是否可用
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    warnings.warn("pytesseract 未安装，请使用 pip install pytesseract 安装")


class TesseractEngine(OCREngine):
    """Tesseract OCR引擎封装"""
    
    def __init__(self, lang="chi_sim+chi_tra", config=""):
        """
        初始化Tesseract引擎
        
        参数:
            lang: 语言模型，默认使用简体中文+繁体中文
            config: 额外的Tesseract配置参数
        """
        if not self.is_available():
            raise ImportError("Tesseract不可用，请安装pytesseract和Tesseract-OCR")
        
        self.lang = lang
        self.config = config
        self.preprocessor = ImagePreprocessor()
        logger.info(f"TesseractEngine 初始化完成，语言: {lang}")
    
    def is_available(self) -> bool:
        """检查引擎是否可用"""
        if not TESSERACT_AVAILABLE:
            return False
        
        # 检查tesseract是否安装在系统中
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    
    def recognize(self, image: np.ndarray, preprocessor_name: str = "original") -> List[OCRResult]:
        """
        使用Tesseract识别文本
        
        参数:
            image: 输入图像
            preprocessor_name: 预处理方法名称
            
        返回:
            OCRResult对象列表
        """
        results = []
        
        # 应用预处理
        if preprocessor_name != "original":
            processor_fn = self.preprocessor.get_preprocessor_by_name(preprocessor_name)
            image = processor_fn(image)
        
        # 获取置信度和文本
        try:
            data = pytesseract.image_to_data(
                image, lang=self.lang, output_type=pytesseract.Output.DICT,
                config=self.config
            )
        except Exception as e:
            logger.error(f"Tesseract识别出错: {str(e)}")
            return []
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:  # 过滤掉置信度为-1的结果
                text = data['text'][i].strip()
                if not text:
                    continue
                
                confidence = float(data['conf'][i]) / 100.0
                x, y, w, h = (
                    data['left'][i],
                    data['top'][i],
                    data['width'][i],
                    data['height'][i]
                )
                
                # 转换为四点坐标 [左上, 右上, 右下, 左下]
                box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                
                results.append(
                    OCRResult(
                        text=text,
                        confidence=confidence,
                        box=box,
                        engine="tesseract",
                        preprocessor=preprocessor_name
                    )
                )
                
        return results


def main():
    """Tesseract引擎命令行入口"""
    parser = argparse.ArgumentParser(description="Tesseract OCR引擎")
    parser.add_argument("--input", "-i", required=True, help="输入图像文件或目录")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--lang", "-l", default="chi_sim+chi_tra", help="语言设置，默认中文")
    parser.add_argument("--config", "-c", default="", help="Tesseract 配置参数")
    parser.add_argument("--preprocess", "-p", help="应用预处理方法")
    parser.add_argument("--no-annotated", action="store_true", help="不输出标注图像")
    parser.add_argument("--no-text", action="store_true", help="不输出文本文件")
    parser.add_argument("--no-json", action="store_true", help="不输出JSON文件")
    
    args = parser.parse_args()
    
    try:
        # 初始化引擎
        engine = TesseractEngine(lang=args.lang, config=args.config)
        
        # 处理图像
        if os.path.isfile(args.input):
            # 单个文件处理
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
        else:
            logger.error(f"输入必须是文件: {args.input}")
    
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")


if __name__ == "__main__":
    main() 