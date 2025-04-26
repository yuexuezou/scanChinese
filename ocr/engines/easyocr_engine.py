#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EasyOCR引擎模块
封装EasyOCR引擎的功能，支持中文识别
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
logger = logging.getLogger("EasyOCREngine")

# 检查EasyOCR是否可用
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    warnings.warn("easyocr 未安装，请使用 pip install easyocr 安装")


class EasyOCREngine(OCREngine):
    """EasyOCR引擎封装"""
    
    def __init__(self, langs=["ch_sim", "en"], gpu=False):
        """
        初始化EasyOCR引擎
        
        参数:
            langs: 语言列表，默认使用简体中文和英文
            gpu: 是否使用GPU加速
        """
        if not self.is_available():
            raise ImportError("EasyOCR不可用，请安装: pip install easyocr")
        
        self.langs = langs
        self.gpu = gpu
        self.preprocessor = ImagePreprocessor()
        
        # 懒加载，只在第一次recognize时初始化reader
        self._reader = None
        
        logger.info(f"EasyOCREngine 初始化完成，语言: {langs}, GPU: {gpu}")
    
    def is_available(self) -> bool:
        """检查引擎是否可用"""
        return EASYOCR_AVAILABLE
    
    @property
    def reader(self):
        """懒加载EasyOCR reader"""
        if self._reader is None:
            try:
                logger.info(f"初始化EasyOCR reader，语言: {self.langs}, GPU: {self.gpu}")
                self._reader = easyocr.Reader(self.langs, gpu=self.gpu)
            except Exception as e:
                logger.error(f"初始化EasyOCR reader失败: {str(e)}")
                raise
        return self._reader
    
    def recognize(self, image: np.ndarray, preprocessor_name: str = "original") -> List[OCRResult]:
        """
        使用EasyOCR识别文本
        
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
        
        # EasyOCR处理
        try:
            ocr_result = self.reader.readtext(image)
        except Exception as e:
            logger.error(f"EasyOCR识别出错: {str(e)}")
            return []
        
        for item in ocr_result:
            box = item[0]
            text = item[1]
            confidence = float(item[2])
            
            # 确保坐标是普通 Python 列表，而不是 numpy 数组
            if isinstance(box, np.ndarray):
                box = box.tolist()
            else:
                # 转换每个坐标点为普通 Python 列表
                box = [[float(p[0]), float(p[1])] for p in box]
            
            results.append(
                OCRResult(
                    text=text,
                    confidence=confidence,
                    box=box,
                    engine="easyocr",
                    preprocessor=preprocessor_name
                )
            )
            
        return results


def main():
    """EasyOCR引擎命令行入口"""
    parser = argparse.ArgumentParser(description="EasyOCR引擎")
    parser.add_argument("--input", "-i", required=True, help="输入图像文件或目录")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--lang", "-l", nargs="+", default=["ch_sim", "en"], 
                        help="语言设置，默认中文和英文")
    parser.add_argument("--gpu", action="store_true", help="是否使用GPU加速")
    parser.add_argument("--preprocess", "-p", help="应用预处理方法")
    parser.add_argument("--no-annotated", action="store_true", help="不输出标注图像")
    parser.add_argument("--no-text", action="store_true", help="不输出文本文件")
    parser.add_argument("--no-json", action="store_true", help="不输出JSON文件")
    
    args = parser.parse_args()
    
    try:
        # 初始化引擎
        engine = EasyOCREngine(langs=args.lang, gpu=args.gpu)
        
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