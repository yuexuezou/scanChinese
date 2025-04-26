#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RapidOCR引擎模块
封装RapidOCR引擎的功能，支持中文识别
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
logger = logging.getLogger("RapidOCREngine")

# 检查RapidOCR是否可用
try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False
    warnings.warn("rapidocr_onnxruntime 未安装，请使用 pip install rapidocr_onnxruntime 安装")


class RapidOCREngine(OCREngine):
    """RapidOCR引擎封装"""
    
    def __init__(self, det_model=None, rec_model=None, cls_model=None):
        """
        初始化RapidOCR引擎
        
        参数:
            det_model: 检测模型路径
            rec_model: 识别模型路径
            cls_model: 分类模型路径
        """
        if not self.is_available():
            raise ImportError("RapidOCR不可用，请安装: pip install rapidocr_onnxruntime")
        
        self.det_model = det_model
        self.rec_model = rec_model
        self.cls_model = cls_model
        self.preprocessor = ImagePreprocessor()
        
        # 懒加载，只在第一次recognize时初始化ocr
        self._ocr = None
        
        logger.info("RapidOCREngine 初始化完成")
    
    def is_available(self) -> bool:
        """检查引擎是否可用"""
        return RAPIDOCR_AVAILABLE
    
    @property
    def ocr(self):
        """懒加载RapidOCR实例"""
        if self._ocr is None:
            try:
                logger.info("初始化RapidOCR")
                # 根据是否提供了模型路径创建RapidOCR实例
                if self.det_model or self.rec_model or self.cls_model:
                    self._ocr = RapidOCR(
                        det_model_path=self.det_model,
                        rec_model_path=self.rec_model,
                        cls_model_path=self.cls_model
                    )
                else:
                    self._ocr = RapidOCR()
            except Exception as e:
                logger.error(f"初始化RapidOCR失败: {str(e)}")
                raise
        return self._ocr
    
    def recognize(self, image: np.ndarray, preprocessor_name: str = "original") -> List[OCRResult]:
        """
        使用RapidOCR识别文本
        
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
        
        # RapidOCR处理
        try:
            ocr_result, _ = self.ocr(image)  # RapidOCR返回 (结果, 耗时)
        except Exception as e:
            logger.error(f"RapidOCR识别出错: {str(e)}")
            return []
        
        # 处理结果
        if ocr_result is None:
            return results
            
        for line in ocr_result:
            try:
                box = line[0]  # 坐标点，[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                text = line[1]  # 文本
                confidence = float(line[2])  # 置信度
                
                results.append(
                    OCRResult(
                        text=text,
                        confidence=confidence,
                        box=box,
                        engine="rapidocr",
                        preprocessor=preprocessor_name
                    )
                )
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"处理单个结果时出错: {str(e)}")
                continue
        
        return results


def main():
    """RapidOCR引擎命令行入口"""
    parser = argparse.ArgumentParser(description="RapidOCR引擎")
    parser.add_argument("--input", "-i", required=True, help="输入图像文件或目录")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--det-model", help="检测模型路径")
    parser.add_argument("--rec-model", help="识别模型路径")
    parser.add_argument("--cls-model", help="分类模型路径")
    parser.add_argument("--preprocess", "-p", help="应用预处理方法")
    parser.add_argument("--no-annotated", action="store_true", help="不输出标注图像")
    parser.add_argument("--no-text", action="store_true", help="不输出文本文件")
    parser.add_argument("--no-json", action="store_true", help="不输出JSON文件")
    
    args = parser.parse_args()
    
    try:
        # 初始化引擎
        engine = RapidOCREngine(
            det_model=args.det_model,
            rec_model=args.rec_model,
            cls_model=args.cls_model
        )
        
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