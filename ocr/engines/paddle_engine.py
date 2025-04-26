#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PaddleOCR引擎模块
封装PaddleOCR引擎的功能，支持中文识别
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
logger = logging.getLogger("PaddleEngine")

# 检查PaddleOCR是否可用
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    warnings.warn("paddleocr 未安装，请使用 pip install paddleocr 安装")


class PaddleEngine(OCREngine):
    """PaddleOCR引擎封装"""
    
    def __init__(self, use_gpu=False, lang="ch", use_angle_cls=True, 
                 det=True, rec=True, cls=True, det_model_dir=None, 
                 rec_model_dir=None, cls_model_dir=None):
        """
        初始化PaddleOCR引擎
        
        参数:
            use_gpu: 是否使用GPU加速
            lang: 语言，默认中文
            use_angle_cls: 是否使用方向分类器
            det: 是否使用检测模型
            rec: 是否使用识别模型
            cls: 是否使用分类模型
            det_model_dir: 检测模型目录
            rec_model_dir: 识别模型目录
            cls_model_dir: 分类模型目录
        """
        if not self.is_available():
            raise ImportError("PaddleOCR不可用，请安装: pip install paddleocr")
        
        self.lang = lang
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        self.det = det
        self.rec = rec
        self.cls = cls
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.cls_model_dir = cls_model_dir
        self.preprocessor = ImagePreprocessor()
        
        # 懒加载，只在第一次recognize时初始化ocr
        self._ocr = None
        
        logger.info(f"PaddleEngine 初始化完成，语言: {lang}, GPU: {use_gpu}")
    
    def is_available(self) -> bool:
        """检查引擎是否可用"""
        return PADDLE_AVAILABLE
    
    @property
    def ocr(self):
        """懒加载PaddleOCR实例"""
        if self._ocr is None:
            try:
                logger.info(f"初始化PaddleOCR，语言: {self.lang}, GPU: {self.use_gpu}")
                self._ocr = PaddleOCR(
                    use_angle_cls=self.use_angle_cls,
                    lang=self.lang,
                    use_gpu=self.use_gpu,
                    det=self.det,
                    rec=self.rec,
                    cls=self.cls,
                    det_model_dir=self.det_model_dir,
                    rec_model_dir=self.rec_model_dir,
                    cls_model_dir=self.cls_model_dir
                )
            except Exception as e:
                logger.error(f"初始化PaddleOCR失败: {str(e)}")
                raise
        return self._ocr
    
    def recognize(self, image: np.ndarray, preprocessor_name: str = "original") -> List[OCRResult]:
        """
        使用PaddleOCR识别文本
        
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
        
        # PaddleOCR需要BGR格式图像
        if len(image.shape) == 2:  # 灰度图像转BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        # PaddleOCR处理
        try:
            ocr_result = self.ocr.ocr(image, cls=self.cls)
        except Exception as e:
            logger.error(f"PaddleOCR识别出错: {str(e)}")
            return []
        
        # 处理结果
        if ocr_result is None or len(ocr_result) == 0:
            return results
            
        # PaddleOCR的结果格式可能会根据版本变化，需要适应
        try:
            # 新版本Paddle返回格式处理
            for line in ocr_result[0]:
                try:
                    box = line[0]
                    text = line[1][0]
                    confidence = float(line[1][1])
                    
                    results.append(
                        OCRResult(
                            text=text,
                            confidence=confidence,
                            box=box,
                            engine="paddleocr",
                            preprocessor=preprocessor_name
                        )
                    )
                except (IndexError, ValueError, TypeError) as e:
                    logger.warning(f"处理单个结果时出错: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"处理PaddleOCR结果出错: {str(e)}")
        
        return results


def main():
    """PaddleOCR引擎命令行入口"""
    parser = argparse.ArgumentParser(description="PaddleOCR引擎")
    parser.add_argument("--input", "-i", required=True, help="输入图像文件或目录")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--lang", "-l", default="ch", help="语言设置，默认中文")
    parser.add_argument("--gpu", action="store_true", help="是否使用GPU加速")
    parser.add_argument("--no-angle-cls", action="store_false", dest="use_angle_cls",
                        help="不使用方向分类器")
    parser.add_argument("--preprocess", "-p", help="应用预处理方法")
    parser.add_argument("--no-annotated", action="store_true", help="不输出标注图像")
    parser.add_argument("--no-text", action="store_true", help="不输出文本文件")
    parser.add_argument("--no-json", action="store_true", help="不输出JSON文件")
    
    args = parser.parse_args()
    
    try:
        # 初始化引擎
        engine = PaddleEngine(
            use_gpu=args.gpu,
            lang=args.lang,
            use_angle_cls=args.use_angle_cls
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