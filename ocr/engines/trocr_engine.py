#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TrOCR引擎模块
封装TrOCR（Transformer OCR）引擎的功能，支持中文识别
"""

import os
import cv2
import numpy as np
import argparse
import logging
from typing import List, Dict, Any, Optional
import warnings
from PIL import Image

from ..base import OCREngine, OCRResult
from ..preprocessor import ImagePreprocessor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TrOCREngine")

# 检查TrOCR依赖是否可用
try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    warnings.warn("transformers或torch未安装，请使用 pip install transformers torch 安装")


class TrOCREngine(OCREngine):
    """TrOCR引擎封装"""
    
    def __init__(self, model_name="microsoft/trocr-base-handwritten", device=None):
        """
        初始化TrOCR引擎
        
        参数:
            model_name: 模型名称或路径，默认使用microsoft/trocr-base-handwritten
            device: 推理设备，如'cuda'或'cpu'，默认None表示自动选择
        """
        if not self.is_available():
            raise ImportError("TrOCR不可用，请安装transformers和torch")
        
        self.model_name = model_name
        
        # 设置设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 加载模型和处理器
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        
        self.preprocessor = ImagePreprocessor()
        logger.info(f"TrOCREngine 初始化完成，模型: {model_name}，设备: {self.device}")
    
    def is_available(self) -> bool:
        """检查引擎是否可用"""
        return TROCR_AVAILABLE
    
    def recognize(self, image: np.ndarray, preprocessor_name: str = "original") -> List[OCRResult]:
        """
        使用TrOCR识别文本
        
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
        
        # OpenCV图像转PIL图像
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        try:
            # 使用TrOCR处理图像
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # TrOCR不提供置信度和边界框，所以我们只返回文本
            # 边界框设为整个图像
            h, w = image.shape[:2]
            box = [[0, 0], [w, 0], [w, h], [0, h]]
            
            results.append(
                OCRResult(
                    text=generated_text,
                    confidence=1.0,  # TrOCR不提供置信度，默认为1.0
                    box=box,
                    engine="trocr",
                    preprocessor=preprocessor_name
                )
            )
            
        except Exception as e:
            logger.error(f"TrOCR识别出错: {str(e)}")
            return []
                
        return results


def main():
    """TrOCR引擎命令行入口"""
    parser = argparse.ArgumentParser(description="TrOCR引擎")
    parser.add_argument("--input", "-i", required=True, help="输入图像文件或目录")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--model", "-m", default="microsoft/trocr-base-handwritten", help="模型名称或路径")
    parser.add_argument("--device", "-d", default=None, help="推理设备，如'cuda'或'cpu'")
    parser.add_argument("--preprocess", "-p", help="应用预处理方法")
    parser.add_argument("--no-annotated", action="store_true", help="不输出标注图像")
    parser.add_argument("--no-text", action="store_true", help="不输出文本文件")
    parser.add_argument("--no-json", action="store_true", help="不输出JSON文件")
    
    args = parser.parse_args()
    
    try:
        # 初始化引擎
        engine = TrOCREngine(model_name=args.model, device=args.device)
        
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
                logger.info(f"识别到文本: {result.get('results', [])[0].text if result.get('results') else '无'}")
        else:
            logger.error(f"输入必须是文件: {args.input}")
    
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")


if __name__ == "__main__":
    main() 