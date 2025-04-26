#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Google Vision OCR引擎模块
封装Google Cloud Vision API的OCR功能，支持中文识别
"""

import os
import cv2
import numpy as np
import argparse
import logging
from typing import List, Dict, Any, Optional
import warnings
import base64
import json
from io import BytesIO

from ..base import OCREngine, OCRResult
from ..preprocessor import ImagePreprocessor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GoogleVisionEngine")

# 检查Google Vision依赖是否可用
try:
    from google.cloud import vision
    from google.oauth2 import service_account
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False
    warnings.warn("Google Cloud Vision库未安装，请使用 pip install google-cloud-vision 安装")


class GoogleVisionEngine(OCREngine):
    """Google Cloud Vision OCR引擎封装"""
    
    def __init__(self, credentials_path=None, language_hints=None):
        """
        初始化Google Vision引擎
        
        参数:
            credentials_path: Google Cloud认证凭据JSON文件路径，如果为None则尝试使用环境变量
            language_hints: 语言提示列表，如['zh-CN', 'en']，用于提高特定语言的识别准确性
        """
        if not self.is_available():
            raise ImportError("Google Cloud Vision API不可用，请安装google-cloud-vision库")
        
        self.credentials_path = credentials_path
        self.language_hints = language_hints or ['zh-CN', 'en']
        
        # 初始化客户端
        try:
            if credentials_path:
                # 使用凭据文件初始化
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                self.client = vision.ImageAnnotatorClient(credentials=credentials)
            else:
                # 使用环境变量初始化（需要设置GOOGLE_APPLICATION_CREDENTIALS环境变量）
                self.client = vision.ImageAnnotatorClient()
                
            self.preprocessor = ImagePreprocessor()
            logger.info("GoogleVisionEngine 初始化完成")
        except Exception as e:
            logger.error(f"初始化Google Vision客户端失败: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """检查引擎是否可用"""
        return GOOGLE_VISION_AVAILABLE
    
    def recognize(self, image: np.ndarray, preprocessor_name: str = "original") -> List[OCRResult]:
        """
        使用Google Cloud Vision API识别文本
        
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
        
        try:
            # 将图像编码为JPEG
            success, buffer = cv2.imencode('.jpg', image)
            if not success:
                logger.error("图像编码失败")
                return results
            
            # 创建Vision API图像对象
            content = buffer.tobytes()
            vision_image = vision.Image(content=content)
            
            # 设置识别选项
            context = vision.ImageContext(language_hints=self.language_hints)
            
            # 调用API进行文本检测
            response = self.client.text_detection(image=vision_image, image_context=context)
            
            # 检查是否有错误
            if response.error.message:
                logger.error(f"Google Vision API错误: {response.error.message}")
                return results
            
            # 处理结果
            for text_annotation in response.text_annotations[1:] if response.text_annotations else []:
                # 提取文本
                text = text_annotation.description
                
                # 提取边界框
                vertices = text_annotation.bounding_poly.vertices
                box = [[vertex.x, vertex.y] for vertex in vertices]
                
                # Google Vision不提供置信度，使用默认值
                confidence = 0.9
                
                results.append(
                    OCRResult(
                        text=text,
                        confidence=confidence,
                        box=box,
                        engine="google_vision",
                        preprocessor=preprocessor_name
                    )
                )
            
            # 如果没有找到任何文本区域，但有整体文本
            if not results and response.text_annotations:
                # 使用第一个注释，它包含整个图像的文本
                full_text = response.text_annotations[0].description
                
                # 整个图像的边界框
                h, w = image.shape[:2]
                box = [[0, 0], [w, 0], [w, h], [0, h]]
                
                results.append(
                    OCRResult(
                        text=full_text,
                        confidence=0.9,
                        box=box,
                        engine="google_vision",
                        preprocessor=preprocessor_name
                    )
                )
            
        except Exception as e:
            logger.error(f"Google Vision API识别出错: {str(e)}")
            
        return results


def main():
    """Google Vision引擎命令行入口"""
    parser = argparse.ArgumentParser(description="Google Cloud Vision OCR引擎")
    parser.add_argument("--input", "-i", required=True, help="输入图像文件或目录")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--credentials", "-c", help="Google Cloud认证凭据JSON文件路径")
    parser.add_argument("--languages", "-l", help="语言提示，逗号分隔，如'zh-CN,en'")
    parser.add_argument("--preprocess", "-p", help="应用预处理方法")
    parser.add_argument("--no-annotated", action="store_true", help="不输出标注图像")
    parser.add_argument("--no-text", action="store_true", help="不输出文本文件")
    parser.add_argument("--no-json", action="store_true", help="不输出JSON文件")
    
    args = parser.parse_args()
    
    try:
        # 解析语言提示
        language_hints = args.languages.split(',') if args.languages else None
        
        # 初始化引擎
        engine = GoogleVisionEngine(
            credentials_path=args.credentials,
            language_hints=language_hints
        )
        
        # 处理图像
        if os.path.isfile(args.input):
            # 单个文件处理
            result = engine.process_image(
                args.input, 
                args.output,
                preprocessor_name=args.preprocess,
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