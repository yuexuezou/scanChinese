#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MangaOCR引擎模块
封装MangaOCR引擎的功能，针对漫画/文本气泡优化
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
logger = logging.getLogger("MangaEngine")

# 检查MangaOCR是否可用
try:
    from manga_ocr import MangaOcr
    MANGA_AVAILABLE = True
except ImportError:
    MANGA_AVAILABLE = False
    warnings.warn("manga-ocr 未安装，请使用 pip install manga-ocr 安装")


class MangaEngine(OCREngine):
    """MangaOCR引擎封装，专为漫画文字气泡优化"""
    
    def __init__(self, force_cpu=False):
        """
        初始化MangaOCR引擎
        
        参数:
            force_cpu: 是否强制使用CPU模式
        """
        if not self.is_available():
            raise ImportError("MangaOCR不可用，请安装: pip install manga-ocr")
        
        self.force_cpu = force_cpu
        self.preprocessor = ImagePreprocessor()
        
        # 懒加载，只在第一次recognize时初始化ocr
        self._ocr = None
        
        logger.info(f"MangaEngine 初始化完成，CPU模式: {force_cpu}")
    
    def is_available(self) -> bool:
        """检查引擎是否可用"""
        return MANGA_AVAILABLE
    
    @property
    def ocr(self):
        """懒加载MangaOCR实例"""
        if self._ocr is None:
            try:
                logger.info(f"初始化MangaOCR，CPU模式: {self.force_cpu}")
                if self.force_cpu:
                    # 在CPU模式下初始化
                    import os
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                self._ocr = MangaOcr()
            except Exception as e:
                logger.error(f"初始化MangaOCR失败: {str(e)}")
                raise
        return self._ocr
    
    def recognize(self, image: np.ndarray, preprocessor_name: str = "original") -> List[OCRResult]:
        """
        使用MangaOCR识别文本
        
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
        
        # 转换为PIL Image (MangaOCR需要)
        from PIL import Image
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # MangaOCR处理
        try:
            text = self.ocr(pil_image)
            
            # MangaOCR只返回文本，没有位置和置信度信息
            # 我们设定一个默认的置信度和整个图像为包围框
            confidence = 0.9  # 默认置信度
            height, width = image.shape[:2]
            box = [[0, 0], [width, 0], [width, height], [0, height]]
            
            if text:  # 仅当有文本时添加结果
                results.append(
                    OCRResult(
                        text=text,
                        confidence=confidence,
                        box=box,
                        engine="mangaocr",
                        preprocessor=preprocessor_name
                    )
                )
        except Exception as e:
            logger.error(f"MangaOCR识别出错: {str(e)}")
        
        return results


def main():
    """MangaOCR引擎命令行入口"""
    parser = argparse.ArgumentParser(description="MangaOCR引擎 - 漫画文字识别引擎")
    parser.add_argument("--input", "-i", required=True, help="输入图像文件路径")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--cpu", action="store_true", help="强制使用CPU模式")
    parser.add_argument("--preprocess", "-p", help="应用预处理方法")
    parser.add_argument("--no-annotated", action="store_true", help="不输出标注图像")
    parser.add_argument("--no-text", action="store_true", help="不输出文本文件")
    parser.add_argument("--no-json", action="store_true", help="不输出JSON文件")
    
    args = parser.parse_args()
    
    try:
        # 初始化引擎
        engine = MangaEngine(force_cpu=args.cpu)
        
        # 处理图像
        if os.path.isfile(args.input):
            # 指定预处理方法
            if args.preprocess:
                preproc = args.preprocess
                # 读取图像
                image = cv2.imread(args.input)
                if image is None:
                    logger.error(f"无法读取图像: {args.input}")
                    return
                
                # 应用预处理
                processor = engine.preprocessor.get_preprocessor_by_name(preproc)
                processed = processor(image)
                
                # 保存预处理后的图像
                if args.output:
                    os.makedirs(args.output, exist_ok=True)
                    base_name = os.path.splitext(os.path.basename(args.input))[0]
                    preproc_path = os.path.join(args.output, f"{base_name}_{preproc}.jpg")
                    cv2.imwrite(preproc_path, processed)
                    logger.info(f"已保存预处理图像: {preproc_path}")
            
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
                
                # 打印识别结果
                for i, res in enumerate(result.get("results", []), 1):
                    logger.info(f"结果 {i}: {res.text} (置信度: {res.confidence:.2f})")
        else:
            logger.error(f"输入必须是文件: {args.input}")
    
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 