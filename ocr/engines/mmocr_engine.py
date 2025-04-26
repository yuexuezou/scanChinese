#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MMOCR引擎模块
封装MMOCR引擎用于识别图像中的文字
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
logger = logging.getLogger("MMOCREngine")

# 检查MMOCR是否可用
try:
    import mmocr
    from mmocr.apis import MMOCRInferencer
    MMOCR_AVAILABLE = True
except ImportError:
    MMOCR_AVAILABLE = False
    warnings.warn("mmocr 未安装，请使用 pip install mmocr 安装")


class MMOCREngine(OCREngine):
    """MMOCR引擎封装"""
    
    def __init__(self, 
                 det="DBNet",
                 rec="CRNN", 
                 device="auto"):
        """
        初始化MMOCR引擎
        
        参数:
            det: 检测模型名称，如'DBNet', 'FCENet'等
            rec: 识别模型名称，如'CRNN', 'SATRN'等
            device: 设备类型，'cpu', 'cuda', 'auto'
        """
        if not self.is_available():
            raise ImportError("MMOCR不可用，请安装: pip install mmocr")
        
        self.det = det
        self.rec = rec
        self.device = device
        self.preprocessor = ImagePreprocessor()
        
        # 懒加载模型
        self._ocr = None
        
        logger.info(f"MMOCREngine 初始化完成，检测模型: {det}, 识别模型: {rec}, 设备: {device}")
    
    def is_available(self) -> bool:
        """检查引擎是否可用"""
        return MMOCR_AVAILABLE
    
    @property
    def ocr(self):
        """懒加载MMOCR实例"""
        if self._ocr is None:
            try:
                logger.info(f"初始化MMOCR，检测模型: {self.det}, 识别模型: {self.rec}, 设备: {self.device}")
                self._ocr = MMOCRInferencer(det=self.det, rec=self.rec, device=self.device)
            except Exception as e:
                logger.error(f"初始化MMOCR失败: {str(e)}")
                raise
        return self._ocr
    
    def recognize(self, image: np.ndarray, preprocessor_name: str = "original") -> List[OCRResult]:
        """
        使用MMOCR识别文本
        
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
        
        # MMOCR处理
        try:
            # 转换为BGR格式 (MMOCR需要)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # MMOCR自身可以处理文件路径或图像数组
            inference_result = self.ocr(image)
            
            # 解析结果
            if inference_result:
                predictions = inference_result.get('predictions', [])
                if predictions and len(predictions) > 0:
                    for pred in predictions:
                        # 获取文本结果
                        rec_texts = pred.get('rec_texts', [])
                        rec_scores = pred.get('rec_scores', [])
                        det_polygons = pred.get('det_polygons', [])
                        
                        # 确保有识别结果
                        if rec_texts and det_polygons:
                            for i, (text, score, polygon) in enumerate(zip(rec_texts, rec_scores, det_polygons)):
                                # MMOCR的多边形坐标可能需要转换为我们所用的格式
                                box = []
                                for j in range(0, len(polygon), 2):
                                    if j + 1 < len(polygon):
                                        box.append([int(polygon[j]), int(polygon[j+1])])
                                
                                # 添加结果
                                if text:  # 仅当有文本时添加结果
                                    results.append(
                                        OCRResult(
                                            text=text,
                                            confidence=float(score),
                                            box=box,
                                            engine="mmocr",
                                            preprocessor=preprocessor_name
                                        )
                                    )
        except Exception as e:
            logger.error(f"MMOCR识别出错: {str(e)}")
        
        return results


def main():
    """MMOCR引擎命令行入口"""
    parser = argparse.ArgumentParser(description="MMOCR引擎")
    parser.add_argument("--input", "-i", required=True, help="输入图像文件路径")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--det", default="DBNet", help="检测模型名称")
    parser.add_argument("--rec", default="CRNN", help="识别模型名称")
    parser.add_argument("--device", default="auto", help="设备类型: cpu, cuda, auto")
    parser.add_argument("--preprocess", "-p", help="应用预处理方法")
    parser.add_argument("--no-annotated", action="store_true", help="不输出标注图像")
    parser.add_argument("--no-text", action="store_true", help="不输出文本文件")
    parser.add_argument("--no-json", action="store_true", help="不输出JSON文件")
    
    args = parser.parse_args()
    
    try:
        # 初始化引擎
        engine = MMOCREngine(det=args.det, rec=args.rec, device=args.device)
        
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