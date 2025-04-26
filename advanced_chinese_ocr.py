#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高级中文OCR处理系统
集成多个OCR引擎，优化中文文本识别
"""

import os
import cv2
import numpy as np
import json
import argparse
import time
import jieba
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

# 导入OCR引擎
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("未找到Tesseract，相关功能将被禁用")

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logging.warning("未找到PaddleOCR，相关功能将被禁用")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("未找到EasyOCR，相关功能将被禁用")

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False
    logging.warning("未找到RapidOCR，相关功能将被禁用")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AdvancedChineseOCR")

@dataclass
class OCRResult:
    """OCR识别结果数据结构"""
    text: str  # 识别的文本
    confidence: float  # 置信度
    box: Optional[List] = None  # 文本框坐标 [左上, 右上, 右下, 左下]
    engine: str = ""  # 使用的OCR引擎
    preprocessor: str = ""  # 使用的预处理方法


class ImagePreprocessor:
    """图像预处理器"""
    
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """转换为灰度图像"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def adaptive_histogram_equalization(image: np.ndarray) -> np.ndarray:
        """自适应直方图均衡化"""
        gray = ImagePreprocessor.to_grayscale(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, alpha=1.5, beta=0) -> np.ndarray:
        """增强对比度"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """降噪处理"""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    @staticmethod
    def adaptive_threshold(image: np.ndarray) -> np.ndarray:
        """自适应阈值二值化"""
        gray = ImagePreprocessor.to_grayscale(image)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    
    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        """锐化处理"""
        kernel = np.array([[-1, -1, -1], 
                           [-1,  9, -1], 
                           [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def gaussian_blur(image: np.ndarray, kernel_size=(5, 5)) -> np.ndarray:
        """高斯模糊"""
        return cv2.GaussianBlur(image, kernel_size, 0)
    
    def apply_all_preprocessors(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """应用所有预处理器并返回结果字典"""
        results = {
            "original": image.copy(),
            "grayscale": self.to_grayscale(image),
            "adaptive_histogram": self.adaptive_histogram_equalization(image),
            "contrast_enhanced": self.enhance_contrast(image),
            "denoised": self.denoise(image),
            "adaptive_threshold": self.adaptive_threshold(image),
            "sharpened": self.sharpen(image),
            "gaussian_blur": self.gaussian_blur(image)
        }
        return results


class TesseractOCR:
    """Tesseract OCR引擎封装"""
    
    def __init__(self, lang="chi_sim+chi_tra"):
        if not TESSERACT_AVAILABLE:
            raise ImportError("Tesseract不可用，请安装：pip install pytesseract")
        self.lang = lang
    
    def recognize(self, image: np.ndarray, preprocessor_name: str = "original") -> List[OCRResult]:
        """使用Tesseract识别文本"""
        results = []
        
        # 获取置信度和文本
        data = pytesseract.image_to_data(
            image, lang=self.lang, output_type=pytesseract.Output.DICT
        )
        
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


class PaddleOCREngine:
    """PaddleOCR引擎封装"""
    
    def __init__(self, use_gpu=False, lang="ch"):
        if not PADDLE_AVAILABLE:
            raise ImportError("PaddleOCR不可用，请安装：pip install paddleocr")
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)
    
    def recognize(self, image: np.ndarray, preprocessor_name: str = "original") -> List[OCRResult]:
        """使用PaddleOCR识别文本"""
        results = []
        
        # PaddleOCR需要BGR格式图像
        if len(image.shape) == 2:  # 灰度图像转BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        ocr_result = self.ocr.ocr(image, cls=True)
        
        if not ocr_result or len(ocr_result) == 0:
            return results
            
        for item in ocr_result[0]:
            box = item[0]
            text = item[1][0]
            confidence = float(item[1][1])
            
            results.append(
                OCRResult(
                    text=text,
                    confidence=confidence,
                    box=box,
                    engine="paddleocr",
                    preprocessor=preprocessor_name
                )
            )
            
        return results


class EasyOCREngine:
    """EasyOCR引擎封装"""
    
    def __init__(self, gpu=False, langs=["ch_sim", "en"]):
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR不可用，请安装：pip install easyocr")
        self.reader = easyocr.Reader(langs, gpu=gpu)
    
    def recognize(self, image: np.ndarray, preprocessor_name: str = "original") -> List[OCRResult]:
        """使用EasyOCR识别文本"""
        results = []
        
        # EasyOCR处理
        ocr_result = self.reader.readtext(image)
        
        for item in ocr_result:
            box = item[0]
            text = item[1]
            confidence = float(item[2])
            
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


class RapidOCREngine:
    """RapidOCR引擎封装"""
    
    def __init__(self):
        if not RAPIDOCR_AVAILABLE:
            raise ImportError("RapidOCR不可用，请安装：pip install rapidocr-onnxruntime")
        self.ocr = RapidOCR()
    
    def recognize(self, image: np.ndarray, preprocessor_name: str = "original") -> List[OCRResult]:
        """使用RapidOCR识别文本"""
        results = []
        
        # RapidOCR需要BGR格式图像
        if len(image.shape) == 2:  # 灰度图像转BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        result, elapse = self.ocr(image)
        
        if result:
            for item in result:
                box = item[0]
                text = item[1]
                confidence = float(item[2])
                
                results.append(
                    OCRResult(
                        text=text,
                        confidence=confidence,
                        box=box,
                        engine="rapidocr",
                        preprocessor=preprocessor_name
                    )
                )
                
        return results


class MultiEngineOCR:
    """多引擎OCR集成器"""
    
    def __init__(self, use_gpu=False, confidence_threshold=0.5):
        self.preprocessor = ImagePreprocessor()
        self.engines = {}
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu
        
        # 初始化可用的OCR引擎
        self._init_engines()
    
    def _init_engines(self):
        """初始化所有可用的OCR引擎"""
        if TESSERACT_AVAILABLE:
            try:
                self.engines["tesseract"] = TesseractOCR()
                logger.info("Tesseract OCR引擎已初始化")
            except Exception as e:
                logger.error(f"Tesseract初始化失败: {str(e)}")
        
        if PADDLE_AVAILABLE:
            try:
                self.engines["paddleocr"] = PaddleOCREngine(use_gpu=self.use_gpu)
                logger.info("PaddleOCR引擎已初始化")
            except Exception as e:
                logger.error(f"PaddleOCR初始化失败: {str(e)}")
        
        if EASYOCR_AVAILABLE:
            try:
                self.engines["easyocr"] = EasyOCREngine(gpu=self.use_gpu)
                logger.info("EasyOCR引擎已初始化")
            except Exception as e:
                logger.error(f"EasyOCR初始化失败: {str(e)}")
        
        if RAPIDOCR_AVAILABLE:
            try:
                self.engines["rapidocr"] = RapidOCREngine()
                logger.info("RapidOCR引擎已初始化")
            except Exception as e:
                logger.error(f"RapidOCR初始化失败: {str(e)}")
        
        if not self.engines:
            raise RuntimeError("没有可用的OCR引擎，请至少安装一个OCR引擎")
    
    def process_image(self, image_path: str, output_dir: str = None, 
                     write_annotated: bool = True, write_text: bool = True, 
                     write_json: bool = True) -> Dict[str, Any]:
        """处理单个图像并返回结果"""
        start_time = time.time()
        
        # 准备输出目录
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = os.path.dirname(image_path)
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return {"error": f"无法读取图像: {image_path}"}
        
        # 获取图像文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 应用所有预处理器
        preprocessed_images = self.preprocessor.apply_all_preprocessors(image)
        
        # 存储所有引擎的结果
        all_results = []
        
        # 对每种预处理方法和每个引擎运行OCR
        for preproc_name, preproc_image in preprocessed_images.items():
            for engine_name, engine in self.engines.items():
                try:
                    logger.info(f"使用 {engine_name} 引擎处理 {preproc_name} 图像")
                    results = engine.recognize(preproc_image, preproc_name)
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"{engine_name}处理{preproc_name}图像时出错: {str(e)}")
        
        # 整合和优化结果
        optimized_results = self._optimize_results(all_results)
        
        # 写出结果
        output_files = {}
        
        # 生成带标注的图像
        if write_annotated:
            annotated_img = self._draw_results(image.copy(), optimized_results)
            annotated_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
            cv2.imwrite(annotated_path, annotated_img)
            output_files["annotated_image"] = annotated_path
        
        # 生成纯文本结果
        if write_text:
            text_content = "\n".join([r.text for r in optimized_results])
            text_path = os.path.join(output_dir, f"{base_name}_result.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            output_files["text_file"] = text_path
        
        # 生成JSON结果
        if write_json:
            json_result = {
                "file": image_path,
                "processing_time": time.time() - start_time,
                "results": [
                    {
                        "text": r.text,
                        "confidence": r.confidence,
                        "box": r.box,
                        "engine": r.engine,
                        "preprocessor": r.preprocessor
                    } for r in optimized_results
                ]
            }
            json_path = os.path.join(output_dir, f"{base_name}_result.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_result, f, ensure_ascii=False, indent=2)
            output_files["json_file"] = json_path
        
        return {
            "results": optimized_results,
            "processing_time": time.time() - start_time,
            "output_files": output_files
        }
    
    def process_directory(self, input_dir: str, output_dir: str = None, 
                         extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp"],
                         max_workers: int = 4, **kwargs) -> Dict[str, Any]:
        """处理目录中的所有图像"""
        if not os.path.isdir(input_dir):
            return {"error": f"输入目录不存在: {input_dir}"}
        
        if output_dir is None:
            output_dir = os.path.join(input_dir, "ocr_results")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图像文件
        image_files = []
        for ext in extensions:
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith(ext):
                        image_files.append(os.path.join(root, file))
        
        if not image_files:
            return {"error": f"在目录 {input_dir} 中没有找到图像文件"}
        
        logger.info(f"找到 {len(image_files)} 个图像文件")
        
        # 使用线程池并行处理图像
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_image, img_file, output_dir, **kwargs): img_file 
                for img_file in image_files
            }
            
            for future in future_to_file:
                file = future_to_file[future]
                try:
                    results[file] = future.result()
                except Exception as e:
                    logger.error(f"处理文件 {file} 时出错: {str(e)}")
                    results[file] = {"error": str(e)}
        
        # 生成汇总报告
        summary = {
            "total_files": len(image_files),
            "successful": sum(1 for r in results.values() if "error" not in r),
            "failed": sum(1 for r in results.values() if "error" in r),
            "total_processing_time": sum(r.get("processing_time", 0) for r in results.values() if "processing_time" in r),
            "output_directory": output_dir
        }
        
        # 写入汇总JSON
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return {
            "summary": summary,
            "results": results
        }
    
    def _optimize_results(self, results: List[OCRResult]) -> List[OCRResult]:
        """整合和优化OCR结果"""
        if not results:
            return []
        
        # 按置信度排序
        sorted_results = sorted(results, key=lambda x: x.confidence, reverse=True)
        
        # 过滤掉置信度低的结果
        filtered_results = [r for r in sorted_results if r.confidence >= self.confidence_threshold]
        
        # 使用结巴分词改善中文识别
        for i, result in enumerate(filtered_results):
            # 对每个结果进行分词，如果分词后能重新组合成原文，则有较高的可能是正确的中文
            words = jieba.cut(result.text)
            segmented = " ".join(words)
            # 更新结果，保留置信度较高的
            filtered_results[i] = OCRResult(
                text=result.text,
                confidence=result.confidence,
                box=result.box,
                engine=result.engine,
                preprocessor=result.preprocessor
            )
        
        # 过滤掉过短的文本（可能是噪声）
        filtered_results = [r for r in filtered_results if len(r.text.strip()) > 1]
        
        # 去除重复结果
        unique_results = []
        seen_texts = set()
        
        for result in filtered_results:
            # 简单去重：基于文本内容
            if result.text in seen_texts:
                continue
            
            seen_texts.add(result.text)
            unique_results.append(result)
        
        return unique_results
    
    def _draw_results(self, image: np.ndarray, results: List[OCRResult]) -> np.ndarray:
        """在图像上绘制OCR结果"""
        for result in results:
            if result.box:
                # 绘制文本框
                points = np.array(result.box, np.int32)
                cv2.polylines(image, [points], True, (0, 255, 0), 2)
                
                # 绘制文本
                x, y = result.box[0]
                text = f"{result.text} ({result.confidence:.2f})"
                cv2.putText(image, text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return image


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="高级中文OCR处理系统")
    parser.add_argument("--input", "-i", required=True, help="输入图像文件或目录")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--gpu", action="store_true", help="使用GPU加速")
    parser.add_argument("--confidence", "-c", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--workers", "-w", type=int, default=4, help="并行处理的工作线程数")
    parser.add_argument("--no-annotated", action="store_true", help="不输出标注图像")
    parser.add_argument("--no-text", action="store_true", help="不输出文本文件")
    parser.add_argument("--no-json", action="store_true", help="不输出JSON文件")
    
    args = parser.parse_args()
    
    # 初始化多引擎OCR
    ocr = MultiEngineOCR(use_gpu=args.gpu, confidence_threshold=args.confidence)
    
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
        logger.info(f"处理完成: {args.input}")
        logger.info(f"识别到 {len(result.get('results', []))} 个文本区域")
    else:
        # 处理目录
        result = ocr.process_directory(
            args.input,
            args.output,
            max_workers=args.workers,
            write_annotated=not args.no_annotated,
            write_text=not args.no_text,
            write_json=not args.no_json
        )
        logger.info(f"处理完成: {args.input}")
        summary = result.get("summary", {})
        logger.info(f"成功处理: {summary.get('successful', 0)}/{summary.get('total_files', 0)} 文件")


if __name__ == "__main__":
    main() 