#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多引擎OCR整合模块
整合多个OCR引擎的结果，提供统一的接口
"""

import os
import cv2
import json
import time
import jieba
import logging
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from .base import OCRResult, NumpyEncoder
from .preprocessor import ImagePreprocessor
from .engines import get_engine_by_name, get_available_engines

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MultiEngineOCR")


class MultiEngineOCR:
    """多引擎OCR整合类"""
    
    def __init__(self, engines=None, use_gpu=False, confidence_threshold=0.5):
        """
        初始化多引擎OCR系统
        
        参数:
            engines: 要使用的引擎列表，默认为所有可用引擎
            use_gpu: 是否使用GPU加速
            confidence_threshold: 置信度阈值
        """
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.preprocessor = ImagePreprocessor()
        
        # 初始化指定的引擎，如果未指定则使用所有可用引擎
        self.engines = []
        self._init_engines(engines)
        
        if not self.engines:
            logger.warning("没有可用的OCR引擎")
    
    def _init_engines(self, engine_names=None):
        """初始化引擎"""
        available_engines = get_available_engines()
        
        if not available_engines:
            logger.error("没有可用的OCR引擎，请安装至少一个OCR引擎")
            return
        
        # 如果未指定引擎，则使用所有可用引擎
        if engine_names is None:
            engine_names = available_engines
        
        # 初始化每个引擎
        for name in engine_names:
            if name not in available_engines:
                logger.warning(f"引擎 {name} 不可用，将被跳过")
                continue
            
            try:
                # 针对不同引擎的参数设置
                if name == "tesseract":
                    engine = get_engine_by_name(name, lang="chi_sim+chi_tra")
                elif name == "paddle":
                    engine = get_engine_by_name(name, use_gpu=self.use_gpu, lang="ch")
                elif name == "easyocr":
                    engine = get_engine_by_name(name, langs=["ch_sim", "en"], gpu=self.use_gpu)
                elif name == "rapidocr":
                    engine = get_engine_by_name(name)
                elif name == "manga":
                    engine = get_engine_by_name(name, force_cpu=not self.use_gpu)
                elif name == "mmocr":
                    engine = get_engine_by_name(name, device="cuda" if self.use_gpu else "cpu")
                elif name == "trocr":
                    engine = get_engine_by_name(name, model_name="microsoft/trocr-base-handwritten", use_gpu=self.use_gpu)
                elif name == "donut":
                    engine = get_engine_by_name(name, model_name="naver-clova-ix/donut-base-finetuned-cord-v2", use_gpu=self.use_gpu)
                else:
                    engine = get_engine_by_name(name)
                
                self.engines.append(engine)
                logger.info(f"已初始化引擎: {engine.get_name()}")
            except Exception as e:
                logger.error(f"初始化引擎 {name} 失败: {str(e)}")
    
    def process_image(self, image_path: str, output_dir: str = None, 
                     write_annotated: bool = True, write_text: bool = True, 
                     write_json: bool = True) -> Dict[str, Any]:
        """
        处理单个图像并返回结果
        
        参数:
            image_path: 输入图像路径
            output_dir: 输出目录
            write_annotated: 是否输出标注图像
            write_text: 是否输出文本文件
            write_json: 是否输出JSON文件
            
        返回:
            包含处理结果的字典
        """
        start_time = time.time()
        
        # 准备输出目录
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = os.path.dirname(image_path) or "."
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return {"error": f"无法读取图像: {image_path}"}
        
        # 获取图像文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 应用所有预处理器
        preprocessed_images = self.preprocessor.apply_all_preprocessors(image)
        
        # 存储所有识别结果
        all_results = []
        results_by_engine = {}
        
        # 对每个引擎的每种预处理方法运行OCR
        for engine in self.engines:
            engine_name = engine.get_name()
            engine_results = []
            
            for preproc_name, preproc_image in preprocessed_images.items():
                try:
                    logger.info(f"使用 {engine_name} 处理 {preproc_name} 图像")
                    results = engine.recognize(preproc_image, preproc_name)
                    engine_results.extend(results)
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"使用 {engine_name} 处理 {preproc_name} 图像时出错: {str(e)}")
            
            results_by_engine[engine_name] = engine_results
        
        # 整合和优化结果
        optimized_results = self._optimize_results(all_results)
        
        # 写出结果
        output_files = {}
        
        # 生成带标注的图像
        if write_annotated:
            annotated_img = self._draw_results(image.copy(), optimized_results)
            annotated_path = os.path.join(output_dir, f"{base_name}_multi_annotated.jpg")
            cv2.imwrite(annotated_path, annotated_img)
            output_files["annotated_image"] = annotated_path
            
            # 为每个引擎生成单独的标注图像
            for engine_name, engine_results in results_by_engine.items():
                if engine_results:
                    engine_annotated = self._draw_results(image.copy(), engine_results)
                    engine_annotated_path = os.path.join(output_dir, f"{base_name}_{engine_name}_annotated.jpg")
                    cv2.imwrite(engine_annotated_path, engine_annotated)
                    output_files[f"annotated_image_{engine_name}"] = engine_annotated_path
        
        # 生成纯文本结果
        if write_text:
            text_content = "\n".join([r.text for r in optimized_results])
            text_path = os.path.join(output_dir, f"{base_name}_multi_result.txt")
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
                        "confidence": float(r.confidence),
                        "box": r.box,
                        "engine": r.engine,
                        "preprocessor": r.preprocessor
                    } for r in optimized_results
                ],
                "results_by_engine": {
                    engine_name: [
                        {
                            "text": r.text,
                            "confidence": float(r.confidence),
                            "box": r.box,
                            "engine": r.engine,
                            "preprocessor": r.preprocessor
                        } for r in engine_results
                    ] for engine_name, engine_results in results_by_engine.items()
                }
            }
            json_path = os.path.join(output_dir, f"{base_name}_multi_result.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            output_files["json_file"] = json_path
        
        return {
            "results": optimized_results,
            "results_by_engine": results_by_engine,
            "processing_time": time.time() - start_time,
            "output_files": output_files
        }
    
    def process_directory(self, input_dir: str, output_dir: str = None, 
                         extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp"],
                         max_workers: int = 4, **kwargs) -> Dict[str, Any]:
        """
        处理目录中的所有图像
        
        参数:
            input_dir: 输入目录
            output_dir: 输出目录
            extensions: 要处理的文件扩展名列表
            max_workers: 并行处理的线程数
            **kwargs: 传递给process_image的其他参数
            
        返回:
            包含处理结果的字典
        """
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
            json.dump(summary, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        return {
            "summary": summary,
            "results": results
        }
    
    def _optimize_results(self, results: List[OCRResult]) -> List[OCRResult]:
        """
        整合和优化OCR结果
        
        参数:
            results: OCR结果列表
            
        返回:
            优化后的OCR结果列表
        """
        if not results:
            return []
        
        # 按置信度排序
        sorted_results = sorted(results, key=lambda x: x.confidence, reverse=True)
        
        # 过滤掉置信度低的结果
        filtered_results = [r for r in sorted_results if r.confidence >= self.confidence_threshold]
        
        # 使用结巴分词改善中文识别
        for i, result in enumerate(filtered_results):
            # 对每个结果进行分词
            words = jieba.cut(result.text)
            segmented = " ".join(words)
            # 更新结果，保留原文本和置信度
            filtered_results[i] = OCRResult(
                text=result.text,
                confidence=float(result.confidence),
                box=result.box,
                engine=result.engine,
                preprocessor=result.preprocessor
            )
        
        # 过滤掉过短的文本（可能是噪声）
        filtered_results = [r for r in filtered_results if len(r.text.strip()) > 1]
        
        # 去除重复结果，合并重叠区域的结果
        unique_results = []
        seen_texts = set()
        overlapping_boxes = defaultdict(list)
        
        # 收集重叠框中的所有候选结果
        for i, result in enumerate(filtered_results):
            # 如果文本完全相同，则直接忽略
            if result.text in seen_texts:
                continue
            
            has_overlap = False
            for j, existing in enumerate(unique_results):
                # 检查与现有结果框是否重叠
                if self._boxes_overlap(result.box, existing.box):
                    has_overlap = True
                    overlapping_boxes[j].append(result)
                    break
            
            if not has_overlap:
                seen_texts.add(result.text)
                unique_results.append(result)
                overlapping_boxes[len(unique_results) - 1].append(result)
        
        # 对于每个重叠区域，选择置信度最高的结果
        for idx, candidates in overlapping_boxes.items():
            if candidates:
                best_candidate = max(candidates, key=lambda x: x.confidence)
                unique_results[idx] = best_candidate
        
        return unique_results
    
    def _boxes_overlap(self, box1, box2, threshold=0.5):
        """检查两个框是否重叠"""
        if not box1 or not box2:
            return False
        
        # 计算两个框的面积
        def polygon_area(vertices):
            x = [vertex[0] for vertex in vertices]
            y = [vertex[1] for vertex in vertices]
            return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(len(vertices) - 1)))
        
        # 计算两个框的交集面积
        def calculate_overlap(box1, box2):
            import shapely.geometry
            poly1 = shapely.geometry.Polygon(box1)
            poly2 = shapely.geometry.Polygon(box2)
            
            if not poly1.is_valid or not poly2.is_valid:
                return 0
            
            try:
                intersection = poly1.intersection(poly2).area
                return intersection / min(poly1.area, poly2.area)
            except:
                return 0
        
        try:
            overlap_ratio = calculate_overlap(box1, box2)
            return overlap_ratio > threshold
        except:
            # 如果计算失败，使用简单的中心点距离判断
            def box_center(box):
                return [sum(p[0] for p in box) / len(box), sum(p[1] for p in box) / len(box)]
            
            center1 = box_center(box1)
            center2 = box_center(box2)
            
            # 计算两个中心点的距离
            dist = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
            
            # 估算框的大小
            def box_size(box):
                x_vals = [p[0] for p in box]
                y_vals = [p[1] for p in box]
                return max(x_vals) - min(x_vals), max(y_vals) - min(y_vals)
            
            size1 = box_size(box1)
            size2 = box_size(box2)
            
            # 如果中心点距离小于框大小的平均值的一半，认为有重叠
            avg_size = (size1[0] + size1[1] + size2[0] + size2[1]) / 4
            return dist < avg_size * 0.5
    
    def _draw_results(self, image: np.ndarray, results: List[OCRResult]) -> np.ndarray:
        """
        在图像上绘制OCR结果
        
        参数:
            image: 输入图像
            results: OCR结果列表
            
        返回:
            标注后的图像
        """
        # 为不同引擎设置不同颜色
        engine_colors = {
            "tesseract": (0, 255, 0),  # 绿色
            "paddleocr": (0, 0, 255),  # 红色
            "easyocr": (255, 0, 0),    # 蓝色
            "rapidocr": (255, 255, 0)  # 青色
        }
        
        for result in results:
            if result.box:
                # 获取引擎对应的颜色，默认为绿色
                color = engine_colors.get(result.engine.lower(), (0, 255, 0))
                
                # 绘制文本框
                try:
                    pts = np.array(result.box, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], True, color, 2)
                    
                    # 绘制文本和引擎标识
                    x, y = result.box[0]
                    text = f"{result.text} ({result.confidence:.2f})"
                    cv2.putText(image, text, (int(x), max(int(y) - 10, 10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # 标注引擎和预处理方法
                    engine_text = f"{result.engine}/{result.preprocessor}"
                    cv2.putText(image, engine_text, (int(x), max(int(y) - 30, 10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                except Exception as e:
                    logger.error(f"绘制结果时出错: {str(e)}")
        
        return image


def main():
    """多引擎OCR命令行入口"""
    parser = argparse.ArgumentParser(description="多引擎中文OCR处理系统")
    parser.add_argument("--input", "-i", required=True, help="输入图像文件或目录")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--engines", "-e", help="要使用的引擎，用逗号分隔，如 'tesseract,easyocr'")
    parser.add_argument("--gpu", action="store_true", help="使用GPU加速")
    parser.add_argument("--confidence", "-c", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--workers", "-w", type=int, default=4, help="并行处理的工作线程数")
    parser.add_argument("--no-annotated", action="store_true", help="不输出标注图像")
    parser.add_argument("--no-text", action="store_true", help="不输出文本文件")
    parser.add_argument("--no-json", action="store_true", help="不输出JSON文件")
    
    args = parser.parse_args()
    
    try:
        # 解析引擎列表
        engines = args.engines.split(",") if args.engines else None
        
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
            
            if "error" in result:
                logger.error(result["error"])
            else:
                logger.info(f"处理完成: {args.input}")
                summary = result.get("summary", {})
                logger.info(f"成功处理: {summary.get('successful', 0)}/{summary.get('total_files', 0)} 文件")
    
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 