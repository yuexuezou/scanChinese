#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OCR性能基准测试工具
测试各个OCR引擎的性能表现，包括速度、准确率等指标
"""

import os
import time
import json
import logging
import argparse
from typing import Dict, List, Any, Optional
import concurrent.futures
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

# 尝试导入项目根目录
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.engines import get_available_engines, get_engine_by_name
from ocr.multi_engine import MultiEngineOCR
from ocr.base import NumpyEncoder

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OCRBenchmark")


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    engine_name: str
    total_time: float
    avg_time_per_image: float
    num_images: int
    num_texts_detected: int
    avg_confidence: float
    error_rate: float
    memory_usage: Optional[float] = None
    gpu_usage: Optional[float] = None


class OCRBenchmark:
    """OCR性能基准测试类"""
    
    def __init__(self, test_images_dir: str, output_dir: str, 
                engines: Optional[List[str]] = None, use_gpu: bool = False):
        """
        初始化基准测试类
        
        参数:
            test_images_dir: 测试图像目录
            output_dir: 输出目录
            engines: 要测试的引擎列表，默认为所有可用引擎
            use_gpu: 是否使用GPU
        """
        self.test_images_dir = test_images_dir
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        
        # 获取可用引擎
        available_engines = get_available_engines()
        logger.info(f"可用引擎: {', '.join(available_engines)}")
        
        # 设置要测试的引擎
        self.engines_to_test = engines if engines else available_engines
        
        # 过滤不可用的引擎
        for engine in self.engines_to_test[:]:
            if engine not in available_engines:
                logger.warning(f"引擎 {engine} 不可用，将从测试中排除")
                self.engines_to_test.remove(engine)
        
        if not self.engines_to_test:
            raise ValueError(f"没有可用的引擎进行测试")
        
        logger.info(f"将测试以下引擎: {', '.join(self.engines_to_test)}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有测试图像
        self.test_images = []
        for root, _, files in os.walk(test_images_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.test_images.append(os.path.join(root, file))
        
        logger.info(f"找到 {len(self.test_images)} 个测试图像")
        
        if not self.test_images:
            raise ValueError(f"在目录 {test_images_dir} 中未找到任何图像")
    
    def run(self) -> Dict[str, BenchmarkResult]:
        """运行基准测试"""
        results = {}
        
        # 为每个引擎运行测试
        for engine_name in self.engines_to_test:
            logger.info(f"测试引擎: {engine_name}")
            
            try:
                engine = get_engine_by_name(engine_name)
                
                start_time = time.time()
                total_texts = 0
                total_confidence = 0.0
                errors = 0
                
                for img_path in self.test_images:
                    try:
                        result = engine.process_image(
                            img_path, 
                            os.path.join(self.output_dir, engine_name),
                            write_annotated=True,
                            write_text=True,
                            write_json=True
                        )
                        
                        if "error" in result:
                            logger.error(f"处理图像 {img_path} 时出错: {result['error']}")
                            errors += 1
                            continue
                        
                        # 统计识别的文本数量和平均置信度
                        num_texts = len(result.get("results", []))
                        total_texts += num_texts
                        
                        if num_texts > 0:
                            avg_confidence = sum(r.confidence for r in result["results"]) / num_texts
                            total_confidence += avg_confidence
                    
                    except Exception as e:
                        logger.error(f"处理图像 {img_path} 时出现异常: {str(e)}")
                        errors += 1
                
                end_time = time.time()
                total_time = end_time - start_time
                
                # 计算结果
                result = BenchmarkResult(
                    engine_name=engine_name,
                    total_time=total_time,
                    avg_time_per_image=total_time / len(self.test_images) if self.test_images else 0,
                    num_images=len(self.test_images),
                    num_texts_detected=total_texts,
                    avg_confidence=total_confidence / len(self.test_images) if total_texts > 0 and self.test_images else 0,
                    error_rate=errors / len(self.test_images) if self.test_images else 0
                )
                
                results[engine_name] = result
                logger.info(f"引擎 {engine_name} 测试完成: 总时间={total_time:.2f}秒, "
                           f"平均时间={result.avg_time_per_image:.2f}秒/图, "
                           f"检测到{total_texts}个文本, "
                           f"平均置信度={result.avg_confidence:.2f}, "
                           f"错误率={result.error_rate:.2f}")
            
            except Exception as e:
                logger.error(f"测试引擎 {engine_name} 时出现错误: {str(e)}")
        
        # 保存测试结果
        self._save_results(results)
        
        # 生成对比图表
        self._generate_charts(results)
        
        return results
    
    def _save_results(self, results: Dict[str, BenchmarkResult]) -> None:
        """保存测试结果"""
        # 将结果转换为字典
        results_dict = {
            name: {
                "engine_name": result.engine_name,
                "total_time": result.total_time,
                "avg_time_per_image": result.avg_time_per_image,
                "num_images": result.num_images,
                "num_texts_detected": result.num_texts_detected,
                "avg_confidence": result.avg_confidence,
                "error_rate": result.error_rate,
                "memory_usage": result.memory_usage,
                "gpu_usage": result.gpu_usage
            }
            for name, result in results.items()
        }
        
        # 保存为JSON文件
        results_path = os.path.join(self.output_dir, "benchmark_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        logger.info(f"测试结果已保存至 {results_path}")
    
    def _generate_charts(self, results: Dict[str, BenchmarkResult]) -> None:
        """生成对比图表"""
        if not results:
            return
        
        # 提取数据
        engine_names = list(results.keys())
        avg_times = [results[name].avg_time_per_image for name in engine_names]
        avg_confidences = [results[name].avg_confidence for name in engine_names]
        error_rates = [results[name].error_rate for name in engine_names]
        
        # 创建图表目录
        charts_dir = os.path.join(self.output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # 绘制平均处理时间图表
        plt.figure(figsize=(10, 6))
        plt.bar(engine_names, avg_times, color='skyblue')
        plt.xlabel('OCR引擎')
        plt.ylabel('平均处理时间 (秒/图)')
        plt.title('各OCR引擎平均处理时间对比')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "avg_time_comparison.png"))
        plt.close()
        
        # 绘制平均置信度图表
        plt.figure(figsize=(10, 6))
        plt.bar(engine_names, avg_confidences, color='lightgreen')
        plt.xlabel('OCR引擎')
        plt.ylabel('平均置信度')
        plt.title('各OCR引擎平均置信度对比')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "avg_confidence_comparison.png"))
        plt.close()
        
        # 绘制错误率图表
        plt.figure(figsize=(10, 6))
        plt.bar(engine_names, error_rates, color='salmon')
        plt.xlabel('OCR引擎')
        plt.ylabel('错误率')
        plt.title('各OCR引擎错误率对比')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "error_rate_comparison.png"))
        plt.close()
        
        logger.info(f"对比图表已生成至 {charts_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="OCR性能基准测试工具")
    parser.add_argument("--images", "-i", required=True, help="测试图像目录")
    parser.add_argument("--output", "-o", default="benchmark_results", help="输出目录")
    parser.add_argument("--engines", "-e", help="要测试的引擎，用逗号分隔")
    parser.add_argument("--gpu", action="store_true", help="使用GPU加速")
    parser.add_argument("--multi-engine", "-m", action="store_true", help="测试多引擎模式")
    
    args = parser.parse_args()
    
    try:
        # 解析引擎列表
        engines = args.engines.split(",") if args.engines else None
        
        # 创建基准测试对象
        benchmark = OCRBenchmark(
            test_images_dir=args.images,
            output_dir=args.output,
            engines=engines,
            use_gpu=args.gpu
        )
        
        # 运行测试
        results = benchmark.run()
        
        # 如果指定了多引擎模式，则测试多引擎
        if args.multi_engine:
            logger.info("开始测试多引擎模式")
            
            multi_output_dir = os.path.join(args.output, "multi_engine")
            os.makedirs(multi_output_dir, exist_ok=True)
            
            # 初始化多引擎OCR
            try:
                ocr = MultiEngineOCR(engines=engines, use_gpu=args.gpu)
                
                start_time = time.time()
                total_texts = 0
                total_confidence = 0.0
                errors = 0
                
                for img_path in benchmark.test_images:
                    try:
                        result = ocr.process_image(
                            img_path, 
                            multi_output_dir,
                            write_annotated=True,
                            write_text=True,
                            write_json=True
                        )
                        
                        if "error" in result:
                            logger.error(f"处理图像 {img_path} 时出错: {result['error']}")
                            errors += 1
                            continue
                        
                        # 统计识别的文本数量和平均置信度
                        num_texts = len(result.get("results", []))
                        total_texts += num_texts
                        
                        if num_texts > 0:
                            avg_confidence = sum(r.confidence for r in result["results"]) / num_texts
                            total_confidence += avg_confidence
                    
                    except Exception as e:
                        logger.error(f"处理图像 {img_path} 时出现异常: {str(e)}")
                        errors += 1
                
                end_time = time.time()
                total_time = end_time - start_time
                
                # 计算结果
                multi_result = BenchmarkResult(
                    engine_name="multi_engine",
                    total_time=total_time,
                    avg_time_per_image=total_time / len(benchmark.test_images) if benchmark.test_images else 0,
                    num_images=len(benchmark.test_images),
                    num_texts_detected=total_texts,
                    avg_confidence=total_confidence / len(benchmark.test_images) if total_texts > 0 and benchmark.test_images else 0,
                    error_rate=errors / len(benchmark.test_images) if benchmark.test_images else 0
                )
                
                # 将多引擎结果添加到总结果
                results["multi_engine"] = multi_result
                
                # 保存更新后的结果
                benchmark._save_results(results)
                benchmark._generate_charts(results)
                
                logger.info(f"多引擎测试完成: 总时间={total_time:.2f}秒, "
                           f"平均时间={multi_result.avg_time_per_image:.2f}秒/图, "
                           f"检测到{total_texts}个文本, "
                           f"平均置信度={multi_result.avg_confidence:.2f}, "
                           f"错误率={multi_result.error_rate:.2f}")
            
            except Exception as e:
                logger.error(f"测试多引擎模式时出现错误: {str(e)}")
        
        logger.info("基准测试完成")
    
    except Exception as e:
        logger.error(f"基准测试过程中出现错误: {str(e)}")


if __name__ == "__main__":
    main() 