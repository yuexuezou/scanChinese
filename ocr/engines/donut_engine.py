#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Donut OCR引擎模块
封装Donut OCR引擎的功能，专为文档和表格理解设计
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
logger = logging.getLogger("DonutEngine")

# 检查Donut依赖是否可用
try:
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    from PIL import Image
    DONUT_AVAILABLE = True
except ImportError:
    DONUT_AVAILABLE = False
    warnings.warn("Donut依赖未安装，请使用 pip install transformers torch 安装")


class DonutEngine(OCREngine):
    """Donut引擎封装，专为文档分析而设计"""
    
    def __init__(self, model_name="naver-clova-ix/donut-base-finetuned-cord-v2", use_gpu=None):
        """
        初始化Donut引擎
        
        参数:
            model_name: 模型名称，默认使用预训练的收据分析模型
                其他可选模型: "naver-clova-ix/donut-base", "naver-clova-ix/donut-base-finetuned-docvqa"
            use_gpu: 是否使用GPU（None表示自动检测，True强制使用，False强制CPU）
        """
        if not self.is_available():
            raise ImportError("Donut不可用，请安装: pip install transformers torch")
        
        self.model_name = model_name
        self.preprocessor = ImagePreprocessor()
        
        # 设置设备
        self.device = self._get_device(use_gpu)
        logger.info(f"使用设备: {self.device}")
        
        # 懒加载，只在第一次recognize时初始化模型
        self._model = None
        self._processor = None
        
        logger.info(f"DonutEngine 初始化完成，模型: {model_name}, 设备: {self.device}")
    
    def _get_device(self, use_gpu):
        """确定使用的设备"""
        if use_gpu is False:
            return "cpu"
        
        if torch.cuda.is_available() and use_gpu is not False:
            return "cuda"
        
        return "cpu"
    
    def is_available(self) -> bool:
        """检查引擎是否可用"""
        return DONUT_AVAILABLE
    
    @property
    def processor(self):
        """懒加载Donut处理器"""
        if self._processor is None:
            try:
                logger.info(f"初始化Donut处理器: {self.model_name}")
                self._processor = DonutProcessor.from_pretrained(self.model_name)
            except Exception as e:
                logger.error(f"初始化Donut处理器失败: {str(e)}")
                raise
        return self._processor
    
    @property
    def model(self):
        """懒加载Donut模型"""
        if self._model is None:
            try:
                logger.info(f"加载Donut模型: {self.model_name}")
                self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
                self._model.to(self.device)
            except Exception as e:
                logger.error(f"加载Donut模型失败: {str(e)}")
                raise
        return self._model
    
    def recognize(self, image: np.ndarray, preprocessor_name: str = "original") -> List[OCRResult]:
        """
        使用Donut识别文本
        
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
        
        # 转换为PIL Image
        if len(image.shape) == 2:  # 灰度图像
            pil_image = Image.fromarray(image)
        else:  # 彩色图像
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Donut处理
        try:
            # 设置任务提示，根据不同模型需要适当调整
            task_prompt = "<s_cord-v2>"
            if "docvqa" in self.model_name:
                task_prompt = "<s_docvqa>"
            
            # 预处理图像
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.to(self.device)
            
            # 设置解码参数
            decoder_input_ids = self.processor.tokenizer(
                task_prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(self.device)
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=self.model.decoder.config.max_position_embeddings,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )
            
            # 解码结果
            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            sequence = sequence.replace(task_prompt, "")
            
            # 处理结果
            if sequence:
                height, width = image.shape[:2]
                box = [[0, 0], [width, 0], [width, height], [0, height]]
                confidence = 0.9  # 默认置信度
                
                results.append(
                    OCRResult(
                        text=sequence,
                        confidence=confidence,
                        box=box,
                        engine="donut",
                        preprocessor=preprocessor_name
                    )
                )
        except Exception as e:
            logger.error(f"Donut识别出错: {str(e)}")
        
        return results


def main():
    """Donut引擎命令行入口"""
    parser = argparse.ArgumentParser(description="Donut引擎 - 文档理解OCR引擎")
    parser.add_argument("--input", "-i", required=True, help="输入图像文件路径")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--model", "-m", default="naver-clova-ix/donut-base-finetuned-cord-v2", 
                       help="模型名称，如'naver-clova-ix/donut-base-finetuned-docvqa'")
    parser.add_argument("--cpu", action="store_true", help="强制使用CPU")
    parser.add_argument("--preprocess", "-p", help="应用预处理方法")
    parser.add_argument("--no-annotated", action="store_true", help="不输出标注图像")
    parser.add_argument("--no-text", action="store_true", help="不输出文本文件")
    parser.add_argument("--no-json", action="store_true", help="不输出JSON文件")
    
    args = parser.parse_args()
    
    try:
        # 初始化引擎
        engine = DonutEngine(model_name=args.model, use_gpu=not args.cpu)
        
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