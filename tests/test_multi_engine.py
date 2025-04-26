#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MultiEngineOCR模块测试
"""

import os
import sys
import unittest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.multi_engine import MultiEngineOCR
from ocr.base import OCRResult, BoundingBox


class TestMultiEngineOCR(unittest.TestCase):
    """MultiEngineOCR测试类"""
    
    def setUp(self):
        """测试准备"""
        # 创建测试图像
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # 在图像上添加文本
        cv2.putText(self.test_image, "测试文本", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 创建临时目录用于输出
        self.temp_output_dir = os.path.join(os.path.dirname(__file__), "temp_output")
        if not os.path.exists(self.temp_output_dir):
            os.makedirs(self.temp_output_dir)
    
    def tearDown(self):
        """测试清理"""
        # 清理临时目录(可选，取决于测试需求)
        pass
    
    @patch('ocr.engines.get_available_engines')
    @patch('ocr.engines.easyocr_engine.EasyOCREngine')
    def test_init(self, mock_easyocr_engine, mock_get_engines):
        """测试初始化"""
        # 模拟可用引擎
        mock_get_engines.return_value = ["easyocr"]
        
        # 模拟EasyOCR引擎实例
        mock_engine_instance = MagicMock()
        mock_easyocr_engine.return_value = mock_engine_instance
        mock_engine_instance.is_available.return_value = True
        
        # 初始化MultiEngineOCR
        ocr = MultiEngineOCR(engines=["easyocr"], use_gpu=False)
        
        # 验证初始化结果
        self.assertIsNotNone(ocr)
        self.assertEqual(len(ocr.engines), 1)
    
    @patch('ocr.engines.get_available_engines')
    @patch('ocr.engines.easyocr_engine.EasyOCREngine')
    def test_process_image(self, mock_easyocr_engine, mock_get_engines):
        """测试处理单张图像"""
        # 模拟可用引擎
        mock_get_engines.return_value = ["easyocr"]
        
        # 模拟EasyOCR引擎实例
        mock_engine_instance = MagicMock()
        mock_easyocr_engine.return_value = mock_engine_instance
        mock_engine_instance.is_available.return_value = True
        
        # 模拟识别结果
        bbox = BoundingBox([[10, 10], [90, 10], [90, 50], [10, 50]])
        ocr_result = OCRResult(
            bbox=bbox,
            text="测试文本",
            confidence=0.95,
            engine="easyocr",
            preprocessor="original"
        )
        mock_engine_instance.recognize.return_value = [ocr_result]
        
        # 初始化MultiEngineOCR并处理图像
        ocr = MultiEngineOCR(engines=["easyocr"], use_gpu=False)
        result = ocr.process_image(
            self.test_image,
            output_dir=self.temp_output_dir,
            write_annotated=True,
            write_text=True,
            write_json=True
        )
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["text"], "测试文本")
        self.assertAlmostEqual(result["results"][0]["confidence"], 0.95)
        
        # 验证输出文件存在
        self.assertIn("output_files", result)
        
    @patch('ocr.engines.get_available_engines')
    @patch('ocr.engines.easyocr_engine.EasyOCREngine')
    @patch('ocr.engines.tesseract_engine.TesseractOCREngine')
    def test_ensemble_results(self, mock_tesseract, mock_easyocr, mock_get_engines):
        """测试结果融合"""
        # 模拟可用引擎
        mock_get_engines.return_value = ["easyocr", "tesseract"]
        
        # 模拟引擎实例
        mock_easyocr_instance = MagicMock()
        mock_easyocr.return_value = mock_easyocr_instance
        mock_easyocr_instance.is_available.return_value = True
        
        mock_tesseract_instance = MagicMock()
        mock_tesseract.return_value = mock_tesseract_instance
        mock_tesseract_instance.is_available.return_value = True
        
        # 模拟识别结果
        bbox1 = BoundingBox([[10, 10], [90, 10], [90, 50], [10, 50]])
        result1 = OCRResult(
            bbox=bbox1,
            text="测试文本1",
            confidence=0.9,
            engine="easyocr",
            preprocessor="original"
        )
        mock_easyocr_instance.recognize.return_value = [result1]
        
        bbox2 = BoundingBox([[10, 10], [90, 10], [90, 50], [10, 50]])
        result2 = OCRResult(
            bbox=bbox2,
            text="测试文本2",
            confidence=0.8,
            engine="tesseract",
            preprocessor="original"
        )
        mock_tesseract_instance.recognize.return_value = [result2]
        
        # 初始化MultiEngineOCR
        ocr = MultiEngineOCR(engines=["easyocr", "tesseract"], use_gpu=False)
        
        # 测试结果融合
        all_results = [result1, result2]
        ensembled = ocr._ensemble_results(all_results)
        
        # 验证融合结果
        self.assertIsNotNone(ensembled)
        self.assertGreaterEqual(len(ensembled), 1)


if __name__ == "__main__":
    unittest.main() 