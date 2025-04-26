#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EasyOCR引擎测试
"""

import os
import sys
import unittest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入待测试模块
try:
    from ocr.engines.easyocr_engine import EasyOCREngine, EASYOCR_AVAILABLE
except ImportError:
    EASYOCR_AVAILABLE = False


@unittest.skipIf(not EASYOCR_AVAILABLE, "EasyOCR不可用，跳过测试")
class TestEasyOCREngine(unittest.TestCase):
    """EasyOCR引擎测试类"""
    
    def setUp(self):
        """测试准备"""
        # 创建测试图像
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # 在图像上添加文本
        cv2.putText(self.test_image, "测试文本", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 模拟EasyOCR结果
        self.mock_result = [
            ([
                [10, 10], [90, 10], [90, 50], [10, 50]
            ], "测试文本", 0.95)
        ]
    
    @patch('easyocr.Reader')
    def test_init(self, mock_reader):
        """测试初始化"""
        # 设置mock对象行为
        mock_reader.return_value = MagicMock()
        
        # 初始化引擎
        engine = EasyOCREngine(langs=["ch_sim"], gpu=False)
        
        # 验证初始化结果
        self.assertIsNotNone(engine)
        self.assertEqual(engine.langs, ["ch_sim"])
        self.assertFalse(engine.gpu)
        # 懒加载，此时应该尚未创建实际reader
        self.assertIsNone(engine._reader)
    
    @patch('easyocr.Reader')
    def test_recognize(self, mock_reader):
        """测试识别方法"""
        # 设置mock对象行为
        mock_instance = MagicMock()
        mock_reader.return_value = mock_instance
        mock_instance.readtext.return_value = self.mock_result
        
        # 初始化引擎并进行识别
        engine = EasyOCREngine(langs=["ch_sim"], gpu=False)
        results = engine.recognize(self.test_image)
        
        # 验证结果
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "测试文本")
        self.assertEqual(results[0].confidence, 0.95)
        self.assertEqual(results[0].engine, "easyocr")
        self.assertEqual(results[0].preprocessor, "original")
        
        # 验证mock方法被调用
        mock_reader.assert_called_once()
        mock_instance.readtext.assert_called_once()
    
    def test_is_available(self):
        """测试可用性检查"""
        if EASYOCR_AVAILABLE:
            engine = EasyOCREngine(langs=["ch_sim"], gpu=False)
            self.assertTrue(engine.is_available())


if __name__ == "__main__":
    unittest.main() 