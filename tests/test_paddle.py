#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PaddleOCR引擎测试
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
    from ocr.engines.paddle_engine import PaddleOCREngine, PADDLE_AVAILABLE
except ImportError:
    PADDLE_AVAILABLE = False


@unittest.skipIf(not PADDLE_AVAILABLE, "PaddleOCR不可用，跳过测试")
class TestPaddleOCREngine(unittest.TestCase):
    """PaddleOCR引擎测试类"""
    
    def setUp(self):
        """测试准备"""
        # 创建测试图像
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # 在图像上添加文本
        cv2.putText(self.test_image, "测试文本", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 模拟PaddleOCR结果
        self.mock_result = [
            {
                'boxes': [[10, 10], [90, 10], [90, 50], [10, 50]],
                'text': '测试文本',
                'confidence': 0.95
            }
        ]
    
    @patch('paddleocr.PaddleOCR')
    def test_init(self, mock_paddle):
        """测试初始化"""
        # 设置mock对象行为
        mock_paddle.return_value = MagicMock()
        
        # 初始化引擎
        engine = PaddleOCREngine(langs="ch", gpu=False)
        
        # 验证初始化结果
        self.assertIsNotNone(engine)
        self.assertEqual(engine.langs, "ch")
        self.assertFalse(engine.gpu)
        # 懒加载，此时应该尚未创建实际paddle ocr实例
        self.assertIsNone(engine._ocr)
    
    @patch('paddleocr.PaddleOCR')
    def test_recognize(self, mock_paddle):
        """测试识别方法"""
        # 设置mock对象行为
        mock_instance = MagicMock()
        mock_paddle.return_value = mock_instance
        mock_instance.ocr.return_value = [self.mock_result]
        
        # 初始化引擎并进行识别
        engine = PaddleOCREngine(langs="ch", gpu=False)
        results = engine.recognize(self.test_image)
        
        # 验证结果
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "测试文本")
        self.assertAlmostEqual(results[0].confidence, 0.95)
        self.assertEqual(results[0].engine, "paddle")
        self.assertEqual(results[0].preprocessor, "original")
        
        # 验证mock方法被调用
        mock_paddle.assert_called_once()
        mock_instance.ocr.assert_called_once()
    
    def test_is_available(self):
        """测试可用性检查"""
        if PADDLE_AVAILABLE:
            engine = PaddleOCREngine(langs="ch", gpu=False)
            self.assertTrue(engine.is_available())


if __name__ == "__main__":
    unittest.main() 