#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tesseract引擎测试
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
    from ocr.engines.tesseract_engine import TesseractOCREngine, TESSERACT_AVAILABLE
except ImportError:
    TESSERACT_AVAILABLE = False


@unittest.skipIf(not TESSERACT_AVAILABLE, "Tesseract不可用，跳过测试")
class TestTesseractOCREngine(unittest.TestCase):
    """Tesseract引擎测试类"""
    
    def setUp(self):
        """测试准备"""
        # 创建测试图像
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # 在图像上添加文本
        cv2.putText(self.test_image, "测试文本", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    @patch('pytesseract.image_to_data')
    def test_recognize(self, mock_image_to_data):
        """测试识别方法"""
        # 设置mock对象行为
        mock_image_to_data.return_value = """level	page_num	block_num	par_num	line_num	word_num	left	top	width	height	conf	text
1	1	0	0	0	0	0	0	100	100	-1	
2	1	1	0	0	0	9	9	82	42	-1	
3	1	1	1	0	0	9	9	82	42	-1	
4	1	1	1	1	0	9	9	82	42	-1	
5	1	1	1	1	1	9	9	82	42	95.0	测试文本"""
        
        # 初始化引擎并进行识别
        engine = TesseractOCREngine(langs="chi_sim+eng", gpu=False)
        results = engine.recognize(self.test_image)
        
        # 验证结果
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "测试文本")
        self.assertAlmostEqual(results[0].confidence, 0.95)
        self.assertEqual(results[0].engine, "tesseract")
        self.assertEqual(results[0].preprocessor, "original")
        
        # 验证mock方法被调用
        mock_image_to_data.assert_called_once()
    
    def test_is_available(self):
        """测试可用性检查"""
        if TESSERACT_AVAILABLE:
            engine = TesseractOCREngine(langs="chi_sim+eng", gpu=False)
            self.assertTrue(engine.is_available())


if __name__ == "__main__":
    unittest.main() 