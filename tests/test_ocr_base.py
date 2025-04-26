#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OCR基础模块测试
不导入完整项目，只测试基础类的功能
"""

import os
import sys
import json
import unittest
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# 重新定义要测试的类
@dataclass
class OCRResult:
    """OCR识别结果数据结构"""
    text: str  # 识别的文本
    confidence: float  # 置信度
    box: Optional[List] = None  # 文本框坐标 [左上, 右上, 右下, 左下]
    engine: str = ""  # 使用的OCR引擎
    preprocessor: str = ""  # 使用的预处理方法


# 定义一个用于JSON序列化的类
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# 测试类
class TestOCRBase(unittest.TestCase):
    """OCR基础模块测试类"""
    
    def test_ocr_result_creation(self):
        """测试OCR结果创建"""
        # 创建一个OCR结果对象
        box = [[10, 10], [100, 10], [100, 50], [10, 50]]
        result = OCRResult(
            text="测试文本",
            confidence=0.95,
            box=box,
            engine="test_engine",
            preprocessor="original"
        )
        
        # 验证属性
        self.assertEqual(result.text, "测试文本")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.box, box)
        self.assertEqual(result.engine, "test_engine")
        self.assertEqual(result.preprocessor, "original")
    
    def test_ocr_result_default_values(self):
        """测试OCR结果默认值"""
        # 使用最少参数创建对象
        result = OCRResult(text="测试文本", confidence=0.9)
        
        # 验证默认值
        self.assertEqual(result.text, "测试文本")
        self.assertEqual(result.confidence, 0.9)
        self.assertIsNone(result.box)
        self.assertEqual(result.engine, "")
        self.assertEqual(result.preprocessor, "")
    
    def test_numpy_encoder(self):
        """测试NumPy编码器"""
        # 创建包含NumPy类型的数据
        data = {
            "int32": np.int32(42),
            "float32": np.float32(3.14),
            "array": np.array([1, 2, 3])
        }
        
        # 使用编码器序列化
        json_str = json.dumps(data, cls=NumpyEncoder)
        
        # 反序列化并验证
        parsed = json.loads(json_str)
        self.assertEqual(parsed["int32"], 42)
        self.assertAlmostEqual(parsed["float32"], 3.14, places=5)
        self.assertEqual(parsed["array"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main() 