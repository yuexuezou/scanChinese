#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OCR基础模块独立测试
不导入完整项目，只测试基础类的功能
"""

import os
import sys
import json
import unittest
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Union


# 定义要测试的类
@dataclass
class BoundingBox:
    """文本框坐标数据结构"""
    points: List[List[int]]  # 多边形点列表
    
    def __post_init__(self):
        # 确保点列表为正确的格式
        if not isinstance(self.points, list):
            raise ValueError("points 必须是一个列表")
        
        # 确保至少有3个点
        if len(self.points) < 3:
            raise ValueError("边界框必须至少有3个点")
        
        # 确保每个点是一个长度为2的列表
        for point in self.points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("每个点必须是一个包含两个整数的列表")
    
    def as_tuple(self) -> Tuple[int, int, int, int]:
        """返回边界框的左上右下坐标元组"""
        x_points = [p[0] for p in self.points]
        y_points = [p[1] for p in self.points]
        return (min(x_points), min(y_points), max(x_points), max(y_points))


@dataclass
class OCRResult:
    """OCR识别结果数据结构"""
    text: str  # 识别的文本
    confidence: float  # 置信度
    bbox: Optional[BoundingBox] = None  # 文本框坐标
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
    
    def test_bounding_box_creation(self):
        """测试边界框创建"""
        # 创建一个边界框对象
        points = [[10, 10], [100, 10], [100, 50], [10, 50]]
        bbox = BoundingBox(points=points)
        
        # 验证属性
        self.assertEqual(bbox.points, points)
        
        # 验证转换为元组
        self.assertEqual(bbox.as_tuple(), (10, 10, 100, 50))
    
    def test_bounding_box_validation(self):
        """测试边界框验证"""
        # 测试点数太少
        with self.assertRaises(ValueError):
            BoundingBox(points=[[10, 10], [100, 10]])
        
        # 测试点格式错误
        with self.assertRaises(ValueError):
            BoundingBox(points=[[10, 10], [100], [100, 50]])
        
        # 测试非列表输入
        with self.assertRaises(ValueError):
            BoundingBox(points="不是列表")
    
    def test_ocr_result_creation(self):
        """测试OCR结果创建"""
        # 创建一个边界框
        bbox = BoundingBox(points=[[10, 10], [100, 10], [100, 50], [10, 50]])
        
        # 创建一个OCR结果对象
        result = OCRResult(
            text="测试文本",
            confidence=0.95,
            bbox=bbox,
            engine="test_engine",
            preprocessor="original"
        )
        
        # 验证属性
        self.assertEqual(result.text, "测试文本")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.bbox, bbox)
        self.assertEqual(result.engine, "test_engine")
        self.assertEqual(result.preprocessor, "original")
    
    def test_ocr_result_default_values(self):
        """测试OCR结果默认值"""
        # 使用最少参数创建对象
        result = OCRResult(text="测试文本", confidence=0.9)
        
        # 验证默认值
        self.assertEqual(result.text, "测试文本")
        self.assertEqual(result.confidence, 0.9)
        self.assertIsNone(result.bbox)
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