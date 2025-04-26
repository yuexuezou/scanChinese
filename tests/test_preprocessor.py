#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像预处理器测试
"""

import os
import sys
import unittest
import numpy as np
import cv2

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.preprocessor import ImagePreprocessor


class TestImagePreprocessor(unittest.TestCase):
    """图像预处理器测试类"""
    
    def setUp(self):
        """测试准备"""
        # 创建一个简单的测试图像
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # 创建一个有随机内容的测试图像，以便更好地测试处理效果
        self.random_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.preprocessor = ImagePreprocessor()
    
    def test_to_grayscale(self):
        """测试灰度转换"""
        gray = self.preprocessor.to_grayscale(self.test_image)
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape, (100, 100))
    
    def test_enhance_contrast(self):
        """测试对比度增强"""
        enhanced = self.preprocessor.enhance_contrast(self.test_image)
        self.assertEqual(enhanced.shape, self.test_image.shape)
        # 检查对比度是否增强（像素值变化）
        self.assertFalse(np.array_equal(enhanced, self.test_image))
    
    def test_adaptive_threshold(self):
        """测试自适应阈值处理"""
        threshold = self.preprocessor.adaptive_threshold(self.test_image)
        self.assertEqual(len(threshold.shape), 2)
        # 检查是否二值化（只有0和255）
        unique_values = np.unique(threshold)
        self.assertTrue(all(v in [0, 255] for v in unique_values))
    
    def test_sharpen(self):
        """测试锐化处理"""
        # 使用随机图像进行测试，因为均匀图像锐化后可能不会变化
        sharpened = self.preprocessor.sharpen(self.random_image)
        self.assertEqual(sharpened.shape, self.random_image.shape)
        # 检查是否锐化（与原图不同）
        self.assertFalse(np.array_equal(sharpened, self.random_image))
    
    def test_gaussian_blur(self):
        """测试高斯模糊"""
        # 使用随机图像进行测试，因为均匀图像模糊后可能不会变化
        blurred = self.preprocessor.gaussian_blur(self.random_image)
        self.assertEqual(blurred.shape, self.random_image.shape)
        # 检查是否模糊（与原图不同）
        self.assertFalse(np.array_equal(blurred, self.random_image))
    
    def test_apply_all_preprocessors(self):
        """测试应用所有预处理器"""
        results = self.preprocessor.apply_all_preprocessors(self.test_image)
        # 检查是否包含所有预处理方法的结果
        self.assertIn("original", results)
        self.assertIn("grayscale", results)
        self.assertIn("adaptive_histogram", results)
        self.assertIn("contrast_enhanced", results)
        self.assertIn("denoised", results)
        self.assertIn("adaptive_threshold", results)
        self.assertIn("sharpened", results)
        self.assertIn("gaussian_blur", results)
        self.assertIn("deskewed", results)
        self.assertIn("binarized", results)
    
    def test_get_preprocessor_by_name(self):
        """测试根据名称获取预处理方法"""
        # 测试有效的预处理器名称
        preprocessor = self.preprocessor.get_preprocessor_by_name("grayscale")
        self.assertIsNotNone(preprocessor)
        
        # 测试处理结果
        result = preprocessor(self.test_image)
        self.assertEqual(len(result.shape), 2)
        
        # 测试无效的预处理器名称，应返回默认处理器（原图复制）
        invalid_preprocessor = self.preprocessor.get_preprocessor_by_name("invalid_name")
        self.assertIsNotNone(invalid_preprocessor)
        
        # 测试默认处理器结果
        result = invalid_preprocessor(self.test_image)
        self.assertTrue(np.array_equal(result, self.test_image))


if __name__ == "__main__":
    unittest.main() 