#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像预处理器独立测试
不导入完整项目，只测试预处理器功能
"""

import os
import sys
import unittest
import numpy as np
import cv2

# 定义要测试的类
class ImagePreprocessor:
    """图像预处理器"""
    
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """转换为灰度图像"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def adaptive_histogram_equalization(image: np.ndarray) -> np.ndarray:
        """自适应直方图均衡化"""
        gray = ImagePreprocessor.to_grayscale(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, alpha=1.5, beta=0) -> np.ndarray:
        """增强对比度"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """降噪处理"""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    @staticmethod
    def adaptive_threshold(image: np.ndarray) -> np.ndarray:
        """自适应阈值二值化"""
        gray = ImagePreprocessor.to_grayscale(image)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    
    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        """锐化处理"""
        kernel = np.array([[-1, -1, -1], 
                          [-1, 9, -1], 
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def gaussian_blur(image: np.ndarray, kernel_size=(5, 5)) -> np.ndarray:
        """高斯模糊"""
        return cv2.GaussianBlur(image, kernel_size, 0)
    
    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """倾斜校正"""
        # 转换为灰度图像
        gray = ImagePreprocessor.to_grayscale(image)
        
        # 二值化
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # 选择最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 计算最小外接矩形
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # 调整角度范围
        if angle < -45:
            angle = 90 + angle
        
        # 只有当角度不是接近水平或垂直时才进行校正
        if abs(angle) < 0.5:
            return image
        
        # 获取旋转矩阵
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 应用旋转
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    @staticmethod
    def binarize(image: np.ndarray) -> np.ndarray:
        """全局二值化"""
        gray = ImagePreprocessor.to_grayscale(image)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def apply_all_preprocessors(self, image: np.ndarray) -> dict:
        """应用所有预处理器并返回结果字典"""
        results = {
            "original": image.copy(),
            "grayscale": self.to_grayscale(image),
            "adaptive_histogram": self.adaptive_histogram_equalization(image),
            "contrast_enhanced": self.enhance_contrast(image),
            "denoised": self.denoise(image),
            "adaptive_threshold": self.adaptive_threshold(image),
            "sharpened": self.sharpen(image),
            "gaussian_blur": self.gaussian_blur(image),
            "deskewed": self.deskew(image),
            "binarized": self.binarize(image)
        }
        return results
    
    def get_preprocessor_by_name(self, name: str):
        """根据名称获取预处理方法"""
        processors = {
            "original": lambda img: img.copy(),
            "grayscale": self.to_grayscale,
            "adaptive_histogram": self.adaptive_histogram_equalization,
            "contrast_enhanced": self.enhance_contrast,
            "denoised": self.denoise,
            "adaptive_threshold": self.adaptive_threshold,
            "sharpened": self.sharpen,
            "gaussian_blur": self.gaussian_blur,
            "deskewed": self.deskew,
            "binarized": self.binarize
        }
        
        return processors.get(name, lambda img: img.copy())


# 测试类
class TestImagePreprocessor(unittest.TestCase):
    """图像预处理器测试类"""
    
    def setUp(self):
        """测试准备"""
        # 创建一个简单的测试图像
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # 创建一个随机图像用于更好的测试效果
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
        # 由于使用的是非零的alpha值，所以结果应该与原图不同
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
        # 使用随机图像，这样锐化后有明显变化
        sharpened = self.preprocessor.sharpen(self.random_image)
        self.assertEqual(sharpened.shape, self.random_image.shape)
        # 检查是否锐化（与原图不同）
        self.assertFalse(np.array_equal(sharpened, self.random_image))
    
    def test_gaussian_blur(self):
        """测试高斯模糊"""
        # 使用随机图像，这样模糊后有明显变化
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