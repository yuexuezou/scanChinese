"""
图像预处理模块
提供多种图像预处理方法，用于提高OCR识别率
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional


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
                           [-1,  9, -1], 
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
    
    def apply_all_preprocessors(self, image: np.ndarray) -> Dict[str, np.ndarray]:
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


def main():
    """测试预处理器"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="测试图像预处理器")
    parser.add_argument("--input", "-i", required=True, help="输入图像路径")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--method", "-m", help="要测试的单个预处理方法")
    
    args = parser.parse_args()
    
    # 读取图像
    image = cv2.imread(args.input)
    if image is None:
        print(f"无法读取图像: {args.input}")
        return
    
    preprocessor = ImagePreprocessor()
    
    # 准备输出目录
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    else:
        args.output = os.path.dirname(args.input) or "."
    
    # 获取图像文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    
    # 如果指定了单个方法，则仅测试该方法
    if args.method:
        processor_fn = preprocessor.get_preprocessor_by_name(args.method)
        if processor_fn:
            processed = processor_fn(image)
            output_path = os.path.join(args.output, f"{base_name}_{args.method}.jpg")
            cv2.imwrite(output_path, processed)
            print(f"已保存 {args.method} 处理结果至 {output_path}")
        else:
            print(f"未知的预处理方法: {args.method}")
    else:
        # 应用所有预处理方法
        results = preprocessor.apply_all_preprocessors(image)
        
        # 保存结果
        for name, processed in results.items():
            output_path = os.path.join(args.output, f"{base_name}_{name}.jpg")
            cv2.imwrite(output_path, processed)
        
        print(f"已保存所有预处理结果至 {args.output}")


if __name__ == "__main__":
    main() 