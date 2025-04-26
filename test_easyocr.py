#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用EasyOCR测试中文图片识别
"""

import os
import cv2
import easyocr
import time
import argparse
import numpy as np

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description="使用EasyOCR测试中文图片")
    parser.add_argument("--input", "-i", required=True, help="输入图像文件或目录")
    parser.add_argument("--output", "-o", default="ocr_results", help="输出目录")
    args = parser.parse_args()
    
    # 初始化输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 初始化OCR引擎
    print("初始化EasyOCR引擎...")
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
    
    # 处理输入
    if os.path.isfile(args.input):
        # 处理单个文件
        process_image(reader, args.input, args.output)
    else:
        # 处理目录
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        for root, _, files in os.walk(args.input):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_exts):
                    image_path = os.path.join(root, file)
                    print(f"处理: {image_path}")
                    process_image(reader, image_path, args.output)

def process_image(reader, image_path, output_dir):
    """处理单个图像"""
    start_time = time.time()
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 获取图像文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # OCR识别
    results = reader.readtext(image)
    
    # 打印识别结果
    if results:
        print(f"图像 {base_name} 识别结果:")
        for i, (box, text, conf) in enumerate(results, 1):
            print(f"  {i}. '{text}' (置信度: {conf:.2f})")
    else:
        print(f"图像 {base_name} 未识别到文本")
    
    # 生成标注图像
    annotated_img = image.copy()
    for box, text, conf in results:
        # 绘制文本框 - 修复了绘制代码
        pts = np.array(box, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(annotated_img, [pts], True, (0, 255, 0), 2)
        
        # 绘制文本
        x, y = box[0]
        cv2.putText(
            annotated_img, 
            f"{text} ({conf:.2f})", 
            (int(x), max(int(y) - 10, 10)), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 255), 
            1
        )
    
    # 保存标注图像
    output_img_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
    cv2.imwrite(output_img_path, annotated_img)
    
    # 生成文本文件
    text_path = os.path.join(output_dir, f"{base_name}_result.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        for _, text, _ in results:
            f.write(text + "\n")
    
    process_time = time.time() - start_time
    print(f"处理完成，耗时: {process_time:.2f} 秒")
    print(f"结果保存至: {output_dir}")

if __name__ == "__main__":
    main() 