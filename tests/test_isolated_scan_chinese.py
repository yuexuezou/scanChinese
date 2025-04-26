#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ScanChinese主模块独立测试
不依赖完整OCR引擎，测试主模块的命令行参数解析等功能
"""

import os
import sys
import unittest
import argparse
from unittest.mock import patch, MagicMock

# 定义类似于scan_chinese.py中的参数解析函数
def parse_arguments(args=None):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="ScanChinese: 多引擎中文OCR处理系统"
    )
    parser.add_argument("--input", "-i", required=True, help="输入图像文件或目录")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--engines", "-e", help="要使用的引擎，用逗号分隔，如 'tesseract,easyocr'")
    parser.add_argument("--gpu", action="store_true", help="使用GPU加速")
    parser.add_argument("--confidence", "-c", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--workers", "-w", type=int, default=4, help="并行处理的工作线程数")
    parser.add_argument("--no-annotated", action="store_true", help="不输出标注图像")
    parser.add_argument("--no-text", action="store_true", help="不输出文本文件")
    parser.add_argument("--no-json", action="store_true", help="不输出JSON文件")
    parser.add_argument("--version", "-v", action="store_true", help="显示版本信息")
    
    return parser.parse_args(args)


# 引擎解析函数
def parse_engines(engines_str):
    """解析引擎列表"""
    if not engines_str:
        return []
    return engines_str.split(",")


# 输入路径验证函数
def validate_input(input_path):
    """验证输入路径"""
    if not input_path:
        return False, "输入路径不能为空"
    
    if os.path.isfile(input_path):
        return True, "文件模式"
    elif os.path.isdir(input_path):
        return True, "目录模式"
    else:
        return False, f"输入路径不存在: {input_path}"


# 测试类
class TestScanChinese(unittest.TestCase):
    """ScanChinese主模块测试类"""
    
    def test_argument_parsing(self):
        """测试命令行参数解析"""
        # 测试必需参数
        args = parse_arguments(["--input", "test.jpg"])
        self.assertEqual(args.input, "test.jpg")
        self.assertIsNone(args.output)
        self.assertIsNone(args.engines)
        self.assertFalse(args.gpu)
        self.assertEqual(args.confidence, 0.5)
        self.assertEqual(args.workers, 4)
        self.assertFalse(args.no_annotated)
        self.assertFalse(args.no_text)
        self.assertFalse(args.no_json)
        self.assertFalse(args.version)
        
        # 测试完整参数
        args = parse_arguments([
            "--input", "test.jpg",
            "--output", "output_dir",
            "--engines", "tesseract,easyocr",
            "--gpu",
            "--confidence", "0.7",
            "--workers", "8",
            "--no-annotated",
            "--no-text",
            "--no-json",
            "--version"
        ])
        
        self.assertEqual(args.input, "test.jpg")
        self.assertEqual(args.output, "output_dir")
        self.assertEqual(args.engines, "tesseract,easyocr")
        self.assertTrue(args.gpu)
        self.assertEqual(args.confidence, 0.7)
        self.assertEqual(args.workers, 8)
        self.assertTrue(args.no_annotated)
        self.assertTrue(args.no_text)
        self.assertTrue(args.no_json)
        self.assertTrue(args.version)
    
    def test_engine_parsing(self):
        """测试引擎列表解析"""
        self.assertEqual(parse_engines(None), [])
        self.assertEqual(parse_engines(""), [])
        self.assertEqual(parse_engines("tesseract"), ["tesseract"])
        self.assertEqual(
            parse_engines("tesseract,easyocr,paddle"), 
            ["tesseract", "easyocr", "paddle"]
        )
    
    @patch('os.path.isfile')
    def test_input_validation(self, mock_isfile):
        """测试输入路径验证"""
        # 测试空路径
        is_valid, message = validate_input("")
        self.assertFalse(is_valid)
        self.assertEqual(message, "输入路径不能为空")
        
        # 测试文件存在
        mock_isfile.return_value = True
        is_valid, mode = validate_input("test.jpg")
        self.assertTrue(is_valid)
        self.assertEqual(mode, "文件模式")
        
        # 测试目录存在
        mock_isfile.return_value = False
        with patch('os.path.isdir', return_value=True):
            is_valid, mode = validate_input("test_dir")
            self.assertTrue(is_valid)
            self.assertEqual(mode, "目录模式")
        
        # 测试路径不存在
        mock_isfile.return_value = False
        with patch('os.path.isdir', return_value=False):
            is_valid, mode = validate_input("nonexistent")
            self.assertFalse(is_valid)
            self.assertEqual(mode, "输入路径不存在: nonexistent")


if __name__ == "__main__":
    unittest.main() 