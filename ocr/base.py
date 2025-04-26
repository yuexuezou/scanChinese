"""
OCR基础模块
定义基础类和接口
"""

import os
import cv2
import json
import time
import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Union

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("scanChinese")

# 针对 numpy.int32 的 JSON 序列化问题
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


@dataclass
class OCRResult:
    """OCR识别结果数据结构"""
    text: str  # 识别的文本
    confidence: float  # 置信度
    box: Optional[List] = None  # 文本框坐标 [左上, 右上, 右下, 左下]
    engine: str = ""  # 使用的OCR引擎
    preprocessor: str = ""  # 使用的预处理方法


class OCREngine(ABC):
    """OCR引擎基类，定义统一接口"""
    
    @abstractmethod
    def recognize(self, image: np.ndarray, preprocessor_name: str = "original") -> List[OCRResult]:
        """识别图像中的文本，返回OCRResult列表"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查引擎是否可用"""
        pass
    
    def get_name(self) -> str:
        """获取引擎名称"""
        return self.__class__.__name__
    
    def process_image(self, image_path: str, output_dir: str = None, 
                     write_annotated: bool = True, write_text: bool = True, 
                     write_json: bool = True) -> Dict[str, Any]:
        """处理单个图像，返回识别结果"""
        start_time = time.time()
        
        # 准备输出目录
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = os.path.dirname(image_path) or "."
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return {"error": f"无法读取图像: {image_path}"}
        
        # 获取图像文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 运行OCR识别
        try:
            results = self.recognize(image)
        except Exception as e:
            logger.error(f"识别图像时出错: {str(e)}")
            return {"error": f"识别图像时出错: {str(e)}"}
        
        # 写出结果
        output_files = {}
        
        # 生成带标注的图像
        if write_annotated:
            annotated_img = self._draw_results(image.copy(), results)
            annotated_path = os.path.join(output_dir, f"{base_name}_{self.get_name()}_annotated.jpg")
            cv2.imwrite(annotated_path, annotated_img)
            output_files["annotated_image"] = annotated_path
        
        # 生成纯文本结果
        if write_text:
            text_content = "\n".join([r.text for r in results])
            text_path = os.path.join(output_dir, f"{base_name}_{self.get_name()}_result.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            output_files["text_file"] = text_path
        
        # 生成JSON结果
        if write_json:
            json_result = {
                "file": image_path,
                "engine": self.get_name(),
                "processing_time": time.time() - start_time,
                "results": [
                    {
                        "text": r.text,
                        "confidence": float(r.confidence),
                        "box": r.box,
                        "engine": r.engine,
                        "preprocessor": r.preprocessor
                    } for r in results
                ]
            }
            json_path = os.path.join(output_dir, f"{base_name}_{self.get_name()}_result.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            output_files["json_file"] = json_path
        
        return {
            "results": results,
            "processing_time": time.time() - start_time,
            "output_files": output_files
        }
    
    def _draw_results(self, image: np.ndarray, results: List[OCRResult]) -> np.ndarray:
        """在图像上绘制OCR结果"""
        for result in results:
            if result.box:
                # 绘制文本框 - 确保是 numpy 数组
                try:
                    pts = np.array(result.box, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], True, (0, 255, 0), 2)
                    
                    # 绘制文本
                    x, y = result.box[0]
                    text = f"{result.text} ({result.confidence:.2f})"
                    cv2.putText(image, text, (int(x), max(int(y) - 10, 10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                except Exception as e:
                    logger.error(f"绘制结果时出错: {str(e)}")
        
        return image


def main():
    """注册为可执行模块时的入口点"""
    logger.error("这是一个基础模块，不应直接执行")
    print("这是OCR基础模块，不应直接执行。请使用具体的OCR引擎模块或多引擎模块。")


if __name__ == "__main__":
    main() 