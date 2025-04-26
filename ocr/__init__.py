"""
scanChinese OCR 包
集成多个OCR引擎的中文文字识别系统
"""

__version__ = "1.0.0"

from .base import OCRResult, OCREngine
from .preprocessor import ImagePreprocessor
from .multi_engine import MultiEngineOCR

# 尝试导入各个引擎
try:
    from .engines.tesseract_engine import TesseractEngine
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from .engines.paddle_engine import PaddleEngine
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

try:
    from .engines.easyocr_engine import EasyOCREngine
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from .engines.rapidocr_engine import RapidOCREngine
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False


def get_available_engines():
    """获取当前可用的OCR引擎列表"""
    engines = []
    
    if TESSERACT_AVAILABLE:
        engines.append("tesseract")
    
    if PADDLE_AVAILABLE:
        engines.append("paddleocr")
    
    if EASYOCR_AVAILABLE:
        engines.append("easyocr")
    
    if RAPIDOCR_AVAILABLE:
        engines.append("rapidocr")
    
    return engines 