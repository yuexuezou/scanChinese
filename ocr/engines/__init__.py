"""
OCR引擎包
封装多个OCR引擎的实现
"""

# 尝试导入各引擎，记录可用性
try:
    from .tesseract_engine import TesseractEngine
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from .paddle_engine import PaddleEngine
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

try:
    from .easyocr_engine import EasyOCREngine
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from .rapidocr_engine import RapidOCREngine
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False

try:
    from .manga_engine import MangaEngine
    MANGA_AVAILABLE = True
except ImportError:
    MANGA_AVAILABLE = False

try:
    from .mmocr_engine import MMOCREngine
    MMOCR_AVAILABLE = True
except ImportError:
    MMOCR_AVAILABLE = False

try:
    from .trocr_engine import TrOCREngine
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False

try:
    from .donut_engine import DonutEngine
    DONUT_AVAILABLE = True
except ImportError:
    DONUT_AVAILABLE = False

try:
    from .google_vision_engine import GoogleVisionEngine
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False


# 引擎注册表
ENGINE_REGISTRY = {
    "tesseract": {"cls": TesseractEngine, "available": TESSERACT_AVAILABLE},
    "paddle": {"cls": PaddleEngine, "available": PADDLE_AVAILABLE},
    "easyocr": {"cls": EasyOCREngine, "available": EASYOCR_AVAILABLE},
    "rapidocr": {"cls": RapidOCREngine, "available": RAPIDOCR_AVAILABLE},
    "manga": {"cls": MangaEngine, "available": MANGA_AVAILABLE},
    "mmocr": {"cls": MMOCREngine, "available": MMOCR_AVAILABLE},
    "trocr": {"cls": TrOCREngine, "available": TROCR_AVAILABLE},
    "donut": {"cls": DonutEngine, "available": DONUT_AVAILABLE},
    "google_vision": {"cls": GoogleVisionEngine, "available": GOOGLE_VISION_AVAILABLE}
}


def get_available_engines():
    """获取当前可用的OCR引擎列表"""
    available_engines = {}
    for name, info in ENGINE_REGISTRY.items():
        available_engines[name] = info["available"]
    return available_engines


def get_engine(engine_name, **kwargs):
    """根据名称获取OCR引擎实例"""
    if engine_name not in ENGINE_REGISTRY:
        raise ValueError(f"不支持的引擎: {engine_name}，可用引擎: {list(ENGINE_REGISTRY.keys())}")
    
    engine_info = ENGINE_REGISTRY[engine_name]
    if not engine_info["available"]:
        raise ImportError(f"引擎 {engine_name} 不可用，请检查依赖是否已安装")
    
    try:
        return engine_info["cls"](**kwargs)
    except Exception as e:
        raise RuntimeError(f"初始化引擎 {engine_name} 失败: {str(e)}")


# 兼容旧版API
get_engine_by_name = get_engine 