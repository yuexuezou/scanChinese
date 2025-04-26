# 高级中文OCR处理系统

这是一个多引擎中文OCR处理系统，整合了多种OCR引擎来提高中文文本识别的准确性。

## 主要特点

- **多引擎架构**：集成了Tesseract、PaddleOCR、EasyOCR、RapidOCR、MangaOCR、MMOCR、TrOCR和Donut多个OCR引擎
- **图像预处理**：提供多种图像处理方法，包括灰度处理、直方图均衡化、对比度增强等
- **结果集成与优化**：整合多个引擎的识别结果，使用置信度排序和过滤
- **中文优化**：使用结巴分词提高中文识别准确率
- **并行处理**：使用线程池实现多图像并行处理
- **多种输出**：支持生成带标注的图像、纯文本结果和JSON格式的详细信息
- **模块化设计**：每个OCR引擎都被封装成独立模块，可单独使用，也可组合使用
- **引擎特色**：
  - Tesseract: 经典OCR引擎，支持多语言
  - PaddleOCR: 百度开发的中文优化OCR引擎
  - EasyOCR: 简单易用的多语言OCR
  - RapidOCR: 轻量高性能OCR引擎
  - MangaOCR: 专为漫画和日文文本优化的OCR
  - MMOCR: 商汤科技的多模态OCR工具箱
  - TrOCR: 微软基于Transformer的OCR模型
  - Donut: 专为文档和表格理解设计的OCR

## 项目规则

以下是本项目必须遵循的规则：

1. **模块化设计**：每个OCR引擎必须封装为独立的脚本模块，可单独调用
2. **统一接口**：所有引擎模块必须实现统一的接口，确保互操作性
3. **完整测试**：所有引擎模块必须经过完整测试验证，包括单元测试和集成测试
4. **错误处理**：必须妥善处理所有可能的错误情况，包括引擎不可用、图像加载失败等
5. **文档完备**：每个模块都必须有详细的文档说明，包括使用方法、参数说明等
6. **版本控制**：明确标注每个引擎支持的版本要求
7. **依赖管理**：清晰列出各模块的依赖，支持按需安装
8. **可扩展性**：系统设计必须允许轻松添加新的OCR引擎
9. **性能优化**：提供性能基准测试，确保系统在大量图像处理时依然高效

## 项目架构

```
scanChinese/
├── README.md                     # 项目说明文档
├── requirements.txt              # 项目依赖
├── scan_chinese.py               # 主入口脚本
├── ocr/
│   ├── __init__.py               # OCR包初始化
│   ├── base.py                   # 基础接口和工具类
│   ├── preprocessor.py           # 图像预处理模块
│   ├── engines/                  # 引擎目录
│   │   ├── __init__.py           # 引擎包初始化
│   │   ├── tesseract_engine.py   # Tesseract引擎模块
│   │   ├── paddle_engine.py      # PaddleOCR引擎模块
│   │   ├── easyocr_engine.py     # EasyOCR引擎模块
│   │   ├── rapidocr_engine.py    # RapidOCR引擎模块
│   │   ├── manga_engine.py       # MangaOCR引擎模块
│   │   ├── mmocr_engine.py       # MMOCR引擎模块
│   │   ├── trocr_engine.py       # TrOCR引擎模块
│   │   └── donut_engine.py       # Donut引擎模块
│   └── multi_engine.py           # 多引擎整合模块
├── tests/                        # 测试目录
│   ├── __init__.py
│   ├── test_preprocessor.py
│   ├── test_tesseract.py
│   ├── test_paddle.py
│   ├── test_easyocr.py
│   ├── test_rapidocr.py
│   └── test_multi_engine.py
├── examples/                     # 示例目录
│   ├── single_engine_demo.py     # 单引擎使用示例
│   └── multi_engine_demo.py      # 多引擎使用示例
└── tools/                        # 工具脚本
    ├── benchmark.py              # 性能基准测试
    └── install_dependencies.py   # 依赖安装助手
```

## 当前实现状态

- [x] 核心框架设计
- [x] 基础接口定义
- [x] 图像预处理模块
- [x] Tesseract引擎封装
- [x] EasyOCR引擎封装
- [x] PaddleOCR引擎封装
- [x] RapidOCR引擎封装
- [x] MangaOCR引擎封装
- [x] MMOCR引擎封装
- [x] TrOCR引擎封装
- [x] Donut引擎封装
- [x] 多引擎整合模块
- [x] 命令行接口
- [x] 基础测试用例
- [ ] 完整测试覆盖
- [ ] 性能优化
- [ ] 文档完善

## 安装

### 1. 安装依赖

```bash
pip install opencv-python numpy jieba shapely
```

根据需要安装OCR引擎：

```bash
# 安装所有引擎
pip install pytesseract paddleocr easyocr rapidocr_onnxruntime transformers torch manga-ocr mmocr donut-python

# 或仅安装需要的引擎
pip install pytesseract  # 仅Tesseract
pip install paddleocr    # 仅PaddleOCR
pip install easyocr      # 仅EasyOCR
pip install rapidocr_onnxruntime  # 仅RapidOCR
pip install manga-ocr    # 仅MangaOCR
pip install mmocr        # 仅MMOCR
pip install transformers torch  # TrOCR和Donut共同依赖
```

注意：系统会检测可用的OCR引擎，如果某个引擎未安装，该功能将被禁用，但不影响其他引擎的使用。

### 2. Tesseract安装

除了Python依赖外，还需要安装Tesseract OCR引擎：

- **Windows**: 
  下载并安装 [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki)，并添加到系统环境变量

- **Linux**: 
  ```bash
  sudo apt-get install tesseract-ocr
  sudo apt-get install tesseract-ocr-chi-sim tesseract-ocr-chi-tra
  ```

- **macOS**: 
  ```bash
  brew install tesseract
  brew install tesseract-lang
  ```

## 使用方法

### 基本用法

处理单个图像文件：

```bash
python scan_chinese.py --input path/to/image.jpg --output path/to/output
```

处理整个目录中的图像：

```bash
python scan_chinese.py --input path/to/images/directory --output path/to/output
```

### 选择特定引擎

```bash
python scan_chinese.py --input image.jpg --engines tesseract,easyocr
```

### 参数说明

- `--input`, `-i`: 输入图像文件或目录（必需）
- `--output`, `-o`: 输出目录（可选，默认与输入文件同目录）
- `--engines`, `-e`: 要使用的OCR引擎，逗号分隔（可选，默认全部可用引擎）
- `--gpu`: 使用GPU加速（如果可用）
- `--confidence`, `-c`: 置信度阈值，默认0.5
- `--workers`, `-w`: 并行处理的线程数，默认4
- `--no-annotated`: 不输出带标注的图像
- `--no-text`: 不输出文本文件
- `--no-json`: 不输出JSON结果文件

## 示例

```bash
# 使用所有可用引擎处理单个文件
python scan_chinese.py -i test.jpg -o results

# 仅使用EasyOCR和PaddleOCR，并设置较高置信度
python scan_chinese.py -i test.jpg -o results -e easyocr,paddleocr -c 0.65

# 处理整个目录，使用8个线程并行处理
python scan_chinese.py -i images/ -o results -w 8

# 使用单个引擎处理
python ocr/engines/easyocr_engine.py -i test.jpg -o results
```

## 如何获取最佳结果

- 对于较复杂的图像，建议使用预处理功能
- 调整置信度阈值可以平衡召回率和准确率
- 对于特定类型的文档，可能需要调整特定的OCR引擎参数

## 示例代码

### 使用单个引擎

```python
from ocr.engines import get_engine_by_name

# 初始化EasyOCR引擎
engine = get_engine_by_name("easyocr", langs=["ch_sim", "en"], gpu=False)

# 处理图像
result = engine.process_image("path/to/image.jpg", "path/to/output")

# 获取识别结果
for item in result["results"]:
    print(f"文本: {item.text}, 置信度: {item.confidence}")
```

### 使用多引擎

```python
from ocr.multi_engine import MultiEngineOCR

# 初始化多引擎OCR
ocr = MultiEngineOCR(engines=["tesseract", "easyocr"], use_gpu=False)

# 处理图像
result = ocr.process_image("path/to/image.jpg", "path/to/output")

# 获取识别结果
for item in result["results"]:
    print(f"文本: {item.text}, 置信度: {item.confidence}, 引擎: {item.engine}")
```

### 使用新增引擎

```python
# 使用MangaOCR引擎识别漫画文本
from ocr.engines import get_engine_by_name

manga_engine = get_engine_by_name("manga", force_cpu=False)
result = manga_engine.process_image("manga_panel.jpg", "output")
print(f"识别到的文本: {result['results'][0].text}")

# 使用TrOCR引擎识别手写文本
trocr_engine = get_engine_by_name("trocr", model_name="microsoft/trocr-base-handwritten")
result = trocr_engine.process_image("handwritten_note.jpg", "output")
print(f"识别到的手写文本: {result['results'][0].text}")

# 使用MMOCR引擎
mmocr_engine = get_engine_by_name("mmocr", det="DBNet", rec="CRNN")
result = mmocr_engine.process_image("chinese_text.jpg", "output")
for res in result["results"]:
    print(f"文本: {res.text}, 置信度: {res.confidence:.2f}")

# 使用Donut引擎识别文档
donut_engine = get_engine_by_name("donut", model_name="naver-clova-ix/donut-base-finetuned-cord-v2")
result = donut_engine.process_image("receipt.jpg", "output")
print(f"识别到的收据文本: {result['results'][0].text}")
```

### 使用多引擎组合

```python
from ocr.multi_engine import MultiEngineOCR

# 组合使用多个引擎
ocr = MultiEngineOCR(engines=["tesseract", "easyocr", "manga", "trocr"], use_gpu=True)
result = ocr.process_image("path/to/image.jpg", "path/to/output")

# 显示各引擎识别结果
for engine_name, engine_results in result.get("results_by_engine", {}).items():
    print(f"\n{engine_name} 识别结果:")
    for res in engine_results:
        print(f"文本: {res.text}, 置信度: {res.confidence:.2f}")
```

## 测试

为了确保系统的稳定性和可靠性，我们提供了多种测试文件。测试覆盖了OCR基础模块、预处理器、以及各OCR引擎的功能。

### 运行所有测试

```bash
python -m unittest discover tests
```

### 运行特定测试

```bash
# 测试预处理器
python tests/test_preprocessor.py

# 测试OCR基础模块
python tests/test_ocr_base.py

# 测试特定引擎
python tests/test_tesseract.py
python tests/test_easyocr.py
python tests/test_paddle.py
python tests/test_rapidocr.py

# 测试多引擎整合
python tests/test_multi_engine.py
```

### 运行独立测试（不依赖完整项目）

对于缺少某些依赖的环境，我们提供了独立测试文件，可以在不安装所有OCR引擎的情况下测试部分功能：

```bash
# 测试预处理器（独立版本）
python tests/test_isolated_preprocessor.py

# 测试OCR基础模块（独立版本）
python tests/test_isolated_ocr_base.py

# 测试主模块（独立版本）
python tests/test_isolated_scan_chinese.py
```

### 测试自定义图像

您可以使用主程序对自己的图像进行测试：

```bash
# 使用所有可用的引擎
python scan_chinese.py -i your_image.jpg -o test_results

# 指定使用特定引擎
python scan_chinese.py -i your_image.jpg -o test_results -e tesseract,easyocr
```

## 许可证

MIT 