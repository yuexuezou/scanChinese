#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
依赖安装助手
用于帮助用户安装各种OCR引擎依赖
"""

import os
import sys
import platform
import subprocess
import argparse
import logging
from typing import List, Dict, Any, Tuple

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DependencyInstaller")


class DependencyInstaller:
    """依赖安装助手类"""
    
    def __init__(self):
        """初始化安装助手"""
        # 获取系统信息
        self.system = platform.system().lower()
        self.is_windows = self.system == "windows"
        self.is_linux = self.system == "linux"
        self.is_mac = self.system == "darwin"
        
        # 检测Python环境
        self.python_version = platform.python_version()
        self.pip_cmd = self._get_pip_command()
        
        # 检测是否有CUDA
        self.has_cuda = self._check_cuda_available()
        
        logger.info(f"系统: {self.system}")
        logger.info(f"Python版本: {self.python_version}")
        logger.info(f"PIP命令: {self.pip_cmd}")
        logger.info(f"CUDA可用: {self.has_cuda}")
    
    def _get_pip_command(self) -> str:
        """获取PIP命令"""
        # 尝试使用pip3，如果失败则使用pip
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "--version"], 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return f"{sys.executable} -m pip"
        except Exception:
            if self.is_windows:
                return "pip"
            else:
                return "pip3"
    
    def _check_cuda_available(self) -> bool:
        """检查CUDA是否可用"""
        # 检查是否有nvidia-smi命令
        try:
            subprocess.check_call(["nvidia-smi"], 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except Exception:
            # 尝试导入torch并检查CUDA
            try:
                import torch
                return torch.cuda.is_available()
            except Exception:
                return False
    
    def _run_command(self, command: str) -> Tuple[bool, str]:
        """运行命令并返回结果"""
        try:
            process = subprocess.Popen(
                command, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            stdout, stderr = process.communicate()
            success = process.returncode == 0
            return success, stdout if success else stderr
        except Exception as e:
            return False, str(e)
    
    def install_core_dependencies(self) -> bool:
        """安装核心依赖"""
        logger.info("安装核心依赖...")
        
        # 核心依赖
        dependencies = [
            "opencv-python>=4.5.0",
            "numpy>=1.19.0",
            "jieba>=0.42.1",
            "shapely>=1.7.0",
            "matplotlib>=3.3.0"
        ]
        
        # 安装依赖
        cmd = f"{self.pip_cmd} install {' '.join(dependencies)}"
        success, output = self._run_command(cmd)
        
        if success:
            logger.info("核心依赖安装成功")
        else:
            logger.error(f"安装核心依赖失败: {output}")
        
        return success
    
    def install_tesseract(self) -> bool:
        """安装Tesseract OCR引擎并自动设置环境变量"""
        logger.info("开始安装Tesseract OCR引擎...")
        
        try:
            # 安装pytesseract
            cmd = f"{self.pip_cmd} install pytesseract -i https://pypi.tuna.tsinghua.edu.cn/simple"
            success, output = self._run_command(cmd)
            if not success:
                logger.error(f"安装pytesseract失败: {output}")
                return False
            
            # 根据操作系统安装Tesseract及设置环境变量
            if self.is_windows:
                # Windows下需要下载安装包
                tesseract_dir = "C:\\Program Files\\Tesseract-OCR"
                tesseract_exe = os.path.join(tesseract_dir, "tesseract.exe")
                
                # 检查是否已安装
                if os.path.exists(tesseract_exe):
                    logger.info("检测到Tesseract已安装")
                else:
                    # 下载安装包
                    logger.info("正在下载Tesseract安装包...")
                    download_dir = os.path.join(os.environ.get("TEMP", "C:\\Windows\\Temp"), "tesseract_installer")
                    os.makedirs(download_dir, exist_ok=True)
                    installer_path = os.path.join(download_dir, "tesseract-installer.exe")
                    
                    # 使用PowerShell下载安装程序
                    download_url = "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.3.1.20230401.exe"
                    download_cmd = f'powershell -Command "Invoke-WebRequest -Uri {download_url} -OutFile {installer_path}"'
                    
                    success, output = self._run_command(download_cmd)
                    if not success or not os.path.exists(installer_path):
                        logger.error(f"下载Tesseract安装包失败: {output}")
                        logger.info("请手动下载安装Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
                        return False
                    
                    # 运行安装程序（静默安装）
                    logger.info("正在安装Tesseract...")
                    install_cmd = f'"{installer_path}" /S /D={tesseract_dir}'
                    success, output = self._run_command(install_cmd)
                    
                    if not success or not os.path.exists(tesseract_exe):
                        logger.error(f"安装Tesseract失败: {output}")
                        logger.info("请手动安装Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
                        return False
                
                # 下载中文语言数据
                tessdata_dir = os.path.join(tesseract_dir, "tessdata")
                os.makedirs(tessdata_dir, exist_ok=True)
                
                # 检查是否已有中文语言文件
                chi_sim_path = os.path.join(tessdata_dir, "chi_sim.traineddata")
                chi_tra_path = os.path.join(tessdata_dir, "chi_tra.traineddata")
                
                if not os.path.exists(chi_sim_path):
                    logger.info("正在下载简体中文语言数据...")
                    chi_sim_url = "https://github.com/tesseract-ocr/tessdata/raw/main/chi_sim.traineddata"
                    download_cmd = f'powershell -Command "Invoke-WebRequest -Uri {chi_sim_url} -OutFile {chi_sim_path}"'
                    success, output = self._run_command(download_cmd)
                    if not success:
                        logger.warning(f"下载简体中文语言数据失败: {output}")
                
                if not os.path.exists(chi_tra_path):
                    logger.info("正在下载繁体中文语言数据...")
                    chi_tra_url = "https://github.com/tesseract-ocr/tessdata/raw/main/chi_tra.traineddata"
                    download_cmd = f'powershell -Command "Invoke-WebRequest -Uri {chi_tra_url} -OutFile {chi_tra_path}"'
                    success, output = self._run_command(download_cmd)
                    if not success:
                        logger.warning(f"下载繁体中文语言数据失败: {output}")
                
                # 设置环境变量
                self._add_to_windows_path(tesseract_dir)
                self._set_windows_env_var("TESSDATA_PREFIX", tessdata_dir)
                
                logger.info("已自动设置Tesseract环境变量")
                
            elif self.is_linux:
                # 使用apt或yum安装Tesseract
                logger.info("正在安装Linux系统级Tesseract...")
                
                # 尝试使用apt-get安装（Debian/Ubuntu）
                cmd_apt = "sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra"
                success_apt, _ = self._run_command(cmd_apt)
                
                if not success_apt:
                    # 尝试使用yum安装（CentOS/RHEL）
                    cmd_yum = "sudo yum install -y tesseract tesseract-langpack-chi-sim tesseract-langpack-chi-tra"
                    success_yum, output = self._run_command(cmd_yum)
                    
                    if not success_yum:
                        logger.error(f"安装Tesseract失败: {output}")
                        return False
                
                # 设置环境变量
                home_dir = os.path.expanduser("~")
                shell_config_file = ""
                
                if os.path.exists(os.path.join(home_dir, ".bashrc")):
                    shell_config_file = os.path.join(home_dir, ".bashrc")
                elif os.path.exists(os.path.join(home_dir, ".bash_profile")):
                    shell_config_file = os.path.join(home_dir, ".bash_profile")
                
                if shell_config_file:
                    # 查找tessdata路径
                    tessdata_paths = [
                        "/usr/share/tesseract-ocr/4.00/tessdata",
                        "/usr/share/tesseract-ocr/tessdata",
                        "/usr/share/tessdata"
                    ]
                    
                    tessdata_dir = ""
                    for path in tessdata_paths:
                        if os.path.exists(path):
                            tessdata_dir = path
                            break
                    
                    if tessdata_dir:
                        # 添加环境变量到shell配置文件
                        with open(shell_config_file, "a") as f:
                            f.write("\n# Tesseract OCR环境变量设置\n")
                            f.write(f"export TESSDATA_PREFIX={tessdata_dir}\n")
                            f.write("export PATH=$PATH:/usr/bin\n")
                        
                        logger.info(f"Tesseract环境变量已添加到{shell_config_file}")
                        logger.info(f"请执行命令'source {shell_config_file}'使环境变量生效")
                    else:
                        logger.warning("无法找到tessdata目录，请手动设置TESSDATA_PREFIX环境变量")
                else:
                    logger.warning("未找到shell配置文件，请手动设置Tesseract环境变量")
                
            elif self.is_mac:
                # 使用Homebrew安装Tesseract
                logger.info("正在安装macOS系统级Tesseract...")
                
                # 检查是否安装了Homebrew
                brew_cmd = "which brew"
                brew_success, _ = self._run_command(brew_cmd)
                
                if not brew_success:
                    # 安装Homebrew
                    logger.info("正在安装Homebrew...")
                    brew_install_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
                    brew_success, output = self._run_command(brew_install_cmd)
                    
                    if not brew_success:
                        logger.error(f"安装Homebrew失败: {output}")
                        logger.info("请手动安装Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
                        return False
                
                # 安装Tesseract和中文语言包
                cmd_brew = "brew install tesseract tesseract-lang"
                success_brew, output = self._run_command(cmd_brew)
                
                if not success_brew:
                    logger.error(f"安装Tesseract失败: {output}")
                    return False
                
                # 设置环境变量
                home_dir = os.path.expanduser("~")
                shell_config_file = ""
                
                if os.path.exists(os.path.join(home_dir, ".zshrc")):
                    shell_config_file = os.path.join(home_dir, ".zshrc")
                elif os.path.exists(os.path.join(home_dir, ".bash_profile")):
                    shell_config_file = os.path.join(home_dir, ".bash_profile")
                
                if shell_config_file:
                    # 获取tessdata路径
                    tessdata_cmd = "brew --prefix tesseract"
                    success, tessdata_output = self._run_command(tessdata_cmd)
                    
                    if success:
                        tessdata_dir = os.path.join(tessdata_output.strip(), "share", "tessdata")
                        
                        # 添加环境变量到shell配置文件
                        with open(shell_config_file, "a") as f:
                            f.write("\n# Tesseract OCR环境变量设置\n")
                            f.write(f"export TESSDATA_PREFIX={tessdata_dir}\n")
                            f.write("export PATH=$PATH:$(brew --prefix tesseract)/bin\n")
                        
                        logger.info(f"Tesseract环境变量已添加到{shell_config_file}")
                        logger.info(f"请执行命令'source {shell_config_file}'使环境变量生效")
                    else:
                        logger.warning("无法找到tessdata目录，请手动设置TESSDATA_PREFIX环境变量")
                else:
                    logger.warning("未找到shell配置文件，请手动设置Tesseract环境变量")
            
            # 验证安装
            verify_cmd = "tesseract --version"
            success, output = self._run_command(verify_cmd)
            
            if success:
                logger.info(f"Tesseract OCR安装验证成功: {output.splitlines()[0]}")
                logger.info("Tesseract OCR引擎安装成功")
                return True
            else:
                logger.warning(f"Tesseract OCR安装验证失败: {output}")
                logger.info("Tesseract可能已安装但环境变量未生效，请重启终端或系统后重试")
                return True  # 仍然返回成功，因为可能只是环境变量未生效
        
        except Exception as e:
            logger.error(f"安装Tesseract OCR引擎时发生错误: {str(e)}")
            return False
    
    def install_paddleocr(self) -> bool:
        """安装PaddleOCR引擎并自动设置环境变量"""
        logger.info("开始安装PaddleOCR引擎...")
        
        try:
            # 安装PaddleOCR
            cmd = f"{self.pip_cmd} install paddleocr>=2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple"
            success, output = self._run_command(cmd)
            
            if not success:
                logger.error(f"安装PaddleOCR失败: {output}")
                return False
            
            # 自动设置环境变量
            if self.is_windows:
                # 获取Python安装目录和site-packages路径
                python_path = os.path.dirname(sys.executable)
                site_packages = os.path.join(python_path, 'Lib', 'site-packages')
                paddle_path = os.path.join(site_packages, 'paddle')
                paddleocr_path = os.path.join(site_packages, 'paddleocr')
                
                # 设置PATH环境变量
                self._add_to_windows_path(python_path)
                
                # 设置PYTHONPATH环境变量
                self._set_windows_env_var("PYTHONPATH", f"{site_packages};{paddle_path};{paddleocr_path}")
                
                # 设置PADDLE_ROOT环境变量
                self._set_windows_env_var("PADDLE_ROOT", paddle_path)
                
                logger.info("已自动设置PaddleOCR相关环境变量")
                
            elif self.is_linux or self.is_mac:
                # 获取用户HOME目录
                home_dir = os.path.expanduser("~")
                
                # 确定shell配置文件
                shell_config_file = ""
                if self.is_linux:
                    if os.path.exists(os.path.join(home_dir, ".bashrc")):
                        shell_config_file = os.path.join(home_dir, ".bashrc")
                    elif os.path.exists(os.path.join(home_dir, ".bash_profile")):
                        shell_config_file = os.path.join(home_dir, ".bash_profile")
                elif self.is_mac:
                    if os.path.exists(os.path.join(home_dir, ".zshrc")):
                        shell_config_file = os.path.join(home_dir, ".zshrc")
                    elif os.path.exists(os.path.join(home_dir, ".bash_profile")):
                        shell_config_file = os.path.join(home_dir, ".bash_profile")
                
                if shell_config_file:
                    # 获取Python包路径
                    python_lib_cmd = f"{sys.executable} -c 'import site; print(site.getsitepackages()[0])'"
                    success, output = self._run_command(python_lib_cmd)
                    if success:
                        site_packages = output.strip()
                        paddle_path = os.path.join(site_packages, 'paddle')
                        paddleocr_path = os.path.join(site_packages, 'paddleocr')
                        
                        # 添加环境变量到shell配置文件
                        with open(shell_config_file, "a") as f:
                            f.write(f"\n# PaddleOCR环境变量设置\n")
                            f.write(f"export PYTHONPATH=$PYTHONPATH:{site_packages}:{paddle_path}:{paddleocr_path}\n")
                            f.write(f"export PADDLE_ROOT={paddle_path}\n")
                        
                        logger.info(f"PaddleOCR环境变量已添加到{shell_config_file}")
                        logger.info(f"请执行命令'source {shell_config_file}'使环境变量生效")
                    else:
                        logger.warning("无法获取Python包路径，请手动设置环境变量")
                else:
                    logger.warning("未找到合适的shell配置文件，请手动设置环境变量")
            
            # 下载必要的模型文件
            logger.info("正在下载PaddleOCR中文OCR模型(可能需要一些时间)...")
            download_cmd = f"{sys.executable} -c \"from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=True, lang='ch')\""
            success, output = self._run_command(download_cmd)
            if not success:
                logger.warning(f"模型下载可能未完成: {output}")
                logger.info("首次使用时可能会自动下载模型")
            
            logger.info("PaddleOCR引擎安装成功")
            return True
        except Exception as e:
            logger.error(f"安装PaddleOCR时发生错误: {str(e)}")
            return False
    
    def _add_to_windows_path(self, path: str) -> bool:
        """在Windows系统添加路径到PATH环境变量"""
        if not self.is_windows:
            return False
        
        try:
            # 使用PowerShell添加PATH环境变量
            ps_cmd = f'powershell -Command "[Environment]::SetEnvironmentVariable(\'Path\', [Environment]::GetEnvironmentVariable(\'Path\', \'Machine\') + \';{path}\', \'Machine\')"'
            success, output = self._run_command(ps_cmd)
            return success
        except Exception as e:
            logger.error(f"添加PATH环境变量失败: {str(e)}")
            return False
    
    def _set_windows_env_var(self, var_name: str, var_value: str) -> bool:
        """在Windows系统设置环境变量"""
        if not self.is_windows:
            return False
        
        try:
            # 使用PowerShell设置环境变量
            ps_cmd = f'powershell -Command "[Environment]::SetEnvironmentVariable(\'{var_name}\', \'{var_value}\', \'Machine\')"'
            success, output = self._run_command(ps_cmd)
            return success
        except Exception as e:
            logger.error(f"设置{var_name}环境变量失败: {str(e)}")
            return False
    
    def install_easyocr(self) -> bool:
        """安装EasyOCR引擎并自动设置环境变量"""
        logger.info("开始安装EasyOCR引擎...")
        
        try:
            # 安装EasyOCR
            cmd = f"{self.pip_cmd} install easyocr -i https://pypi.tuna.tsinghua.edu.cn/simple"
            success, output = self._run_command(cmd)
            
            if not success:
                logger.error(f"安装EasyOCR失败: {output}")
                return False
            
            # 自动设置环境变量
            if self.is_windows:
                # 获取Python安装目录和site-packages路径
                python_path = os.path.dirname(sys.executable)
                site_packages = os.path.join(python_path, 'Lib', 'site-packages')
                easyocr_path = os.path.join(site_packages, 'easyocr')
                
                # 设置PATH环境变量
                self._add_to_windows_path(python_path)
                
                # 设置PYTHONPATH环境变量
                self._set_windows_env_var("PYTHONPATH", f"{site_packages};{easyocr_path}")
                
                # 设置EASYOCR_MODULE_PATH环境变量
                self._set_windows_env_var("EASYOCR_MODULE_PATH", easyocr_path)
                
                # 设置CUDA相关环境变量(如果可用)
                if self.has_cuda:
                    # 尝试定位CUDA安装路径
                    cuda_paths = [
                        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
                        "C:\\CUDA"
                    ]
                    
                    for base_path in cuda_paths:
                        if os.path.exists(base_path):
                            # 找到最新版本的CUDA文件夹
                            cuda_versions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith("v")]
                            if cuda_versions:
                                latest_cuda = sorted(cuda_versions)[-1]
                                cuda_path = os.path.join(base_path, latest_cuda)
                                
                                # 设置CUDA_HOME环境变量
                                self._set_windows_env_var("CUDA_HOME", cuda_path)
                                
                                # 添加CUDA bin到PATH
                                cuda_bin = os.path.join(cuda_path, "bin")
                                self._add_to_windows_path(cuda_bin)
                                
                                logger.info(f"已设置CUDA环境变量，CUDA路径: {cuda_path}")
                                break
                
                logger.info("已自动设置EasyOCR相关环境变量")
                
            elif self.is_linux or self.is_mac:
                # 获取用户HOME目录
                home_dir = os.path.expanduser("~")
                
                # 确定shell配置文件
                shell_config_file = ""
                if self.is_linux:
                    if os.path.exists(os.path.join(home_dir, ".bashrc")):
                        shell_config_file = os.path.join(home_dir, ".bashrc")
                    elif os.path.exists(os.path.join(home_dir, ".bash_profile")):
                        shell_config_file = os.path.join(home_dir, ".bash_profile")
                elif self.is_mac:
                    if os.path.exists(os.path.join(home_dir, ".zshrc")):
                        shell_config_file = os.path.join(home_dir, ".zshrc")
                    elif os.path.exists(os.path.join(home_dir, ".bash_profile")):
                        shell_config_file = os.path.join(home_dir, ".bash_profile")
                
                if shell_config_file:
                    # 获取Python包路径
                    python_lib_cmd = f"{sys.executable} -c 'import site; print(site.getsitepackages()[0])'"
                    success, output = self._run_command(python_lib_cmd)
                    if success:
                        site_packages = output.strip()
                        easyocr_path = os.path.join(site_packages, 'easyocr')
                        
                        # 添加环境变量到shell配置文件
                        with open(shell_config_file, "a") as f:
                            f.write(f"\n# EasyOCR环境变量设置\n")
                            f.write(f"export PYTHONPATH=$PYTHONPATH:{site_packages}:{easyocr_path}\n")
                            f.write(f"export EASYOCR_MODULE_PATH={easyocr_path}\n")
                            
                            # 如果有CUDA，设置CUDA相关环境变量
                            if self.has_cuda:
                                if self.is_linux:
                                    f.write("export CUDA_HOME=/usr/local/cuda\n")
                                    f.write("export PATH=$PATH:$CUDA_HOME/bin\n")
                                    f.write("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64\n")
                                elif self.is_mac:
                                    f.write("export CUDA_HOME=/usr/local/cuda\n")
                                    f.write("export PATH=$PATH:$CUDA_HOME/bin\n")
                                    f.write("export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$CUDA_HOME/lib\n")
                        
                        logger.info(f"EasyOCR环境变量已添加到{shell_config_file}")
                        logger.info(f"请执行命令'source {shell_config_file}'使环境变量生效")
                    else:
                        logger.warning("无法获取Python包路径，请手动设置环境变量")
                else:
                    logger.warning("未找到合适的shell配置文件，请手动设置环境变量")
            
            # 下载中文模型
            logger.info("正在下载EasyOCR中文模型(可能需要一些时间)...")
            download_cmd = f"{sys.executable} -c \"import easyocr; reader = easyocr.Reader(['ch_sim', 'en'])\""
            success, output = self._run_command(download_cmd)
            if not success:
                logger.warning(f"模型下载可能未完成: {output}")
                logger.info("首次使用时可能会自动下载模型")
            
            logger.info("EasyOCR引擎安装成功")
            return True
        except Exception as e:
            logger.error(f"安装EasyOCR引擎时发生错误: {str(e)}")
            return False
    
    def install_rapidocr(self) -> bool:
        """安装RapidOCR引擎"""
        logger.info("安装RapidOCR引擎...")
        
        # 安装RapidOCR
        cmd = f"{self.pip_cmd} install rapidocr_onnxruntime>=1.3.0"
        success, output = self._run_command(cmd)
        
        if success:
            logger.info("RapidOCR引擎安装成功")
        else:
            logger.error(f"安装RapidOCR失败: {output}")
        
        return success
    
    def install_manga(self) -> bool:
        """安装MangaOCR引擎"""
        logger.info("安装MangaOCR引擎...")
        
        # 安装MangaOCR
        cmd = f"{self.pip_cmd} install manga-ocr>=0.1.7"
        success, output = self._run_command(cmd)
        
        if success:
            logger.info("MangaOCR引擎安装成功")
        else:
            logger.error(f"安装MangaOCR失败: {output}")
        
        return success
    
    def install_mmocr(self) -> bool:
        """安装MMOCR引擎"""
        logger.info("安装MMOCR引擎...")
        
        # 安装MMCV
        cmd_mmcv = f"{self.pip_cmd} install -U openmim"
        success_mmcv, output = self._run_command(cmd_mmcv)
        
        if not success_mmcv:
            logger.error(f"安装openmim失败: {output}")
            return False
        
        # 使用mim安装mmcv
        cmd_mim = f"{sys.executable} -m mim install mmcv"
        success_mim, output = self._run_command(cmd_mim)
        
        if not success_mim:
            logger.error(f"安装mmcv失败: {output}")
            return False
        
        # 安装MMOCR
        cmd = f"{self.pip_cmd} install mmocr>=1.0.0"
        success, output = self._run_command(cmd)
        
        if success:
            logger.info("MMOCR引擎安装成功")
        else:
            logger.error(f"安装MMOCR失败: {output}")
        
        return success
    
    def install_transformers(self) -> bool:
        """安装Transformers库（TrOCR和Donut的依赖）"""
        logger.info("安装Transformers库（TrOCR和Donut的依赖）...")
        
        # 安装PyTorch
        if self.has_cuda:
            # 安装支持CUDA的PyTorch
            cmd_torch = f"{self.pip_cmd} install torch torchvision torchaudio"
        else:
            # 安装CPU-only的PyTorch
            cmd_torch = f"{self.pip_cmd} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        
        success_torch, output = self._run_command(cmd_torch)
        
        if not success_torch:
            logger.error(f"安装PyTorch失败: {output}")
            return False
        
        # 安装Transformers
        cmd = f"{self.pip_cmd} install transformers>=4.18.0"
        success, output = self._run_command(cmd)
        
        if success:
            logger.info("Transformers库安装成功")
        else:
            logger.error(f"安装Transformers失败: {output}")
        
        return success
    
    def install_donut(self) -> bool:
        """安装Donut引擎"""
        logger.info("安装Donut引擎...")
        
        # 安装Donut
        cmd = f"{self.pip_cmd} install donut-python>=1.0.9"
        success, output = self._run_command(cmd)
        
        if success:
            logger.info("Donut引擎安装成功")
        else:
            logger.error(f"安装Donut失败: {output}")
        
        return success
    
    def install_all(self) -> Dict[str, bool]:
        """安装所有依赖"""
        results = {}
        
        # 安装核心依赖
        results["core"] = self.install_core_dependencies()
        
        # 安装各引擎
        results["tesseract"] = self.install_tesseract()
        results["paddleocr"] = self.install_paddleocr()
        results["easyocr"] = self.install_easyocr()
        results["rapidocr"] = self.install_rapidocr()
        results["manga"] = self.install_manga()
        results["mmocr"] = self.install_mmocr()
        
        # 安装Transformers（TrOCR依赖）
        results["transformers"] = self.install_transformers()
        
        # 安装Donut
        results["donut"] = self.install_donut()
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="OCR依赖安装助手")
    parser.add_argument("--all", action="store_true", help="安装所有依赖")
    parser.add_argument("--core", action="store_true", help="安装核心依赖")
    parser.add_argument("--tesseract", action="store_true", help="安装Tesseract OCR引擎")
    parser.add_argument("--paddleocr", action="store_true", help="安装PaddleOCR引擎")
    parser.add_argument("--easyocr", action="store_true", help="安装EasyOCR引擎")
    parser.add_argument("--rapidocr", action="store_true", help="安装RapidOCR引擎")
    parser.add_argument("--manga", action="store_true", help="安装MangaOCR引擎")
    parser.add_argument("--mmocr", action="store_true", help="安装MMOCR引擎")
    parser.add_argument("--transformers", action="store_true", help="安装Transformers库（TrOCR依赖）")
    parser.add_argument("--donut", action="store_true", help="安装Donut引擎")
    
    args = parser.parse_args()
    
    # 创建安装助手
    installer = DependencyInstaller()
    
    # 如果没有指定任何选项，则显示帮助
    if not any(vars(args).values()):
        parser.print_help()
        logger.info("\n请指定要安装的依赖")
        return
    
    # 安装指定依赖
    if args.all:
        logger.info("开始安装所有依赖...")
        results = installer.install_all()
        
        # 显示安装结果
        logger.info("\n安装结果:")
        for name, success in results.items():
            logger.info(f"{name}: {'成功' if success else '失败'}")
        
        # 检查是否有失败的安装
        if not all(results.values()):
            logger.warning("部分依赖安装失败，请手动安装或检查问题后重试")
        else:
            logger.info("所有依赖安装成功！")
    else:
        # 安装核心依赖
        if args.core:
            installer.install_core_dependencies()
        
        # 安装各引擎
        if args.tesseract:
            installer.install_tesseract()
        
        if args.paddleocr:
            installer.install_paddleocr()
        
        if args.easyocr:
            installer.install_easyocr()
        
        if args.rapidocr:
            installer.install_rapidocr()
        
        if args.manga:
            installer.install_manga()
        
        if args.mmocr:
            installer.install_mmocr()
        
        if args.transformers:
            installer.install_transformers()
        
        if args.donut:
            installer.install_donut()


if __name__ == "__main__":
    main() 