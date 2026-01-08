from .spectra import WaveletSpectra
from .wisdom import (
    generate_wisdom,
    list_wisdom_files,
    clear_wisdom,
    get_wisdom_dir,
)

# 可以在这里放版本号
__version__ = "0.1.0"

# 方便调试：尝试导入 C++ 扩展并打印状态
try:
    from. import _wcsm
    _backend_available = True
except ImportError:
    _backend_available = False
    # 这里不要报错，允许用户在没编译时查看文档或代码

__all__ = [
    'WaveletSpectra',
    'generate_wisdom',
    'list_wisdom_files',
    'clear_wisdom',
    'get_wisdom_dir',
    '__version__',
]