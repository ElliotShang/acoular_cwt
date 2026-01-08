"""
FFTW Wisdom 管理模块

提供 wisdom 文件的生成、存储和加载功能。
Wisdom 文件存放在用户主目录的 .acoular_cwt/ 目录下。
"""

import os
from pathlib import Path
import fcwt

# 默认 wisdom 存放目录
WISDOM_DIR = Path.home() / '.acoular_cwt' / 'wisdom'


def get_wisdom_dir():
    """获取 wisdom 文件存放目录"""
    return WISDOM_DIR


def ensure_wisdom_dir():
    """确保 wisdom 目录存在"""
    WISDOM_DIR.mkdir(parents=True, exist_ok=True)
    return WISDOM_DIR


def generate_wisdom(max_size=8192, threads=1, flags="FFTW_MEASURE"):
    """
    生成 FFTW Wisdom 优化文件
    
    Wisdom 文件将存放在 ~/.acoular_cwt/wisdom/ 目录下。
    生成后，后续的 CWT 计算将自动加载这些优化方案以提高性能。
    
    Parameters
    ----------
    max_size : int
        最大 FFT 大小（必须是 2 的幂次，默认 8192）。
        会为从 2048 到 max_size 的所有 2 的幂次生成优化方案。
    threads : int
        线程数（默认 1）。应与运行时的 OMP_NUM_THREADS 一致。
    flags : str
        FFTW 优化标志，可选值：
        - "FFTW_ESTIMATE": 最快，优化效果最差
        - "FFTW_MEASURE": 中等速度，推荐（默认）
        - "FFTW_PATIENT": 较慢，优化效果较好
        - "FFTW_EXHAUSTIVE": 最慢，优化效果最好
    
    Returns
    -------
    list
        生成的 wisdom 文件路径列表
    
    Examples
    --------
    >>> import acoular_cwt
    >>> acoular_cwt.generate_wisdom()  # 使用默认参数生成
    >>> acoular_cwt.generate_wisdom(max_size=16384, threads=4)  # 自定义参数
    """
    # 验证 max_size 是 2 的幂次
    if max_size & (max_size - 1) != 0 or max_size < 2048:
        raise ValueError(f"max_size 必须是 2 的幂次且 >= 2048，当前值: {max_size}")
    
    # 验证 flags
    valid_flags = ["FFTW_ESTIMATE", "FFTW_MEASURE", "FFTW_PATIENT", "FFTW_EXHAUSTIVE"]
    if flags not in valid_flags:
        raise ValueError(f"flags 必须是 {valid_flags} 之一，当前值: {flags}")
    
    # 确保目录存在
    wisdom_dir = ensure_wisdom_dir()
    
    # 保存当前工作目录
    original_cwd = os.getcwd()
    
    try:
        # 切换到 wisdom 目录（因为 fCWT 会在当前目录生成文件）
        os.chdir(wisdom_dir)
        
        print(f"正在生成 FFTW Wisdom 文件...")
        print(f"  存放目录: {wisdom_dir}")
        print(f"  最大 FFT 大小: {max_size}")
        print(f"  线程数: {threads}")
        print(f"  优化模式: {flags}")
        print()
        
        # 创建 FCWT 对象
        wavelet = fcwt.Morlet(6.0)
        fcwt_obj = fcwt.FCWT(wavelet, threads, True, True)
        
        # 生成优化方案
        fcwt_obj.create_FFT_optimization_plan(max_size, flags)
        
        # 列出生成的文件
        generated_files = list(wisdom_dir.glob('n*_t*.wis'))
        
        print(f"\n✓ Wisdom 文件生成成功！")
        print(f"生成的文件:")
        for wis_file in generated_files:
            file_size = wis_file.stat().st_size
            print(f"  - {wis_file.name} ({file_size} bytes)")
        
        return [str(f) for f in generated_files]
        
    finally:
        # 恢复原工作目录
        os.chdir(original_cwd)


def list_wisdom_files():
    """
    列出已生成的 wisdom 文件
    
    Returns
    -------
    list
        wisdom 文件路径列表
    """
    if not WISDOM_DIR.exists():
        return []
    return [str(f) for f in WISDOM_DIR.glob('n*_t*.wis')]


def clear_wisdom():
    """
    清除所有已生成的 wisdom 文件
    
    Returns
    -------
    int
        删除的文件数量
    """
    if not WISDOM_DIR.exists():
        return 0
    
    count = 0
    for wis_file in WISDOM_DIR.glob('n*_t*.wis'):
        wis_file.unlink()
        count += 1
    
    print(f"已删除 {count} 个 wisdom 文件")
    return count


def setup_wisdom_environment():
    """
    设置环境以便 fCWT 能找到 wisdom 文件
    
    在运行 CWT 计算前调用此函数，它会临时切换工作目录到 wisdom 存放位置。
    
    Returns
    -------
    str or None
        原始工作目录（如果需要恢复），如果 wisdom 目录不存在则返回 None
    """
    if not WISDOM_DIR.exists() or not list(WISDOM_DIR.glob('n*_t*.wis')):
        return None
    
    original_cwd = os.getcwd()
    os.chdir(WISDOM_DIR)
    return original_cwd


def restore_working_directory(original_cwd):
    """
    恢复原始工作目录
    
    Parameters
    ----------
    original_cwd : str
        要恢复的工作目录路径
    """
    if original_cwd:
        os.chdir(original_cwd)

