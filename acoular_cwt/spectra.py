import numpy as np
import acoular
from traits.api import Float, Int, Trait, Property, CArray, cached_property, Bool
from acoular.internal import digest # 用于生成缓存签名
import os
import sys
from pathlib import Path

# Wisdom 文件存放目录
_WISDOM_DIR = Path.home() / '.acoular_cwt' / 'wisdom'

# 确保在 Windows 上能找到依赖的 DLL (如 fftw3f.dll)
if hasattr(os, 'add_dll_directory'):
    try:
        # 1. 开发模式：尝试从 external/fCWT/libs 加载
        _package_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_package_dir)
        _external_dll_dir = os.path.join(_project_root, 'external', 'fCWT', 'libs')
        if os.path.exists(_external_dll_dir):
            os.add_dll_directory(_external_dll_dir)
        
        # 2. 发布模式：尝试从当前包目录加载 (setup.py 会把 DLL 复制到这里)
        os.add_dll_directory(_package_dir)
    except Exception:
        pass

try:
    from . import _wcsm
except ImportError:
    import _wcsm

class WaveletSpectra(acoular.PowerSpectra):
    """
        基于 CWT/STFT 的瞬态声源成像谱生成器。
        该类伪装成标准的 PowerSpectra，但通过 C++ 后端动态计算瞬时 CSM。
        """

    # --- 用户交互参数 ---

    # 目标时刻 (秒)
    target_time = Float(0.0, desc="instantaneous time to process (s)")

    # 目标频率 (Hz)
    # 注意：标准 PowerSpectra 计算所有频率，我们只算这一个
    target_freq = Float(1000.0, desc="target frequency to image (Hz)")

    # 算法选择
    method = Trait('CWT', 'STFT', desc="method for time-frequency analysis")

    # 算法参数
    smoothing_width = Int(0, desc="temporal smoothing width (samples)")
    # STFT专用: 窗口大小 / CWT专用: Sigma
    # 为了复用界面，我们定义一个通用参数，或者分别定义
    param_val = Float(6.0, desc="CWT sigma or STFT block_size")
    # 覆盖 csm 属性，使其不再读取 HDF5，而是动态计算
    # dependencies 定义了哪些参数变化时需要重新计算
    csm = Property(depends_on=['source.digest', 'target_time', 'target_freq',
                               'method', 'smoothing_width', 'param_val'])
    # 禁用 Acoular 默认的磁盘缓存 (对于瞬态交互，磁盘IO比计算还慢)
    cached = Bool(False)

    @cached_property
    def _get_csm(self):
        # 1. 准备数据源
        # Acoular 的 source 通常是懒加载的，我们需要确保拿到采样率等信息
        fs = self.source.sample_freq
        # 将时间转换为样本索引
        global_idx = int(self.target_time * fs)
        
        # 获取总样本数
        # 优先尝试直接访问 numsamples
        # 如果失败，尝试通过 HDF5 句柄访问 (针对 TimeSamples 对象)
        num_samples = 0
        try:
            num_samples = self.source.numsamples
        except AttributeError:
            if hasattr(self.source, 'h5f'):
                try:
                    # 尝试从 HDF5 根节点的 time_data 表获取长度
                    # 这通常是 TimeSamples 存储数据的地方
                    if hasattr(self.source.h5f.root, 'time_data'):
                        num_samples = self.source.h5f.root.time_data.shape[0]
                except Exception:
                    pass
        
        # 如果仍然获取失败，记录错误并返回空结果
        if num_samples == 0:
            # 尝试最后一种常见情况：数据可能在内存中 (e.g. MaskedTimeSamples)
            if hasattr(self.source, 'data'):
                 try:
                     num_samples = self.source.data.shape[0]
                 except AttributeError:
                     pass

        if num_samples == 0:
            # 这是一个严重错误，意味着无法确定数据长度
            # 为了防止 C++ 端崩溃，我们返回一个最小的 valid 结果（全零）
            # 或者抛出异常让用户知道。这里选择打印错误并返回零值以保持程序运行。
            print(f"[ERROR] Could not determine numsamples for source type: {type(self.source)}")
            return np.zeros((1, 1, 1), dtype=complex)

        # 边界检查
        if global_idx < 0 or global_idx >= num_samples:
            # 这是一个常见错误，返回全零矩阵比报错更鲁棒，或者打印警告
            pass

        safe_radius = int(1.0 * fs)
        start = max(0, global_idx - safe_radius)
        end = min(num_samples, global_idx + safe_radius)

        data_chunk = self.source.data[start:end]
        # 计算相对于 chunk 的目标索引
        rel_idx = global_idx - start

        # 切换到 wisdom 目录以便 fCWT 能找到优化文件
        original_cwd = None
        if _WISDOM_DIR.exists() and list(_WISDOM_DIR.glob('n*_t*.wis')):
            original_cwd = os.getcwd()
            os.chdir(_WISDOM_DIR)
        
        try:
            # 调用C++扩展计算CSM (1,M,M)
            csm_out = _wcsm.compute_csm(
                data_chunk,
                float(fs),
                float(self.target_freq),
                int(rel_idx),
                self.method,
                self.smoothing_width,
                float(self.param_val),
            )
        finally:
            # 恢复原工作目录
            if original_cwd:
                os.chdir(original_cwd)

        return csm_out

    def fftfreq(self):
        """
        欺骗 BeamformerBase。
        它会调用这个方法查询 CSM 对应的频率列表。
        我们返回只包含 target_freq 的列表。
        这样 Beamformer.synthetic(freq,...) 就会索引到第 0 个 CSM。
        """
        return np.array([self.target_freq], dtype=float)

    def calc_csm(self):
        """
        覆盖基类的 calc_csm。
        PowerSpectra 在某些初始化路径会调用这个。
        我们什么都不做，因为我们的 _get_csm 是按需触发的。
        """
        pass

