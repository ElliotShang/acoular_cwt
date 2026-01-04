
import sys
import os
from pathlib import Path

# --- 环境变量配置与依赖导入 (防止卡死) ---
print("正在设置环境变量 (增强版)...", flush=True)
# [CRITICAL] 允许库重复加载，解决 Anaconda MKL 冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置环境变量以避免 OpenBLAS 与 Numba 的线程冲突
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
# 限制 Numba 线程数
os.environ['NUMBA_NUM_THREADS'] = '1'
# 防止 Numba 使用 AVX 等高级指令集导致的老旧 CPU 崩溃或虚拟机崩溃
os.environ['NUMBA_CPU_NAME'] = 'generic'
os.environ['NUMBA_DISABLE_AVX'] = '1'  # 重新禁用 AVX 以防止 0xC0000005 崩溃
# 禁用 ETS (Enthought Tool Suite) 的 GUI 工具包检测，防止卡死
os.environ['ETS_TOOLKIT'] = 'null'
# 重定向 Numba 缓存到临时目录，防止读取旧缓存卡死
import tempfile
os.environ['NUMBA_CACHE_DIR'] = os.path.join(tempfile.gettempdir(), 'numba_cache')

print(f"  Numba cache dir: {os.environ['NUMBA_CACHE_DIR']}", flush=True)

# [CRITICAL] 导入顺序调整: h5py 必须最先导入以锁定 DLL 版本
print("正在导入 h5py (环境稳定器)...", flush=True)
import h5py

print("正在导入 numpy...", flush=True)
import numpy as np

# 逐步导入依赖
print("正在导入 scipy...", flush=True)
import scipy
print("正在导入 numba...", flush=True)
import numba
print(f"  Numba version: {numba.__version__}")
print("正在导入 tables...", flush=True)
import tables
print("正在导入 traits...", flush=True)
import traits

print("正在分模块导入 acoular...", flush=True)
try:
    print("  -> importing acoular.configuration...", flush=True)
    import acoular.configuration
    # 立即禁用缓存
    acoular.configuration.config.global_caching = "none"
    print("  -> config imported and caching disabled.", flush=True)
    
    print("  -> importing acoular.sources...", flush=True)
    import acoular.sources
    print("  -> importing acoular.microphones...", flush=True)
    import acoular.microphones
    print("  -> importing acoular.spectra...", flush=True)
    import acoular.spectra
    print("  -> importing acoular.grids...", flush=True)
    import acoular.grids
    print("  -> importing acoular.environments...", flush=True)
    import acoular.environments
    print("  -> importing acoular.fbeamform (JIT heavy)...", flush=True)
    import acoular.fbeamform
    print("  -> importing acoular (full)...", flush=True)
    import acoular as ac
except Exception as e:
    print(f"\n!!! 导入 Acoular 模块时出错: {e}")
    import traceback
    traceback.print_exc()

# ----------------------------------------

# Adjust path to find mock_cwt in the current directory or package
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    # Try relative import if run as module
    from .mock_cwt import MockWaveletSpectra
except ImportError:
    try:
        # Try direct import if run as script in the folder
        from mock_cwt import MockWaveletSpectra
    except ImportError:
        # Try importing as package
        from acoular_cwt.mock_cwt import MockWaveletSpectra

def create_dummy_signal(filename, fs=51200, duration=1.0):
    """
    如果文件不存在，创建一个符合 Acoular 格式的 dummy h5 文件。
    使用 Acoular 内置工具生成包含单个点声源的信号。
    """
    print(f"Creating dummy data file: {filename}")
    
    # 1. 加载麦克风阵列几何 (使用 Acoular 自带的 array_64.xml)
    micgeofile = Path(ac.__file__).parent / 'xml' / 'array_64.xml'
    if not micgeofile.exists():
        print(f"Warning: Mic geometry file {micgeofile} not found. Ensure acoular is installed correctly.")
        # Fallback or exit? For now let's assume it exists as it is standard in acoular
    
    m = ac.MicGeom(file=micgeofile)
    
    # 2. 创建噪声源
    num_samples = int(duration * fs)
    n1 = ac.WNoiseGenerator(sample_freq=fs, num_samples=num_samples, seed=2)
    
    # 3. 创建点声源
    p1 = ac.PointSource(signal=n1, mics=m, loc=(0, 0, -0.3))
    
    # 4. 保存为 H5 文件
    wh5 = ac.WriteH5(source=p1, file=filename)
    wh5.save()

def run_verification():
    # 1. 准备数据源
    # 在当前脚本目录下创建数据文件，避免污染根目录或其他地方
    datafile = Path(__file__).parent / "test_signal.h5"
    create_dummy_signal(str(datafile))
    
    if not datafile.exists():
        print("Error: Data file could not be created.")
        return

    # 连接 TimeSamples
    ts = ac.TimeSamples(file=str(datafile))
    print(f"TimeSamples loaded: {ts.num_samples} samples, {ts.num_channels} channels, {ts.sample_freq} Hz")

    # 2. 实例化 MockWaveletSpectra
    print("Instantiating MockWaveletSpectra...")
    # target_freq 设置为 1000Hz, block_size 等参数虽然在 CWT 实现中可能不直接对应 FFT 块，
    # 但 PowerSpectra 父类可能需要它们初始化。
    mws = MockWaveletSpectra(source=ts, target_freq=1000.0)

    # 3. 动态改变 target_time 并验证 CSM
    test_times = [0.1, 0.2, 0.5] # 测试几个不同的时间点 (秒)
    
    print("\nStarting CSM verification loop:")
    print("-" * 50)
    
    previous_csm = None

    for t in test_times:
        print(f"Updating target_time to: {t:.4f} s")
        mws.target_time = t
        
        # 获取 CSM
        # 注意：每次访问 .csm 属性时，由于 depends_on 机制，应该触发重新计算
        csm = mws.csm
        
        # 检查形状 (frequencies, num_mics, num_mics)
        # mock_cwt 中 FREQ_NUM=1
        expected_shape = (1, ts.num_channels, ts.num_channels)
        
        print(f"  -> CSM Shape: {csm.shape}")
        
        if csm.shape != expected_shape:
            print(f"  [FAIL] Shape mismatch! Expected {expected_shape}, got {csm.shape}")
        else:
            print(f"  [PASS] Shape correct.")
            
        # 计算特征值或范数来代表 CSM 内容
        norm_val = np.linalg.norm(csm)
        print(f"  -> CSM Norm: {norm_val:.4f}")
        
        if previous_csm is not None:
            # 验证 CSM 是否发生变化
            diff = np.linalg.norm(csm - previous_csm)
            if diff > 1e-10:
                 print(f"  [PASS] CSM updated successfully (diff: {diff:.4e})")
            else:
                 print(f"  [WARNING] CSM did not change! (diff: {diff:.4e})")
        
        previous_csm = np.copy(csm)
        print("-" * 50)

    print("\nVerification process completed.")

if __name__ == "__main__":
    run_verification()

