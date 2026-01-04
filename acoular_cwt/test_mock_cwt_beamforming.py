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
print("正在导入 matplotlib...", flush=True)
import matplotlib.pyplot as plt

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

def create_test_signal_if_missing(filename, fs=51200, duration=1.0):
    if Path(filename).exists():
        print(f"Using existing data file: {filename}")
        return

    print(f"Creating dummy data file: {filename}")
    
    # 1. 加载麦克风阵列几何 (使用 Acoular 自带的 array_64.xml)
    micgeofile = Path(ac.__file__).parent / 'xml' / 'array_64.xml'
    if not micgeofile.exists():
         # Fallback to local xml if exists, or error
         print("Error: acoular mic geometry not found.")
         return

    m = ac.MicGeom(file=micgeofile)
    
    # 2. 创建噪声源
    num_samples = int(duration * fs)
    
    # 使用固定强度的白噪声 (RMS=1.0)
    n1 = ac.WNoiseGenerator(sample_freq=fs, num_samples=num_samples, seed=1, rms=1.0)
    
    # 3. 创建移动声源
    # 定义移动轨迹，在 0.5s 时刻实现位置跳变
    # 增加定义点以实现更清晰的跳变
    # 0-0.49s: 位于 (-0.15, -0.15, -0.3)
    # 0.5-1.0s: 位于 (0.15, 0.15, -0.3)
    t1 = ac.Trajectory()
    t1.points = {
        0.0: (-0.15, -0.15, -0.3),
        0.48: (-0.15, -0.15, -0.3),  # 保持在左下方
        0.5: (0.15, 0.15, -0.3),      # 跳变到右上方
        1.0: (0.15, 0.15, -0.3)       # 保持在右上方
    }
    
    # 使用 MovingPointSource
    p1 = ac.MovingPointSource(signal=n1, mics=m, trajectory=t1)
    
    # 4. 保存为 H5 文件
    wh5 = ac.WriteH5(source=p1, file=filename)
    wh5.save()

def run_beamforming_test():
    # 1. 准备数据源
    datafile = Path(__file__).parent / "test_moving_source.h5"
    # 强制重新创建文件以确保包含最新的移动声源
    if datafile.exists():
        try:
            os.remove(datafile)
        except:
            pass
    create_test_signal_if_missing(str(datafile))
    
    micgeofile = Path(ac.__file__).parent / 'xml' / 'array_64.xml'
    
    # 连接组件
    mg = ac.MicGeom(file=micgeofile)
    ts = ac.TimeSamples(file=str(datafile))
    
    # 2. 实例化 MockWaveletSpectra
    print("Instantiating MockWaveletSpectra...")
    # 目标频率 2000Hz (CWT将只计算这个频率的CSM)
    mws = MockWaveletSpectra(source=ts, target_freq=2000.0)
    
    # 3. 设置 Beamformer 流程
    print("Setting up Beamformer...")
    rg = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=-0.3, increment=0.01)
    st = ac.SteeringVector(grid=rg, mics=mg)
    
    # BeamformerBase 接受 freq_data=mws
    # 此时 mws 假装是 PowerSpectra
    bb = ac.BeamformerBase(freq_data=mws, steer=st)
    
    # 4. 测试两个不同的时间点并绘图
    # 声源在 0.5s 发生位置跳变
    # t=0.2s 时应在 (-0.15, -0.15)
    # t=0.8s 时应在 (0.15, 0.15)
    test_times = [0.2, 0.8]
    
    plt.figure(figsize=(10, 5))
    
    for i, t in enumerate(test_times):
        expected_pos = "(-0.15, -0.15)" if t < 0.5 else "(0.15, 0.15)"
        print(f"\n{'='*60}")
        print(f"Computing Beamforming map at t={t}s")
        print(f"  Expected source position: {expected_pos}")
        print(f"{'='*60}")
        
        # 更新 MockWaveletSpectra 的时间
        mws.target_time = t
        
        # [CRITICAL FIX]
        # Acoular 的 BeamformerBase 会缓存结果。它依赖于 freq_data.digest。
        # 如果 MockWaveletSpectra 的 digest 机制没有完美地将 target_time 的变化
        # 传递给 BeamformerBase，后者就会直接返回缓存的旧结果。
        # 为了强制刷新，最简单的方法是在这里重新实例化 BeamformerBase，或者清除其缓存。
        # 由于我们全局禁用了缓存 (global_caching="none")，理论上不应该缓存，
        # 但 Beamformer 内部对象可能仍持有旧状态。
        # 重建 BeamformerBase 是最安全的。
        bb = ac.BeamformerBase(freq_data=mws, steer=st)
        
        # 调用 bb.synthetic()
        # 参数1: 频率 (Must match target_freq because MockWaveletSpectra only returns that freq)
        # 参数2: 带宽 (设为 0 或 3 这里的实现其实 MockWaveletSpectra 只有一个频率，
        # Acoular 的 synthetic 会在其返回的 fftfreq 中查找这个频率)
        
        # 注意: bb.synthetic(f, oct) 会去 freq_data.fftfreq() 里找在 f 周围的频率。
        # 我们的 Mock 只返回 [2000.0]。
        # 所以查询 2000 应该能命中。
        try:
            pm = bb.synthetic(2000.0, 0) # 0 bandwidth = single frequency line
            
            Lm = ac.L_p(pm)
            
            print(f"  -> Result map max level: {Lm.max():.2f} dB")
            
            plt.subplot(1, 2, i+1)
            plt.imshow(Lm.T, origin='lower', vmin=Lm.max() - 10, extent=rg.extent, interpolation='bicubic')
            plt.colorbar(label='SPL [dB]')
            
            # 标注预期声源位置
            if t < 0.5:
                plt.plot(-0.15, -0.15, 'r*', markersize=15, label='Expected source')
            else:
                plt.plot(0.15, 0.15, 'r*', markersize=15, label='Expected source')
            
            plt.legend()
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')
            plt.title(f't={t}s (Freq=2000Hz)\nSource at {expected_pos}')
            
        except Exception as e:
            print(f"  [ERROR] Beamforming failed: {e}")
            import traceback
            traceback.print_exc()

    plt.tight_layout()
    plt.show()
    print("\nTest completed.")

if __name__ == "__main__":
    run_beamforming_test()

