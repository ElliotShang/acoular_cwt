# Acoular-CWT

**Acoular-CWT** 是一个为 [Acoular](http://www.acoular.org/) 框架开发的高性能扩展库，旨在提供基于 **连续小波变换 (CWT)** 和 **短时傅里叶变换 (STFT)** 的瞬时互谱矩阵 (CSM) 计算能力。

它利用 [fCWT](https://github.com/ArtsiomCV/fCWT) 库（CWT 领域最快的实现之一）和 [FFTW3](http://www.fftw.org/) 进行底层加速，通过 C++ 扩展和 OpenMP 并行计算，实现了实时/近实时的瞬态声源成像。

## 主要特性

- **高性能 CWT 后端**：基于 fCWT 库，利用 AVX/AVX2 指令集加速。
- **并行计算**：OpenMP 多线程支持，充分利用多核 CPU。
- **无缝集成**：提供 `WaveletSpectra` 类，完全兼容 Acoular 的 `PowerSpectra` 接口。
- **动态计算**：无需预先生成巨大的 CSM 矩阵，按需计算特定时刻和频率的 CSM，节省内存。
- **Wisdom 优化**：支持 FFTW Wisdom 机制，根据硬件自动优化 FFT 执行计划。

## 安装

### 前置要求

- Python 3.8+
- C++ 编译器 (支持 C++20)
- CMake 3.18+
- [fCWT](https://github.com/ArtsiomCV/fCWT) (通常作为子模块包含)
- [FFTW3](http://www.fftw.org/) (Windows 下需提供 .lib/.dll)

### 源码安装

```bash
# 1. 克隆仓库
git clone https://github.com/yourusername/acoular-cwt.git
cd acoular-cwt

# 2. 安装 Python 依赖
pip install -r requirements.txt
# 或者直接安装：
pip install numpy acoular traits

# 3. 编译并安装
pip install .
```

### 开发模式安装

```bash
pip install -e .
```

## 使用示例

### 1. 基础用法

```python
import acoular as ac
from acoular_cwt import WaveletSpectra

# 1. 加载数据
ts = ac.TimeSamples(name='source_data.h5')
mg = ac.MicGeom(from_file='array_64.xml')

# 2. 创建 WaveletSpectra 对象
# target_freq: 目标频率 (Hz)
# method: 'CWT' (默认) 或 'STFT'
mws = WaveletSpectra(source=ts, target_freq=2000.0, method='CWT')

# 3. 设置 Beamformer
rg = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=0.3, increment=0.01)
st = ac.SteeringVector(grid=rg, mics=mg)
bb = ac.BeamformerBase(freq_data=mws, steer=st)

# 4. 计算特定时刻的声源图
target_time = 0.5  # 秒
mws.target_time = target_time

# 重新实例化 Beamformer 以刷新缓存 (因为 mws 是动态的)
bb = ac.BeamformerBase(freq_data=mws, steer=st)
pm = bb.synthetic(2000.0, 0)
Lm = ac.L_p(pm)

# 5. 绘图
import matplotlib.pyplot as plt
plt.imshow(Lm.T, origin='lower', extent=rg.extent, interpolation='bicubic')
plt.show()
```

### 2. 性能优化 (FFTW Wisdom)

首次安装后，建议生成 FFTW 优化方案（Wisdom 文件），这将根据您的硬件配置优化计算性能。

```python
import acoular_cwt

# 生成优化文件 (只需运行一次)
# 文件将保存在 ~/.acoular_cwt/wisdom/
acoular_cwt.generate_wisdom(max_size=8192, threads=4)
```

## 配置说明

- **环境变量**：
  - `OMP_NUM_THREADS`: 控制并行计算线程数 (建议设置为 CPU 物理核心数)

## License

MIT License
