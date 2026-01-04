import numpy as np
import scipy.signal
import acoular
from traits.api import Float, Int, Property, cached_property, CArray
import fcwt

class MockWaveletSpectra(acoular.PowerSpectra):
    target_time = Float(0.0, desc="瞬时成像的时间点")
    target_freq = Float(1000.0, desc="目标频率(Hz)")

    csm = Property(depends_on=['target_time', 'target_freq','source.digest'])

    cached = False

    def _get_csm(self):
        fs = self.source.sample_freq
        num_channels = self.source.num_channels

        index_center = int(self.target_time*fs)
        radius = int(0.1*fs)
        start, stop = max(0, index_center-radius), index_center+radius

        # data shape: (samples, channels)
        data_block = self.source.data[start:stop, :]
        data_length = data_block.shape[0]

        # 目标时刻在data_block中的位置
        rel_idx = index_center - start

        # 存放各个通道在target_time附近一段时刻的小波系数用于平均
        # 平均窗口大小 (样本数)
        avg_window = 256 
        half_window = avg_window // 2
        
        # 确保窗口不超出数据范围
        idx_start = max(0, rel_idx - half_window)
        idx_end = min(data_length, rel_idx + half_window)
        actual_window_len = idx_end - idx_start
        
        if actual_window_len <= 0:
             # 极端情况处理
             return np.zeros((1, num_channels, num_channels), dtype=np.complex128)

        # 形状: (num_channels, actual_window_len)
        snapshots_block = np.zeros((num_channels, actual_window_len), dtype=np.complex64)

        # 3. 计算CWT
        morl = fcwt.Morlet(6.0) # 初始morlet小波系数
        FREQ_START = self.target_freq  # Hz
        FREQ_END = self.target_freq  # Hz
        FREQ_NUM = 1
        freqs = np.zeros(FREQ_NUM, dtype=np.float32)
        scales = fcwt.Scales(morl, fcwt.FCWT_LINSCALES, int(fs), FREQ_START, FREQ_END, FREQ_NUM)
        scales.getFrequencies(freqs)
        
        # FCWT对象
        nthreads = 12
        use_optimization_plan = False
        use_normalization = True
        cwt_obj = fcwt.FCWT(morl, nthreads, use_optimization_plan, use_normalization)
        
        for ch in range(num_channels):
            # 计算CWT系数
            output = np.zeros((FREQ_NUM, data_length), dtype=np.complex64)
            cwt_obj.cwt(data_block[:, ch].astype(np.float32), scales, output)
            
            # 提取时间窗口内的系数
            snapshots_block[ch, :] = output[0, idx_start:idx_end]

        # 4. 构建CSM (进行时间平均)
        # C = (X * X^H) / N
        csm_matrix = np.dot(snapshots_block, snapshots_block.T.conj()) / actual_window_len
        
        return csm_matrix[np.newaxis, :, :]

    def fftfreq(self):
        return np.array([self.target_freq], dtype=np.float32)






