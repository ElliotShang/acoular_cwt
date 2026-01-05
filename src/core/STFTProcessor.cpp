//
// Created by Hengrui Shang on 2025/12/29.
//
#include "TimeFrequencyProcessor.h"
#include <cmath>
#include <complex>
#include "fftw3.h" 
#include <memory>
#include <numbers>
const float PI_VAL = std::numbers::pi_v<float>;
#include <vector>
#include <omp.h>

class STFTProcessor : public TimeFrequencyProcessor
{
    private:
        int width; // STFT窗口宽度

        // 汉宁窗生成函数
        std::vector<float> createHanningWindow(int size) const {
            std::vector<float> window(size);
            for (int i = 0; i < size; ++i) {
                // formula: 0.5 * (1 - cos(2 * pi * i / (N - 1)))
                window[i] = 0.5f * (1.0f - std::cos(2.0f * PI_VAL * i / (size - 1)));
            }
            return window;
        }

    public:
        explicit STFTProcessor(int window_width) : width(window_width) {}

        STFTProcessor() : STFTProcessor(1024) {}

        std::vector<Complex> computeSnapshot(
            std::span<const Complex> data,
            int num_channels,
            float fs,
            float target_freq,
            size_t target_idx
        ) override {
            std::vector<Complex> snapshot(num_channels);
            
            // 1. 基础参数检查
            if (width <= 0) return snapshot; 
            
            // 2. 预计算 Hanning 窗 (只读，线程安全)
            const auto window = createHanningWindow(width);
            
            // 3. 计算每个通道的样本数
            size_t num_samples_per_channel = data.size() / num_channels;

            // 4. 确定要处理的时间段
            // start_idx: 窗口在原始数据中的起始索引
            long long start_idx_long = static_cast<long long>(target_idx) - width / 2;
            
            #pragma omp parallel
            {
                // ---- 线程局部存储 (Thread Local Storage) ----
                // 每个线程分配自己的输入/输出 buffer 和 plan，避免竞争
                
                std::vector<std::complex<float>> thread_in(width);
                std::vector<std::complex<float>> thread_out(width);
                
                // 创建 plan (FFTW plan 创建需加锁)
                fftwf_plan plan = nullptr;
                
                #pragma omp critical
                {
                    plan = fftwf_plan_dft_1d(width, 
                                             reinterpret_cast<fftwf_complex*>(thread_in.data()), 
                                             reinterpret_cast<fftwf_complex*>(thread_out.data()), 
                                             FFTW_FORWARD, FFTW_ESTIMATE);
                }

                #pragma omp for
                for (int ch = 0; ch < num_channels; ++ch) {
                    // a. 获取当前通道的数据视图
                    size_t ch_offset = ch * num_samples_per_channel;
                    
                    // b. 填充输入缓冲区 (加窗)
                    for (int i = 0; i < width; ++i) {
                        long long current_sample_idx = start_idx_long + i;
                        
                        // 边界处理：Zero Padding
                        if (current_sample_idx >= 0 && current_sample_idx < static_cast<long long>(num_samples_per_channel)) {
                            // 读取数据并乘窗函数
                            Complex val = data[ch_offset + current_sample_idx];
                            thread_in[i] = val * window[i];
                        } else {
                            thread_in[i] = 0.0f;
                        }
                    }
                    
                    // c. 执行 FFT
                    fftwf_execute(plan);
                    
                    // d. 提取目标频率的系数
                    // bin_idx = target_freq * width / fs
                    int bin_idx = static_cast<int>(std::round(target_freq * width / fs));
                    
                    if (bin_idx >= 0 && bin_idx < width) {
                        snapshot[ch] = thread_out[bin_idx];
                    } else {
                        snapshot[ch] = 0.0f;
                    }
                }
                
                // 销毁线程局部的 plan (FFTW destroy 需加锁，虽然部分版本线程安全，但保险起见)
                #pragma omp critical
                {
                    fftwf_destroy_plan(plan);
                }
            } // end parallel

            return snapshot;
        }
};
