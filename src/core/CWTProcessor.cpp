//
// Created by Hengrui Shang on 2025/12/29.
//
#include "CWTProcessor.h"
#include "omp.h"
#include <algorithm>

CWTProcessor::CWTProcessor(const float sigma) 
    : wavelet(std::make_unique<Morlet>(sigma)),
      fcwt(std::make_unique<FCWT>(wavelet.get(), 1, true, false))
{
}

CWTProcessor::CWTProcessor() : CWTProcessor(6.0f)
{
}

std::vector<Complex> CWTProcessor::computeSnapshot(
    std::span<const float> data, //  span 特性，无需采用引用传递而是值传递，输入原始信号为实数
    int num_channels,
    float fs,
    float target_freq,
    size_t target_idx,
    int smoothing_width
    ) {

    const int K = 2*smoothing_width+1;
    std::vector<Complex> snapshot(num_channels*K);
    // 1. 计算尺度
    auto num_samples = data.size() / num_channels;

    Scales scales(wavelet.get(), FCWT_LINFREQS, fs, target_freq, target_freq, 1);
    #pragma omp parallel for
    for (auto ch=0; ch<num_channels; ch++){
        auto offset = ch * num_samples;
        std::span<const float> channel_data = data.subspan(offset, num_samples);
        
        // b. 输出缓冲区
        std::vector<Complex> channel_output(num_samples);
        
        // c. 计算CWT - fcwt可以直接接受float输入
        fcwt->cwt(const_cast<float*>(channel_data.data()), static_cast<int>(num_samples), channel_output.data(), &scales);
        
        // d. 提取 range: [target_idx - width, target_idx + width]
        // 假设输入数据data已经成了各个通道数据按不同行排列，不同列的排列为时间
        for (auto k=0; k<K; k++)
        {
            int time_offset = k-smoothing_width;
            int current_idx = std::clamp(static_cast<int>(target_idx)+time_offset, 0, static_cast<int>(num_samples)-1);
            snapshot[ch * K + k] = channel_output[current_idx];
        }
    }
    return snapshot;
}
