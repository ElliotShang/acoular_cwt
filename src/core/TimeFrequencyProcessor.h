//
// Created by Hengrui Shang on 2025/12/29.
//

#ifndef ACOULAR_CWT_TIMEFREQUENCYPROCESSOR_H
#define ACOULAR_CWT_TIMEFREQUENCYPROCESSOR_H
#pragma once
#include <vector>
#include <complex>
#include <span>

using Complex = std::complex<float>;

class TimeFrequencyProcessor
{
    public:
        virtual ~TimeFrequencyProcessor() = default;

        virtual std::vector<Complex> computeSnapshot(
            std::span<const float> data, //  span 特性，无需采用引用传递而是值传递，输入原始信号为实数
            int num_channels,
            float fs,
            float target_freq,
            size_t target_idx,
            int smoothing_width = 0 // 新增平滑参数
        ) = 0;

};

#endif //ACOULAR_CWT_TIMEFREQUENCYPROCESSOR_H
