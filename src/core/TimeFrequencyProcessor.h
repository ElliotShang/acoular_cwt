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
            std::span<const Complex> data, //  span 特性，无需采用引用传递而是值传递
            int num_channels,
            float fs,
            float target_freq,
            size_t target_idx
        ) = 0;

};

#endif //ACOULAR_CWT_TIMEFREQUENCYPROCESSOR_H
