//
// Created by Hengrui Shang on 2025/12/29.
//

#ifndef ACOULAR_CWT_CWTPROCESSOR_H
#define ACOULAR_CWT_CWTPROCESSOR_H
#pragma once

#include "TimeFrequencyProcessor.h"
#include "fcwt.h"
#include <memory>
#include <vector>
#include <span>

class CWTProcessor : public TimeFrequencyProcessor
{
    private:
        std::unique_ptr<Wavelet> wavelet;
        std::unique_ptr<FCWT> fcwt;

    public:
        // 使用成员初始化列表
        explicit CWTProcessor(const float sigma);
        CWTProcessor();

        std::vector<Complex> computeSnapshot(
            std::span<const float> data, //  span 特性，无需采用引用传递而是值传递，输入原始信号为实数
            int num_channels,
            float fs,
            float target_freq,
            size_t target_idx,
            int smoothing_width
            ) override;
};

#endif //ACOULAR_CWT_CWTPROCESSOR_H
