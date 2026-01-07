//
// Created by Hengrui Shang on 2025/12/29.
//

#ifndef ACOULAR_CWT_STFTPROCESSOR_H
#define ACOULAR_CWT_STFTPROCESSOR_H
#pragma once

#include "TimeFrequencyProcessor.h"
#include <vector>
#include <span>

class STFTProcessor : public TimeFrequencyProcessor
{
    private:
        int width; // STFT窗口宽度

        // 汉宁窗生成函数
        std::vector<float> createHanningWindow(int size) const;

    public:
        explicit STFTProcessor(int window_width);
        STFTProcessor();

        std::vector<Complex> computeSnapshot(
            std::span<const float> data,  // 输入原始信号为实数
            int num_channels,
            float fs,
            float target_freq,
            size_t target_idx,
            int smoothing_width
        ) override;
};

#endif //ACOULAR_CWT_STFTPROCESSOR_H
