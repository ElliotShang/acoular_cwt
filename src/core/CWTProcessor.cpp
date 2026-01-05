//
// Created by Hengrui Shang on 2025/12/29.
//
#include "TimeFrequencyProcessor.h"
#include "omp.h"
#include "fcwt.h"
#include <memory>

class CWTProcessor : public TimeFrequencyProcessor
{
    private:
        std::unique_ptr<Wavelet> wavelet;
        std::unique_ptr<FCWT> fcwt;

    public:
        // 使用成员初始化列表
        explicit CWTProcessor(const float sigma) 
            : wavelet(std::make_unique<Morlet>(sigma)),
              fcwt(std::make_unique<FCWT>(wavelet.get(), 1, true, false))
        {
        }

        CWTProcessor() : CWTProcessor(6.0f)
        {
        }

        std::vector<Complex> computeSnapshot(
            std::span<const Complex> data, //  span 特性，无需采用引用传递而是值传递
            int num_channels,
            float fs,
            float target_freq,
            size_t target_idx
            ) override {
            std::vector<Complex> snapshot(num_channels);
            // 1. 计算尺度
            auto num_samples = data.size() / num_channels;

            Scales scales(wavelet.get(), FCWT_LINFREQS, fs, target_freq, target_freq, 1);
            #pragma omp parallel for
            for (auto ch=0; ch<num_channels; ch++){
                // 假设输入数据data已经成了各个通道数据按不同行排列，不同列的排列为时间
                auto offset = ch * num_samples;
                std::span<const Complex> channel_data = data.subspan(offset, num_samples);
                // b. 输出缓冲区
                std::vector<Complex> channel_output(num_samples);
                // c. 计算CWT
                fcwt->cwt(const_cast<Complex*>(channel_data.data()), static_cast<int>(num_samples), channel_output.data(), &scales);
                // d. 提取目标频率的系数
                if (target_idx < num_samples) {
                    snapshot[ch] = channel_output[target_idx];
                } else {
                    snapshot[ch] = Complex(0,0);
                }
            }
            return snapshot;
        }
};
