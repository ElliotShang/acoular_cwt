#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <memory>
#include "core/TimeFrequencyProcessor.h"
#include "core/CWTProcessor.h"
#include "core/STFTProcessor.h"
#include "utils/MathUtils.h"

namespace py = pybind11;

enum class AlgoType {
    CWT,
    STFT
};

// 模板化计算Padding, 编译期剪枝
template <AlgoType Type>
constexpr size_t get_safe_padding(float fs, float freq, float param_val, int smoothing_width)
{
    size_t base_padding = 0;
    if constexpr (Type == AlgoType::STFT)
    {
        base_padding = static_cast<size_t>(param_val);
    }else if constexpr (Type == AlgoType::CWT)
    {
        constexpr float SAFETY_FACTOR = 10.0f;
        // 防止频率过低导致除零
        // param_val 对应 Morlet 小波的 sigma
        // COI 估算公式: K * (fs / f) * sigma
        float effective_freq = (freq < 1e-5f)? 1.0f : freq;
        base_padding = static_cast<size_t>((SAFETY_FACTOR * param_val * fs) / effective_freq);
    }
    return base_padding+smoothing_width;
}

// 计算CSM
template <AlgoType Type>
py::array_t<std::complex<float>> compute_csm_template(
    float* raw_ptr,
    size_t n_total_samples,
    size_t n_channels,
    float fs,
    float freq,
    size_t global_target_idx,
    int smoothing_width,
    float param_val)
{
    // 计算padding
    size_t padding = get_safe_padding<Type>(fs, freq, param_val, smoothing_width);
    // 切片计算范围
    long long start_idx = static_cast<long long>(global_target_idx) - static_cast<long long>(padding);
    long long end_idx = static_cast<long long>(global_target_idx) + static_cast<long long>(padding) + 1;
    long long safe_start = std::max(0LL, start_idx);
    long long safe_end = std::min((long long)n_total_samples, end_idx);  // 修复：应该使用min而不是max
    size_t chunk_len = safe_end - safe_start;
    size_t relative_idx = global_target_idx - safe_start;

    // 预先分配内存和输入数据转置，openmp并行
    std::vector<float> chunk_data(n_channels * chunk_len);
    #pragma omp parallel for
    for (int ch = 0; ch < static_cast<int>(n_channels); ++ch) {
        float* dst = &chunk_data[ch * chunk_len];
        for (size_t t = 0; t < chunk_len; ++t) {
            // Acoular layout: (Time, Channels) -> ptr[t * M + ch]
            size_t src_idx = (safe_start + t) * n_channels + ch;
            dst[t] = raw_ptr[src_idx];
        }
    }

    // 编译期策略实例化处理器
    std::unique_ptr<TimeFrequencyProcessor> processor;
    if constexpr (Type == AlgoType::CWT)
    {
        processor = std::make_unique<CWTProcessor>(param_val); // CWT时候parma_val是小波中的sigma
    }else
    {
        processor = std::make_unique<STFTProcessor>(param_val); // STFT时候param_val是STFT中的窗口宽度
    }

    // 执行计算
    int effective_smoothing = (Type == AlgoType::STFT)? 0 : smoothing_width;
    auto snapshots = processor->computeSnapshot(
        chunk_data, n_channels, fs, freq, relative_idx, effective_smoothing);

    // 计算CSM
    auto csm_vec = MathUtils::vectorToCSM<MathUtils::CsmMode::Full>(
        snapshots, n_channels, effective_smoothing);
    
    // 返回结果 - 创建一个3D numpy数组 (1, n_channels, n_channels)
    // 遵循Acoular标准：(num_freqs, num_channels, num_channels)
    auto result = py::array_t<std::complex<float>>(
        {static_cast<py::ssize_t>(1), 
         static_cast<py::ssize_t>(n_channels), 
         static_cast<py::ssize_t>(n_channels)}
    );
    
    // 拷贝数据到numpy数组
    auto buf = result.request();
    std::complex<float>* ptr = static_cast<std::complex<float>*>(buf.ptr);
    std::copy(csm_vec.begin(), csm_vec.end(), ptr);
    
    return result;
}

// 运行时候分发
py::array_t<std::complex<float>> compute_csm_dispatch(
    py::array_t<float> input_array,
    float fs,
    float target_freq,
    size_t global_target_idx,
    std::string method,
    int smoothing_width,
    float param_val)
{
    auto buf = input_array.request();
    if (buf.ndim!= 2) throw std::runtime_error("Input must be 2D array (Samples x Channels)");
    size_t n_samples = buf.shape[0];
    size_t n_channels = buf.shape[1];
    auto* raw_ptr = static_cast<float*>(buf.ptr);
    // 编译器模板分发
    if (method == "CWT")
    {
        return compute_csm_template<AlgoType::CWT> (
            raw_ptr,n_samples,n_channels,fs,target_freq,
            global_target_idx,smoothing_width,param_val);
    }else if(method == "STFT")
    {
        return compute_csm_template<AlgoType::STFT> (
            raw_ptr,n_samples,n_channels,fs,target_freq,
            global_target_idx,smoothing_width,param_val);
    }else
    {
        throw std::invalid_argument("Unknown method: " + method);
    }
}

// pybinding 接口定义
PYBIND11_MODULE(_wcsm, m)
{
    m.doc() = "Accelerated Transient CSM Engine with Compile-time Optimization";
    m.def("compute_csm", &compute_csm_dispatch,
        "Compute instantaneous CSM",
        py::arg("input_array"),
        py::arg("fs"),
        py::arg("target_freq"),
        py::arg("target_idx"),
        py::arg("method"),
        py::arg("smoothing_width") = 0,
        py::arg("param_val") = 6.0f
        );
}