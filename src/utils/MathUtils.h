//
// Created by Hengrui Shang on 2025/12/29.
//

#ifndef ACOULAR_CWT_MATHUTILS_H
#define ACOULAR_CWT_MATHUTILS_H
#include <complex>
#include <vector>
#include <span>
#include <iostream>
#include <omp.h>
#include <immintrin.h>

namespace MathUtils
{
    enum class CsmMode
    {
        Full,
        RemoveDiagonal
    };

    //AVX 加速的复数共轭点积, float单精度版本
    inline std::complex<float> complex_conj_dot_product_avx(
        const std::complex<float>* v_i,
        const std::complex<float>* v_j,
        const size_t K
    ){
        __m256 sum_real_vec = _mm256_setzero_ps();  // 8个float的实部累加器
        __m256 sum_imag_vec = _mm256_setzero_ps();  // 8个float的虚部累加器
        size_t k = 0;
        for (; k +3 < K; k += 4){
            // 处理四个复数
            __m256 vi = _mm256_loadu_ps(reinterpret_cast<const float*>(&v_i[k]));
            __m256 vj = _mm256_loadu_ps(reinterpret_cast<const float*>(&v_j[k]));
            // 实部和虚部分离
            __m256 vi_unpacked_low  = _mm256_shuffle_ps(vi, vi, 0xA0); // [r0,r0,r1,r1,r2,r2,r3,r3]
            __m256 vi_unpacked_high = _mm256_shuffle_ps(vi, vi, 0xF5); // [i0,i0,i1,i1,i2,i2,i3,i3]
            __m256 vj_unpacked_low  = _mm256_shuffle_ps(vj, vj, 0xA0);
            __m256 vj_unpacked_high = _mm256_shuffle_ps(vj, vj, 0xF5);
            // （a+bi）*conj(c+di) = (ac+bd) + (bc-ad)i
            __m256 ac = _mm256_mul_ps(vi_unpacked_low, vj_unpacked_low);
            __m256 bd = _mm256_mul_ps(vi_unpacked_high, vj_unpacked_high);
            __m256 bc = _mm256_mul_ps(vi_unpacked_high, vj_unpacked_low);
            __m256 ad = _mm256_mul_ps(vi_unpacked_low, vj_unpacked_high);

            sum_real_vec = _mm256_add_ps(sum_real_vec, _mm256_add_ps(ac, bd));
            sum_imag_vec = _mm256_add_ps(sum_imag_vec, _mm256_sub_ps(bc, ad));
        }
        float sum_real_arr[8], sum_imag_arr[8];
        _mm256_storeu_ps(sum_real_arr, sum_real_vec);
        _mm256_storeu_ps(sum_imag_arr, sum_imag_vec);
        float sum_real = 0.0f, sum_imag = 0.0f;
        for (int i = 0; i < 8; ++i) {
            sum_real += sum_real_arr[i];
            sum_imag += sum_imag_arr[i];
        }
        for (; k < K; ++k){
            float a = v_i[k].real();
            float b = v_i[k].imag();
            float c = v_j[k].real();
            float d = v_j[k].imag();
            sum_real += a * c + b * d;
            sum_imag += b * c - a * d;
        }
        return std::complex<float>(sum_real, sum_imag);
    }

    template<CsmMode Mode = CsmMode::Full, typename T>
    void computeCSM(
        std::span<const std::complex<T>> snapshot,
        std::span<std::complex<T>> output_csm,
        size_t n_channels,
        int smoothing_width)
    {
        // security check
        const int K = 2 * smoothing_width+1;
        T norm_factor = 1.0/static_cast<T>(K);
        if (snapshot.size() != n_channels*K || output_csm.size() != n_channels*n_channels)
        {
            std::cerr << "Snapshot size mismatch!" << std::endl;
            return;
        }
        #pragma omp parallel for schedule(dynamic) if(n_channels > 32)
        for (int i = 0; i < static_cast<int>(n_channels); ++i)
        {
            auto v_i = snapshot.subspan(i*K, K);
            // 复共轭矩阵的特性，j不从0开始
            for (int j = i; j < static_cast<int>(n_channels); ++j)
            {
                auto v_j = snapshot.subspan(j*K, K);
                // 接下来计算小向量的共轭点积
                std::complex<T> sum = {0.0, 0.0};

                T sum_real = 0.0;
                T sum_imag = 0.0;
                #pragma omp simd reduction(+:sum_real,sum_imag)
                for (size_t k = 0; k < K; ++k) // 此处可以SIMD向量化并行处理
                {
                    T a = v_i[k].real();
                    T b = v_i[k].imag();
                    T c = v_j[k].real();
                    T d = v_j[k].imag();
                    sum_real += a * c + b * d;
                    sum_imag += b * c - a * d;
                }
                sum = std::complex<T>(sum_real * norm_factor, sum_imag * norm_factor);
                // 基于Hermit矩阵的对称性进行填充
                size_t idx_upper = static_cast<size_t>(i) * n_channels + static_cast<size_t>(j); // 上三角索引
                size_t idx_lower = static_cast<size_t>(j) * n_channels + static_cast<size_t>(i); // 下三角索引
                if (i == j)
                {
                    if constexpr (Mode == CsmMode::RemoveDiagonal)
                        output_csm[idx_upper] = {0.0, 0.0};
                    else
                        output_csm[idx_upper] = sum;
                }else
                {
                    output_csm[idx_upper] = sum;
                    output_csm[idx_lower] = std::conj(sum);
                }
            }
        }
    }
    // std::vector ——> CSM
    template <CsmMode Mode = CsmMode::Full, typename T>
    std::vector<std::complex<T>> vectorToCSM(const std::vector<std::complex<T>>& vec,
        size_t n_channels,
        int smoothing_width)
    {
        size_t K = 2 * smoothing_width+1;
        // 2. 维度安全检查,确保输入数据的大小完全等于 M * K
        if (vec.size()!= n_channels * K) {
            std::cerr << "[MathUtils] Error: Input dimension mismatch!" << std::endl
                      << "  Expected: " << n_channels * K
                      << " (Channels=" << n_channels << ", K=" << K << ")" << std::endl
                      << "  Actual:   " << vec.size() << std::endl;
            // 出错返回空向量，避免崩溃
            return {};
        }
        // 预先分配CSM矩阵内存
        std::vector<std::complex<T>> csm(n_channels * n_channels);
        // 开始计算内核
        computeCSM(
            std::span{vec},
            std::span{csm},
            n_channels,
            smoothing_width);
        return csm;
    }
}
#endif //ACOULAR_CWT_MATHUTILS_H