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
constexpr size_t get_safe_padding()
{
    
}