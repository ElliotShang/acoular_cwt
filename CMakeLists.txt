cmake_minimum_required(VERSION 3.18)
project(acoular_cwt VERSION 0.1.0 LANGUAGES CXX)

# =========================================================
# 1. 配置 Python 环境
# =========================================================
# 尝试自动探测激活的 Conda 环境
if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Detected active Conda environment: $ENV{CONDA_PREFIX}")
    # 将 Conda 环境路径加入 CMAKE_PREFIX_PATH，这样 find_package(pybind11) 也能找到
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    
    # 提示 FindPython 优先使用虚拟环境中的 Python
    set(Python_FIND_VIRTUALENV FIRST)
endif()

# 如果是在 IDE 中且未激活 Conda，可能需要手动指定 Python_EXECUTABLE
# set(Python_EXECUTABLE "/path/to/conda/python")

# 查找 Python 解释器和开发资源
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)

message(STATUS ">> Python 探测结果:")
message(STATUS "   - Executable: ${Python_EXECUTABLE}")
message(STATUS "   - Include Dir: ${Python_INCLUDE_DIRS}")
message(STATUS "   - Libraries: ${Python_LIBRARIES}")

# =========================================================
# 2. 编译器与依赖项配置
# =========================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /MP)
    # Windows 下可能需要指定 NOMINMAX 防止 min/max 宏冲突
    add_compile_options(/DNOMINMAX)
else()
    add_compile_options(-O3 -march=native -ffast-math -fopenmp -fvisibility=hidden)
endif()

# 查找 pybind11
find_package(pybind11 CONFIG REQUIRED)

# 查找 OpenMP 和 FFTW3
find_package(OpenMP REQUIRED)

# 配置 FFTW3
if(MSVC)
    # Windows 下 PkgConfig 可能不好用，如果你有 vcpkg 或手动安装的库，请在这里配置
    # 假设 external/fCWT/libs 下有库文件 (根据你的目录结构)
    set(FFTW3_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/external/fCWT/libs")
    # 注意：这里需要确认 .lib 文件名，通常是 libfftw3f-3.lib 或 fftw3f.lib
    set(FFTW3_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/external/fCWT/libs/fftw3f.lib") 
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW3 REQUIRED fftw3f)
endif()

# =========================================================
# 3. 构建目标 - fCWT 静态库
# =========================================================
set(FCWT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/fCWT/src/fcwt")
add_library(fcwt_lib STATIC "${FCWT_DIR}/fcwt.cpp")
target_include_directories(fcwt_lib PUBLIC "${FCWT_DIR}" ${FFTW3_INCLUDE_DIRS})
target_link_libraries(fcwt_lib PUBLIC OpenMP::OpenMP_CXX ${FFTW3_LIBRARIES})

# =========================================================
# 4. 构建核心模块 (acoular_cwt._wcsm)
# =========================================================
# 收集所有源文件
file(GLOB_RECURSE SOURCES
        "src/wcsm.cpp"
        "src/binding.cpp"
        "src/core/*.cpp"   # 包含 CWTProcessor.cpp, STFTProcessor.cpp
        "src/utils/*.cpp"  # 如果 utils 下有 cpp 文件也会包含
)

# Python 扩展模块
# 使用 pybind11_add_module 创建模块
pybind11_add_module(_wcsm ${SOURCES})

# 链接库：链接 fcwt_lib, OpenMP, 以及 FFTW3 (STFT模块需要)
target_link_libraries(_wcsm PRIVATE fcwt_lib OpenMP::OpenMP_CXX ${FFTW3_LIBRARIES})

# 包含目录
# 允许 #include "core/TimeFrequencyProcessor.h" (因为 core 在 src 下)
# 允许 #include "utils/MathUtils.h"
target_include_directories(_wcsm PRIVATE src)

# 安装
install(TARGETS _wcsm DESTINATION acoular_cwt)
