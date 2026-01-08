cmake_minimum_required(VERSION 3.18)
project(acoular_cwt VERSION 0.1.0 LANGUAGES CXX)

# =========================================================
# 1. 配置 Python 环境（支持多平台）
# =========================================================
# 自动检测并配置 Python 环境
if(WIN32)
    # Windows 环境（Visual Studio 编译）
    message(STATUS "Detected Windows environment")
    
    # 优先检查环境变量中的 CONDA_PREFIX（但必须是 Windows 路径）
    if(DEFINED ENV{CONDA_PREFIX})
        string(SUBSTRING "$ENV{CONDA_PREFIX}" 0 1 first_char)
        if(NOT first_char STREQUAL "/")
            # 是 Windows 路径，使用环境变量
            message(STATUS "Using Conda environment from CONDA_PREFIX: $ENV{CONDA_PREFIX}")
            list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
            set(Python_FIND_VIRTUALENV FIRST)
        else()
            message(WARNING "CONDA_PREFIX points to WSL/Linux path, ignoring in Windows build")
        endif()
    endif()
    
    # 如果环境变量不可用，使用固定路径
    if(NOT Python_EXECUTABLE)
        set(Python_ROOT_DIR "G:/Anaconda3/envs/acoustics")
        set(Python_EXECUTABLE "G:/Anaconda3/envs/acoustics/python.exe")
        message(STATUS "Using fallback Python path: ${Python_EXECUTABLE}")
    endif()
    
    set(Python_FIND_REGISTRY LAST)
    set(Python_FIND_STRATEGY LOCATION)
    
else()
    # Linux/WSL 环境
    message(STATUS "Detected Linux/WSL environment")
    
    # 优先使用 CONDA_PREFIX 环境变量
    if(DEFINED ENV{CONDA_PREFIX})
        message(STATUS "Using Conda environment: $ENV{CONDA_PREFIX}")
        list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
        set(Python_FIND_VIRTUALENV FIRST)
    else()
        # 如果没有激活 Conda 环境，使用固定路径
        set(Python_ROOT_DIR "/home/shanghr/miniforge3/envs/acoular_cwt")
        message(STATUS "Using fallback Python path: ${Python_ROOT_DIR}")
    endif()
endif()

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
    # Windows 下通用选项
    # /wd4244 - 忽略 float 到 int 的隐式转换警告
    add_compile_options(/arch:AVX2 /MP /DNOMINMAX /wd4244)
    
    # 根据构建类型设置优化选项
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        # Debug 模式使用默认的 /Od 和 /RTC1，不额外添加优化选项
        # 如果需要在 Debug 模式下也优化，需要移除 /RTC1
        # add_compile_options(/Od)
    endif()
else()
    add_compile_options(-march=native -ffast-math -fopenmp -fvisibility=hidden)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3)
    endif()
endif()

# 查找 pybind11
# 将 Python 环境路径加入 CMAKE_PREFIX_PATH，帮助查找 pybind11
if(WIN32)
    # Windows 环境下，pybind11 通常在 Library 子目录
    if(Python_ROOT_DIR)
        list(PREPEND CMAKE_PREFIX_PATH "${Python_ROOT_DIR}")
        list(PREPEND CMAKE_PREFIX_PATH "${Python_ROOT_DIR}/Library")
    endif()
else()
    # Linux/WSL 环境下，确保 Python 环境路径在搜索路径中
    if(DEFINED ENV{CONDA_PREFIX})
        list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    elseif(Python_ROOT_DIR)
        list(PREPEND CMAKE_PREFIX_PATH "${Python_ROOT_DIR}")
    endif()
endif()

find_package(pybind11 CONFIG REQUIRED)

# 查找 OpenMP 和 FFTW3
find_package(OpenMP REQUIRED)

if(MSVC)
    # 强制 OpenMP 目标使用 /openmp:experimental 而不是默认的 -openmp
    # 这解决了两个问题：
    # 1. 避免同时出现 /openmp:llvm 和 -openmp 导致的覆盖问题
    # 2. 启用 OpenMP SIMD 支持 (MSVC 要求 /openmp:experimental 才能支持 simd)
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES 
        INTERFACE_COMPILE_OPTIONS "/openmp:experimental"
    )
endif()

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
