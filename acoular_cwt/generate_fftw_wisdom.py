"""
生成 FFTW Wisdom 优化文件

这个脚本会为不同大小的 FFT 生成预计算的优化方案（wisdom 文件）。
生成的 .wis 文件将放在当前目录，可以加速后续的 CWT 计算。
"""

import sys
import os
from pathlib import Path
import fcwt

def generate_wisdom(max_size=8192, threads=1):
    """
    生成 FFTW wisdom 文件
    
    Parameters
    ----------
    max_size : int
        最大 FFT 大小（必须是 2 的幂次）
    threads : int
        线程数
    """
    print(f"开始生成 FFTW Wisdom 文件...")
    print(f"  最大 FFT 大小: {max_size}")
    print(f"  线程数: {threads}")
    print(f"  输出目录: {os.getcwd()}")
    print()
    
    # 创建一个简单的 Morlet 小波（只是为了初始化 FCWT 对象）
    wavelet = fcwt.Morlet(6.0)
    
    # 创建 FCWT 对象
    # 参数: (wavelet, threads, use_fft_planning, use_optimization_schemes)
    fcwt_obj = fcwt.FCWT(wavelet, threads, True, True)
    
    # 生成优化方案
    # 参数: (max_size, optimization_flags)
    # optimization_flags: "ESTIMATE", "MEASURE", "PATIENT", "EXHAUSTIVE"
    # "MEASURE" 是合理的选择，平衡了生成时间和优化质量
    print("正在生成优化方案（这可能需要几分钟）...")
    print("使用 FFTW_MEASURE 模式进行优化...\n")
    
    try:
        fcwt_obj.create_FFT_optimization_plan(max_size, "FFTW_MEASURE")
        print("\n✓ Wisdom 文件生成成功！")
        print(f"\n生成的文件:")
        # 列出生成的 .wis 文件
        for wis_file in Path('.').glob('n*_t*.wis'):
            file_size = wis_file.stat().st_size
            print(f"  - {wis_file.name} ({file_size} bytes)")
    except Exception as e:
        print(f"\n✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="生成 FFTW Wisdom 优化文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python generate_fftw_wisdom.py                 # 使用默认设置 (max_size=8192, threads=1)
  python generate_fftw_wisdom.py --max-size 16384 --threads 4
        """
    )
    parser.add_argument(
        '--max-size', 
        type=int, 
        default=8192,
        help='最大 FFT 大小（必须是 2 的幂次，默认: 8192）'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=1,
        help='线程数（默认: 1，应与运行时的 OMP_NUM_THREADS 一致）'
    )
    
    args = parser.parse_args()
    
    # 验证 max_size 是 2 的幂次
    if args.max_size & (args.max_size - 1) != 0 or args.max_size < 2048:
        print(f"错误: max_size 必须是 2 的幂次且 >= 2048，当前值: {args.max_size}")
        sys.exit(1)
    
    success = generate_wisdom(args.max_size, args.threads)
    sys.exit(0 if success else 1)

