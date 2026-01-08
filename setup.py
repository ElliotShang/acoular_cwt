import os
import re
import sys
import platform
import subprocess
import shutil

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection or explicit inclusion of package data
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

        # --- Windows 特殊处理：复制 DLL ---
        if platform.system() == "Windows":
            # 尝试找到 fftw3f.dll 并复制到包目录
            # 假设 DLL 在 external/fCWT/libs 下
            dll_src = os.path.join(ext.sourcedir, 'external', 'fCWT', 'libs', 'fftw3f.dll')
            if os.path.exists(dll_src):
                print(f"Copying {dll_src} to {extdir}")
                shutil.copy(dll_src, extdir)
            else:
                print(f"Warning: fftw3f.dll not found at {dll_src}")

setup(
    name='acoular_cwt',
    version='0.1.0',
    author='Hengrui Shang',
    author_email='your.email@example.com',
    description='Acoular extension for CWT/STFT based CSM calculation',
    long_description='',
    packages=find_packages(),
    # 告诉 setuptools 包含非 python 文件 (如 .dll)
    package_data={
        'acoular_cwt': ['*.dll', '*.so', '*.pyd'],
    },
    ext_modules=[CMakeExtension('acoular_cwt._wcsm')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=[
        'numpy',
        'acoular',
        'traits',
    ],
)

