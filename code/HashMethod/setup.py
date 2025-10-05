from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys
# 自动判断操作系统优化参数
compile_args = {
    'windows': ['/O2'] if sys.platform == 'win32' else ['-O2'],  # Windows的MSVC用/O2，MinGW用-O2
    'unix': ['-O3', '-march=native']  # Linux/macOS用O3和本地CPU优化
}
extra_compile_args = compile_args['windows'] if sys.platform.startswith('win') else compile_args['unix']

# 模块名
module_names = [
    "core_label_propagation",
    "core_second_cluster",
    "core_spectral_label_EBMD",
    "core_spectral_label"
]
extensions = [
    Extension(
        name=mod,
        sources=[mod + ".pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    ) for mod in module_names
]

setup(
    name="spectral_label",
    ext_modules=cythonize(extensions),
)

#  python setup.py build_ext --inplace