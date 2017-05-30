import sys
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np
# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"
extra_compile_args = ['-Wno-cpp', '-Wno-unused-function', '-std=c99']\
  if sys.platform != 'win32' else []
ext_modules = [
    Extension(        
        'pycocotools._mask',
        language='c++',
        sources=['../common/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs = [np.get_include(), '../common'],
        extra_compile_args=extra_compile_args,
    )
]

setup(name='pycocotools',
      packages=['pycocotools'],
      package_dir = {'pycocotools': 'pycocotools'},
      version='2.0',
      ext_modules=
          cythonize(ext_modules)
      )