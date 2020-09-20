import numpy as np
from setuptools import Extension, setup

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run
# "python setup.py build_ext install"
# Note that the original compile flags below are GCC flags unsupported by
# the Visual C++ 2015 build tools.
# They can safely be removed.

ext_modules = [
    Extension(
        'pycocotools._mask',
        sources=['common/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs=[np.get_include(), 'common'],
        # extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
        extra_compile_args=[],
    )
]

setup(name='mmpycocotools',
      packages=['pycocotools'],
      package_dir={'pycocotools': 'pycocotools'},
      install_requires=[
          'setuptools>=18.0', 'cython>=0.27.3', 'matplotlib>=2.1.0'
      ],
      version='12.0.3',
      ext_modules=ext_modules)
