from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

# To install and compile to your anaconda/python site-packages, simply run:
# $ pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
# Note that the original compile flags below are GCC flags unsupported by the Visual C++ 2015 build tools.
# They can safely be removed.

ext_modules = [
    Extension(
        'pycocotools._mask',
        sources=['../common/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs = [np.get_include(), '../common'],
        extra_compile_args=[] # originally was ['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    )
]

setup(name='pycocotools',
      packages=['pycocotools'],
      package_dir = {'pycocotools': 'pycocotools'},
      version='2.0',
      ext_modules=
          cythonize(ext_modules)
      )
