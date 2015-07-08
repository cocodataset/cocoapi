from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

# To compile run "python setup.py build_ext --inplace"

setup(name='pycocotools',
      packages=['pycocotools'],
      package_dir = {'pycocotools': 'pycocotools'},
      version='1.0.2',
      include_dirs = [np.get_include(), '../MatlabAPI/private'],
      ext_modules=
        cythonize('pycocotools/_mask.pyx'),
      )