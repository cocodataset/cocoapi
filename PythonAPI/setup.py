from setuptools import dist, setup, Extension


install_requires=[
    'setuptools>=18.0',
    'cython>=0.27.3',
    'matplotlib>=2.1.0'
]

dist.Distribution().fetch_build_eggs(install_requires)


import numpy as np


# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

ext_modules = [
    Extension(
        'pycocotools._mask',
        sources=['../common/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs = [np.get_include(), '../common'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    )
]

setup(
    name='pycocotools',
    packages=['pycocotools'],
    package_dir={'pycocotools': 'pycocotools'},
    version='2.0',
    ext_modules=ext_modules
)
