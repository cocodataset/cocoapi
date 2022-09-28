from setuptools import setup

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

setup(name='pycocotools',
      packages=['pycocotools'],
      package_dir={'pycocotools': 'pycocotools'},
      install_requires=['setuptools>=18.0', 'matplotlib>=2.1.0'],
      version='2.0')
