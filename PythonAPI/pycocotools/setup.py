"""
    setup.py file for SWIG example
"""
from distutils.core import setup, Extension
import numpy

polyiou_module = Extension('_polyiou',
                           sources=['polyiou_wrap.cxx', 'polyiou.cpp'],
                           )
setup(name = 'polyiou',
      version = '0.1',
      author = "SWIG Docs",
      description = """Simple swig example from docs""",
      ext_modules = [polyiou_module],
      py_modules = ["polyiou"],
)