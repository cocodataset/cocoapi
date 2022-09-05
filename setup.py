from setuptools import setup, find_packages

setup(name='fdetapi', version='1.0', packages=find_packages(include=["common.*", "LuaAPI.*", "PythonAPI.*"]))
