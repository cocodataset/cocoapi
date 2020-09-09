"""LVIS (pronounced ‘el-vis’): is a new dataset for Large Vocabulary Instance
Segmentation. We collect over 2 million high-quality instance segmentation
masks for over 1200 entry-level object categories in 164k images. LVIS API
enables reading and interacting with annotation files, visualizing annotations,
and evaluating results.

"""
import os.path
import sys

import setuptools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lvis'))

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, 'requirements.txt')) as f:
    reqs = f.read()

DISTNAME = 'mmlvis'
DESCRIPTION = 'Python API for LVIS dataset.'
AUTHOR = 'Agrim Gupta'
REQUIREMENTS = (reqs.strip().split('\n'), )
DOCLINES = (__doc__ or '')

if __name__ == '__main__':
    setuptools.setup(name=DISTNAME,
                     install_requires=REQUIREMENTS,
                     packages=setuptools.find_packages(),
                     version='10.5.3',
                     description=DESCRIPTION,
                     long_description=DOCLINES,
                     long_description_content_type='text/markdown',
                     author=AUTHOR)
