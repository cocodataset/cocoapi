MS COCO API - http://mscoco.org/

Microsoft COCO is a large image dataset designed for object detection, segmentation, and caption generation. This package provides Matlab and Python APIs that assists in loading, parsing and visualizing the annotations in COCO. Please visit http://mscoco.org/ for more information on COCO, including for the data, paper, and tutorials. The exact format of the annotations is also described on the COCO website.

In addition to this API, please download both the COCO images and annotations in order to run the demos and use the API. Both are available on the project website.
-Please download, unzip, and place the images in: coco/images/
-Please download and place the annotations in: coco/annotations/

After downloading the images and annotations run either the Matlab or Python demos for example usage.

To install Python API, run "python setup.py build_ext --inplace" to compile Cython code.