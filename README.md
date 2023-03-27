COCO API - http://cocodataset.org/

COCO is a large image dataset designed for object detection, segmentation, person keypoints detection, stuff segmentation, and caption generation. This package provides Python APIs adopted from `https://github.com/cocodataset/cocoapi.git` that assists in loading, parsing, and visualizing the annotations in COCO. Please visit http://cocodataset.org/ for more information on COCO, including for the data, paper, and tutorials. The exact format of the annotations is also described on the COCO website. The Python APIs are complete, however some packages in numpy from 2020 are deprecated. I modified the Python APIs to be compatible with current numpy version.

In addition to this API, please download both the COCO images and annotations in order to run the demos and use the API. Both are available on the project website.
- Please download, unzip, and place the images in: coco/images/
- Please download and place the annotations in: `coco/annotations/` . The annotation can be downloaded with `wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip`
For substantially more details on the API please see http://cocodataset.org/#download.

After downloading the images and annotations, run the Python demos for example usage.

To install Python APIs:
```bash
pip3 install cpython
cd PythonAPI
make
```

