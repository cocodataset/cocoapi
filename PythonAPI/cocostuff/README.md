# COCO 2017 Stuff Segmentation Challenge API
On this page we present all files that are relevant to the COCO 2017 Stuff Segmentation Challenge.
Note that we currently only support Python for this challenge, but that you can find various scripts below to convert annotations from COCO ground-truth (GT) format to .png format, as well as from .png format back to COCO result format.

The following is an overview of all the demo files (in *PythonAPI/cocostuff*):
- **pycocostuffDemo.py**: A preview script that shows how to use the COCO API. It lists the categories, super-categories and shows the annotations of an example image.
- **pycocostuffEvalDemo.py**: Shows how to use the main evaluation script of the Stuff Segmentation Challenge.

Below are essential scripts used to convert between different file formats (in *PythonAPI/cocostuff*):
- **cocoXToPngDemo.py**: Converts either COCO GT or COCO result .json files to one .png file per image.
- **pngToCocoResultDemo.py**: Converts a folder of .png images with segmentation results back to the COCO result format. 
- **internalToCocoGTDemo.py**: Converts our internal .mat representation of the ground-truth annotations to COCO format.

Internally these scripts make use of (in *PythonAPI/pycocotools*):
- **cocostuffeval.py**: Internal functions for evaluating stuff segmentations against a ground-truth.
- **cocostuffhelper.py**: Helper functions used to convert between different formats for the COCO Stuff Segmentation Challenge.
- As well as general parts of the COCO and Mask API.
