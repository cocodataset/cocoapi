# COCO Stuff Segmentation Challenge API
The COCO Stuff API extends the [COCO API](https://github.com/cocodataset/cocoapi) with additional tools for stuff segmentation. 
On this page we present all files that are relevant to the COCO Stuff API. 
The API currently fully supports Matlab and Python. 
For use with other languages we also provide various scripts below to convert annotations from COCO ground-truth (GT) format to .png format, 
as well as from .png format back to COCO result format.

## COCO API notes
In addition to this API, please download both the COCO images and annotations in order to run the demos and use the API. Both are available on the project website.
- Please download, unzip, and place the images in: coco/images/
- Please download and place the annotations in: coco/annotations/
For substantially more details on the API please see [http://cocodataset.org](http://cocodataset.org).

After downloading the images and annotations, run the Matlab or Python demos for example usage.

To install:
- For Matlab, add coco/MatlabApi to the Matlab path (OSX/Linux binaries provided)
- For Python, run "make" under coco/PythonAPI

## Stuff API files
The following is an overview of all the demo files (in *<Language>API/cocostuff*):
- **cocoStuffDemo**: A preview script that shows how to use the COCO API. It lists the categories, super-categories and shows the annotations of an example image.
- **cocoStuffEvalDemo**: Shows how to use the main evaluation script of the Stuff Segmentation Challenge.

Below are essential scripts used to convert between different file formats (in *<Language>API/cocostuff*):
- **cocoSegmentationToPngDemo**: Converts COCO segmentation .json files (GT or results) to one .png file per image.
- **pngToCocoResultDemo**: Converts a folder of .png images with segmentation results back to the COCO result format. 

Internally these scripts make use of (in *PythonAPI/pycocotools* and *MatlabAPI/cocostuff*):
- **cocoStuffEval**: Internal functions for evaluating stuff segmentations against a ground-truth.
- **cocoStuffHelper**: Helper functions used to convert between different formats for the COCO Stuff Segmentation Challenge.
