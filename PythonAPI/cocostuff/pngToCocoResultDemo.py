#!/usr/bin/python

__author__ = 'hcaesar'

# Converts a folder of .png images with segmentation results back
# to the COCO result format. 
#
# The .png images should be indexed images with or without a color
# palette for visualization.
#
# Note that this script only works with image names in COCO 2017
# format (000000000934.jpg). The older format
# (COCO_train2014_000000000934.jpg) is not supported.
#
# See cocoSegmentationToPngDemo.py for the reverse conversion.
#
# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
# Licensed under the Simplified BSD License [see coco/license.txt]

import os
import io
import json
import re
from pycocotools.cocostuffhelper import pngToCocoResult

def pngToCocoResultDemo(dataDir='../..', resType='examples', indent=None):
    '''
    Converts a folder of .png images with segmentation results back
    to the COCO result format. 
    :param dataDir: location of the COCO root folder
    :param resType: identifier of the result annotation file
    :param indent: number of whitespaces used for JSON indentation
    :return: None
    '''

    # Define paths
    pngFolder = '%s/results/segmentations/%s' % (dataDir, resType)
    jsonPath = '%s/results/stuff_%s_results.json' % (dataDir, resType)

    # Get images in png folder
    imgNames = os.listdir(pngFolder)
    imgNames = [imgName[:-4] for imgName in imgNames if imgName.endswith('.png')]
    imgNames.sort()
    imgCount = len(imgNames)

    # Init
    annCount = 0

    with io.open(jsonPath, 'w', encoding='utf8') as output:
        print('Writing results to: %s' % jsonPath)

        # Annotation start
        output.write(unicode('[\n'))

        for i, imgName in zip(xrange(0, imgCount), imgNames):
            print('Converting png image %d of %d: %s' % (i+1, imgCount, imgName))

            # Add stuff annotations
            pngPath = '%s/%s.png' % (pngFolder, imgName)
            tokens = imgName.split('_')
            if len(tokens) == 1:
                # COCO 2017 format
                imgId = int(imgName)
            elif len(tokens) == 3:
                # Previous COCO format
                imgId = int(tokens[2])
            else:
                raise Exception('Error: Invalid COCO file format!')

            anns = pngToCocoResult(pngPath, imgId)

            # Write JSON
            str_ = json.dumps(anns, indent=indent)
            str_ = str_[1:-1]
            if len(str_) > 0:
                output.write(unicode(str_))
                annCount = annCount + 1

            # Add comma separator
            if i < imgCount-1 and len(str_) > 0:
                output.write(unicode(','))

            # Add line break
            output.write(unicode('\n'))

        # Annotation end
        output.write(unicode(']'))

        # Create an error if there are no annotations
        if annCount == 0:
            raise Exception('The output file has 0 annotations and will not work with the COCO API!')

if __name__ == "__main__":
    pngToCocoResultDemo()
