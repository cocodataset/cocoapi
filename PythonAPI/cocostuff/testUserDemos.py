#!/usr/bin/python

__author__ = 'hcaesar'

# A simple test case that runs all COCO Stuff demos relevant to a user.
#
# Note: Some demos require user interaction (e.g. to close a figure).
#
# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
# Licensed under the Simplified BSD License [see coco/license.txt]

import cocoSegmentationToPngDemo
import pngToCocoResultDemo
import cocoStuffDemo
import cocoStuffEvalDemo

# Run demos
print('Running test...')
cocoSegmentationToPngDemo.cocoSegmentationToPngDemo()
pngToCocoResultDemo.pngToCocoResultDemo()
cocoStuffDemo.cocoStuffDemo()
cocoStuffEvalDemo.cocoStuffEvalDemo()
print('Test successfully finished!')
