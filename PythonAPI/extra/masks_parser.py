
"""masks_parser.py
This program performs a .JSON segmentation masks parsing and applies conversion formats: RLE <-> Polygon.
Note: Conversion only applied to COCO-format annotations "iscrowd==1"
"""
__author__ = "afigueres@fluendo.com"

import encodings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import requests
import cv2
import json
from itertools import groupby
import sys
from pathlib import Path
import os
# Get current directory

cur_dir = os.getcwd()
sys.path.append(os.path.join(cur_dir, "../pycocotools"))  # To find local version

from pycocotools.coco import COCO
import pycocotools._mask as _mask

## Global defs       
def RLE2poly(file_path):
    """
    Converts binary mask format to RLE and applies this conversion on same input file
    :param file_path: COCO annotation path directory (dataset)
    :return: Null (returns overwritten file)
    """
    try:
        coco_annotation = COCO(annotation_file=file_path)

        for x in range(0,len(coco_annotation.dataset['annotations']),1):
            # Conversion only applicable for COCO's "iscrowd" annotations
            if(int(coco_annotation.dataset['annotations'][x]['iscrowd']) == 1):
                RLE_mask = coco_annotation.annToMask(coco_annotation.dataset['annotations'][x])
                list_test = []
                polygon_decoded= polygonFromMask(RLE_mask)
                list_test.append(polygon_decoded)
                # Rewrite dataset
                with open(file_path, "r",encoding='utf-8') as jsonFile:
                    data = json.load(jsonFile)
                    data['annotations'][x]['segmentation'] = list_test
                
                with open(file_path, "w",encoding='utf-8') as jsonFile:
                    json.dump(data, jsonFile)

        print("Conversion DONE -> RLE2poly")
    except:
        print("Error: Incorrect RLE2poly conversion. System aborted")
        sys.exit(0)

def poly2RLE(file_path):
    """
    Converts polygon object format to RLE and applies this conversion on same input file
    :param file_path: COCO annotation path directory (dataset)
    :return: Null (returns overwritten file)
    """
    try:
        coco_annotation = COCO(annotation_file=file_path)
        
        for x in range(0,len(coco_annotation.dataset['annotations']),1):
            # Conversion only applicable for COCO's "iscrowd" annotations
            if(int(coco_annotation.dataset['annotations'][x]['iscrowd']) == 1):
                RLE_candidate = coco_annotation.annToMask(coco_annotation.dataset['annotations'][x])
                list_test = []
                RLE_encoded= binary_mask_to_rle(RLE_candidate)
                list_test.append(RLE_encoded)
                #annotation = coco_annotation.dataset['annotations'][x]['segmentation'][0] = RLE_encoded

                # Rewrite dataset
                with open(file_path, "r",encoding='utf-8') as jsonFile:
                    data = json.load(jsonFile)
                    data['annotations'][x]['segmentation'] = RLE_encoded
                
                with open(file_path, "w",encoding='utf-8') as jsonFile:
                    json.dump(data, jsonFile)
        
        print("Conversion DONE -> poly2RLE")
    except:
        print("Error: Incorrect poly2RLE conversion. System aborted")
        sys.exit(0)

def polygonFromMask(maskedArr): 
    """
    Converts binary mask format to RLE
    Source: https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    :param maskedArr: binary mask
    :return: polygon mask  ([x, y, w, h], area )
    """
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    RLEs = _mask.frPyObjects(segmentation, maskedArr.shape[0], maskedArr.shape[1])
    RLE = _mask.merge(RLEs)
    # RLE = mask.encode(np.asfortranarray(maskedArr))
    area = _mask.area(RLEs)
    [x, y, w, h] = cv2.boundingRect(maskedArr)

    return segmentation[0] 


def binary_mask_to_rle(binary_mask):
    """
    Converts binary mask format to RLE
    Source: https://stackoverflow.com/a/49547872
    :param binary_mask: binary mask
    :return: RLE mask 
    """
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle




if __name__ == "__main__":
    # Input dataset path
    DATASET_PATH = './samples/instances_val2017_1img-sample.json'

    # SAMPLE #1: Convert RLE to Polygon
    RLE2poly(DATASET_PATH)

    # SAMPLE #2: Convert Polygon to RLE
    poly2RLE(DATASET_PATH)