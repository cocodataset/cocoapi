#!/usr/bin/python

__author__ = 'hcaesar'

# Converts our internal .mat representation of the ground-truth
# annotations to COCO format.
#
# The resulting annotation files can be downloaded from: 
# http://cocodataset.org/dataset.htm#download
#
# The script has no immediate use to the user, as we do not provide
# the .mat GT files. However it does help to better understand the
# COCO annotation format.
#
# Note: This script only supports Matlab files <= v7.0.
#
# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
# Licensed under the Simplified BSD License [see coco/license.txt]

from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocostuffhelper import cocoSegmentationToSegmentationMap, segmentationToCocoMask
import numpy as np
import scipy.io # To open matlab <= v7.0 files
import io
import os
import json
import time

def internalToCocoGTDemo(dataType='train2017', dataDir='../..',
        imgCount=float('inf'), stuffStartId=92, stuffEndId=182, mergeThings=True, 
        indent=None, includeCrowd=False, outputAnnots=True):
    '''
    Converts our internal .mat representation of the ground-truth annotations to COCO format.
    :param dataType: the name of the subset: train201x, val201x, test-dev201x or test201x
    :param dataDir: location of the COCO root folder
    :param imgCount: the number of images to use for the .json file
    :param stuffStartId: id where stuff classes start
    :param stuffEndId: id where stuff classes end
    :param mergeThings: merges all 91 thing classes into a single class 'other' with id 183
    :param indent: number of whitespaces used for JSON indentation
    :param includeCrowd: whether to include 'crowd' thing annotations as 'other' (or void)
    :param outputAnnots: whether to include annotations (for test images we only release ids)
    :return: None
    '''

    # Define paths
    imgCountStr = ('_%d' % imgCount) if imgCount < float('inf') else ''
    annotFolder = '%s/annotations/internal/%s' % (dataDir, dataType)
    if dataType == 'test-dev5k2017':
        dataTypeInst =  'test-dev2017'
    else:
        dataTypeInst = dataType
    annPath = '%s/annotations/instances_%s.json' % (dataDir, dataTypeInst)
    if outputAnnots:
        jsonPath = '%s/annotations/stuff_%s%s.json' % (dataDir, dataType, imgCountStr)
    else:
        jsonPath = '%s/annotations/stuff_image_info_%s%s.json' % (dataDir, dataType, imgCountStr)

    # Check if output file already exists
    if os.path.exists(jsonPath):
        raise Exception('Error: Output file already exists: %s' % jsonPath)

    # Check if input folder exists
    if not os.path.exists(annotFolder):
        raise Exception('Error: Input folder does not exist: %s' % annotFolder)

    # Get images
    imgNames = os.listdir(annotFolder)
    imgNames = [imgName[:-4] for imgName in imgNames if imgName.endswith('.mat')]
    imgNames.sort()
    if imgCount < len(imgNames):
        imgNames = imgNames[0:imgCount]
    imgCount = len(imgNames)
    imgIds = [int(imgName) for imgName in imgNames]

    # Load COCO API for things
    cocoGt = COCO(annPath)

    # Init
    # annId must be unique, >=1 and cannot overlap with the detection annotations
    if dataType == 'train2017':
        annIdStart = int(1e7)
    elif dataType == 'val2017':
        annIdStart = int(2e7)
    elif dataType == 'test-dev2017':
        annIdStart = int(3e7)
    elif dataType == 'test-challenge2017':
        annIdStart = int(4e7)
    elif dataType == 'test-dev5k2017': # Redundant test set annotations
        annIdStart = int(5e7)
    else:
        raise Exception('Error: Unknown dataType %s specified!' % dataType)
    annId = annIdStart
    startTime = time.clock()

    print("Writing JSON metadata...")
    with io.open(jsonPath, 'w', encoding='utf8') as output:
        # Write info
        infodata = {'description': 'COCO 2017 Stuff Dataset',
                    'url': 'http://cocodataset.org',
                    'version': '1.0',
                    'year': 2017,
                    'contributor': 'H. Caesar, J. Uijlings, M. Maire, T.-Y. Lin, P. Dollar and V. Ferrari',
                    'date_created': '2017-08-31 00:00:00.0'}
        infodata = {'info': infodata}
        infoStr = json.dumps(infodata, indent=indent)
        infoStr = infoStr[1:-1] + ',\n'  # Remove brackets and add comma

        # Write images
        imdata = [i for i in cocoGt.dataset['images'] if i['id'] in imgIds]
        imdata = {'images': imdata}
        imStr = json.dumps(imdata, indent=indent)
        imStr = imStr[1:-1] + ',\n'  # Remove brackets and add comma

        # Write licenses
        licdata = {'licenses': cocoGt.dataset['licenses']}
        licStr = json.dumps(licdata, indent=indent)
        licStr = licStr[1:-1] + ',\n'  # Remove brackets and add comma

        # Write categories
        catdata = []
        catdata.extend([
            {'id': 92, 'name': 'banner', 'supercategory': 'textile'},
            {'id': 93, 'name': 'blanket', 'supercategory': 'textile'},
            {'id': 94, 'name': 'branch', 'supercategory': 'plant'},
            {'id': 95, 'name': 'bridge', 'supercategory': 'building'},
            {'id': 96, 'name': 'building-other', 'supercategory': 'building'},
            {'id': 97, 'name': 'bush', 'supercategory': 'plant'},
            {'id': 98, 'name': 'cabinet', 'supercategory': 'furniture-stuff'},
            {'id': 99, 'name': 'cage', 'supercategory': 'structural'},
            {'id': 100, 'name': 'cardboard', 'supercategory': 'raw-material'},
            {'id': 101, 'name': 'carpet', 'supercategory': 'floor'},
            {'id': 102, 'name': 'ceiling-other', 'supercategory': 'ceiling'},
            {'id': 103, 'name': 'ceiling-tile', 'supercategory': 'ceiling'},
            {'id': 104, 'name': 'cloth', 'supercategory': 'textile'},
            {'id': 105, 'name': 'clothes', 'supercategory': 'textile'},
            {'id': 106, 'name': 'clouds', 'supercategory': 'sky'},
            {'id': 107, 'name': 'counter', 'supercategory': 'furniture-stuff'},
            {'id': 108, 'name': 'cupboard', 'supercategory': 'furniture-stuff'},
            {'id': 109, 'name': 'curtain', 'supercategory': 'textile'},
            {'id': 110, 'name': 'desk-stuff', 'supercategory': 'furniture-stuff'},
            {'id': 111, 'name': 'dirt', 'supercategory': 'ground'},
            {'id': 112, 'name': 'door-stuff', 'supercategory': 'furniture-stuff'},
            {'id': 113, 'name': 'fence', 'supercategory': 'structural'},
            {'id': 114, 'name': 'floor-marble', 'supercategory': 'floor'},
            {'id': 115, 'name': 'floor-other', 'supercategory': 'floor'},
            {'id': 116, 'name': 'floor-stone', 'supercategory': 'floor'},
            {'id': 117, 'name': 'floor-tile', 'supercategory': 'floor'},
            {'id': 118, 'name': 'floor-wood', 'supercategory': 'floor'},
            {'id': 119, 'name': 'flower', 'supercategory': 'plant'},
            {'id': 120, 'name': 'fog', 'supercategory': 'water'},
            {'id': 121, 'name': 'food-other', 'supercategory': 'food-stuff'},
            {'id': 122, 'name': 'fruit', 'supercategory': 'food-stuff'},
            {'id': 123, 'name': 'furniture-other', 'supercategory': 'furniture-stuff'},
            {'id': 124, 'name': 'grass', 'supercategory': 'plant'},
            {'id': 125, 'name': 'gravel', 'supercategory': 'ground'},
            {'id': 126, 'name': 'ground-other', 'supercategory': 'ground'},
            {'id': 127, 'name': 'hill', 'supercategory': 'solid'},
            {'id': 128, 'name': 'house', 'supercategory': 'building'},
            {'id': 129, 'name': 'leaves', 'supercategory': 'plant'},
            {'id': 130, 'name': 'light', 'supercategory': 'furniture-stuff'},
            {'id': 131, 'name': 'mat', 'supercategory': 'textile'},
            {'id': 132, 'name': 'metal', 'supercategory': 'raw-material'},
            {'id': 133, 'name': 'mirror-stuff', 'supercategory': 'furniture-stuff'},
            {'id': 134, 'name': 'moss', 'supercategory': 'plant'},
            {'id': 135, 'name': 'mountain', 'supercategory': 'solid'},
            {'id': 136, 'name': 'mud', 'supercategory': 'ground'},
            {'id': 137, 'name': 'napkin', 'supercategory': 'textile'},
            {'id': 138, 'name': 'net', 'supercategory': 'structural'},
            {'id': 139, 'name': 'paper', 'supercategory': 'raw-material'},
            {'id': 140, 'name': 'pavement', 'supercategory': 'ground'},
            {'id': 141, 'name': 'pillow', 'supercategory': 'textile'},
            {'id': 142, 'name': 'plant-other', 'supercategory': 'plant'},
            {'id': 143, 'name': 'plastic', 'supercategory': 'raw-material'},
            {'id': 144, 'name': 'platform', 'supercategory': 'ground'},
            {'id': 145, 'name': 'playingfield', 'supercategory': 'ground'},
            {'id': 146, 'name': 'railing', 'supercategory': 'structural'},
            {'id': 147, 'name': 'railroad', 'supercategory': 'ground'},
            {'id': 148, 'name': 'river', 'supercategory': 'water'},
            {'id': 149, 'name': 'road', 'supercategory': 'ground'},
            {'id': 150, 'name': 'rock', 'supercategory': 'solid'},
            {'id': 151, 'name': 'roof', 'supercategory': 'building'},
            {'id': 152, 'name': 'rug', 'supercategory': 'textile'},
            {'id': 153, 'name': 'salad', 'supercategory': 'food-stuff'},
            {'id': 154, 'name': 'sand', 'supercategory': 'ground'},
            {'id': 155, 'name': 'sea', 'supercategory': 'water'},
            {'id': 156, 'name': 'shelf', 'supercategory': 'furniture-stuff'},
            {'id': 157, 'name': 'sky-other', 'supercategory': 'sky'},
            {'id': 158, 'name': 'skyscraper', 'supercategory': 'building'},
            {'id': 159, 'name': 'snow', 'supercategory': 'ground'},
            {'id': 160, 'name': 'solid-other', 'supercategory': 'solid'},
            {'id': 161, 'name': 'stairs', 'supercategory': 'furniture-stuff'},
            {'id': 162, 'name': 'stone', 'supercategory': 'solid'},
            {'id': 163, 'name': 'straw', 'supercategory': 'plant'},
            {'id': 164, 'name': 'structural-other', 'supercategory': 'structural'},
            {'id': 165, 'name': 'table', 'supercategory': 'furniture-stuff'},
            {'id': 166, 'name': 'tent', 'supercategory': 'building'},
            {'id': 167, 'name': 'textile-other', 'supercategory': 'textile'},
            {'id': 168, 'name': 'towel', 'supercategory': 'textile'},
            {'id': 169, 'name': 'tree', 'supercategory': 'plant'},
            {'id': 170, 'name': 'vegetable', 'supercategory': 'food-stuff'},
            {'id': 171, 'name': 'wall-brick', 'supercategory': 'wall'},
            {'id': 172, 'name': 'wall-concrete', 'supercategory': 'wall'},
            {'id': 173, 'name': 'wall-other', 'supercategory': 'wall'},
            {'id': 174, 'name': 'wall-panel', 'supercategory': 'wall'},
            {'id': 175, 'name': 'wall-stone', 'supercategory': 'wall'},
            {'id': 176, 'name': 'wall-tile', 'supercategory': 'wall'},
            {'id': 177, 'name': 'wall-wood', 'supercategory': 'wall'},
            {'id': 178, 'name': 'water-other', 'supercategory': 'water'},
            {'id': 179, 'name': 'waterdrops', 'supercategory': 'water'},
            {'id': 180, 'name': 'window-blind', 'supercategory': 'window'},
            {'id': 181, 'name': 'window-other', 'supercategory': 'window'},
            {'id': 182, 'name': 'wood', 'supercategory': 'solid'}
        ])
        if mergeThings:
            catdata.extend([{'id': stuffEndId+1, 'name': 'other', 'supercategory': 'other'}])
        catdata = {'categories': catdata}
        catStr = json.dumps(catdata, indent=indent)
        catStr = catStr[1:-1]  # Remove brackets

        # Write opening braces, headers and annotation start to disk
        output.write(unicode('{\n' + infoStr + imStr + licStr + catStr))

        # Start annots
        if outputAnnots:
            output.write(unicode(',\n"annotations": \n[\n'))
            for i, imgName in enumerate(imgNames):

                # Write annotations
                imgId = imgIds[i]
                diffTime = time.clock() - startTime
                print "Writing JSON annotation %d of %d (%.1fs): %s..." % (i+1, imgCount, diffTime, imgName)

                # Read annotation file
                annotPath = os.path.join(annotFolder, imgName)
                matfile = scipy.io.loadmat(annotPath)
                labelMap = matfile['S']
                if not np.all([j == 0 or j >= stuffStartId for j in np.unique(labelMap)]):
                    raise Exception('Error: .mat annotation files should not contain thing labels!')

                # Merge thing classes
                if mergeThings:
                    # Get thing GT
                    labelMapThings = cocoSegmentationToSegmentationMap(cocoGt, imgId, checkUniquePixelLabel=False, includeCrowd=includeCrowd)
                    if labelMap.shape[0] != labelMapThings.shape[0] \
                        or labelMap.shape[1] != labelMapThings.shape[1]:
                        raise Exception('Error: Stuff segmentation map has different size from thing segmentation map!')

                    # Set all thing classes to the new 'other' class
                    labelMap[labelMapThings > 0] = stuffEndId + 1

                # Add stuff annotations
                labelsAll = np.unique(labelMap)
                labelsValid = [i for i in labelsAll if i >= stuffStartId]
                for i, labelId in enumerate(labelsValid):
                    # Add a comma and line break after each annotation
                    assert annId - annIdStart <= 1e7, 'Error: Annotation ids are not unique!'
                    if annId == annIdStart:
                        annotStr = ''
                    else:
                        annotStr = ',\n'

                    # Create mask and encode it
                    Rs = segmentationToCocoMask(labelMap, labelId)

                    # Create annotation data
                    anndata = {}
                    anndata['id'] = annId
                    anndata['image_id'] = int(imgId)
                    anndata['category_id'] = int(labelId)
                    anndata['segmentation'] = Rs
                    anndata['area'] = float(mask.area(Rs))
                    anndata['bbox'] = mask.toBbox(Rs).tolist()
                    anndata['iscrowd'] = 0

                    # Write JSON
                    annotStr = annotStr + json.dumps(anndata, indent=indent)
                    output.write(unicode(annotStr))

                    # Increment annId
                    annId = annId + 1

            # End annots
            output.write(unicode('\n]'))

        # Global end
        output.write(unicode('\n}'))

if __name__ == "__main__":
    internalToCocoGTDemo()
