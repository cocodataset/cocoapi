__author__ = 'Xinwei Liu'
__version__ = '2.0'
# Interface for accessing the Microsoft COCO dataset.

# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>annToMask, COCO>showAnns

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Xinwei Liu on November 30, 2018.
# Licensed under the Simplified BSD License [see bsd.txt]

import os
import zipfile
import numpy as np
import cv2
import json
import time
from . coco import COCO



def timer(func):
    '''
    Define a timer, pass in one, and 
    return another method with the timing feature attached
    '''
    def wrapper(*args):
        start = time.time()
        print('Loading json in memory ...')
        value = func(*args)
        end = time.time()
        print('used time: {0:g} s'.format(end - start))
        return value

    return wrapper


class ImageZ:
    '''
    Working with compressed files under the images
    '''

    def __init__(self, root, dataType):
        '''
        root:: root dir
        dataType in ['test2014', 'test2015',
                    'test2017', 'train2014',
                    'train2017', 'unlabeled2017',
                    'val2014', 'val2017']
        '''
        self.Z = self.__get_Z(root, dataType)
        self.names = self.__get_names(self.Z)

    @staticmethod
    def __get_Z(root, dataType):
        '''
        Get the file name of the compressed file under the images
        '''
        dataType = dataType + '.zip'
        return zipfile.ZipFile(os.path.join(root, dataType))

    @staticmethod
    def __get_names(Z):
        names = []
        for name in Z.namelist():
            if not name.endswith('/'):
                names.append(name)
        return names

    def buffer2array(self, image_name):
        '''
        Get picture data directly without decompression

        Parameters
        ===========
        Z:: Picture data is a ZipFile object
        '''
        buffer = self.Z.read(image_name)
        image = np.frombuffer(buffer, dtype="B")  # 将 buffer 转换为 np.uint8 数组
        img_cv = cv2.imdecode(image, cv2.IMREAD_COLOR)  # BGR 格式
        img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, item):
        names = self.names[item]
        if isinstance(item, slice):
            return [self.buffer2array(name) for name in names]
        else:
            return self.buffer2array(names)

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        for name in self.names:
            yield self.buffer2array(name)


class AnnZ(dict):
    '''
    Working with compressed files under annotations
    '''

    def __init__(self, root, annType, *args, **kwds):
        '''
        dataType in [
              'annotations_trainval2014',
              'annotations_trainval2017',
              'image_info_test2014',
              'image_info_test2015',
              'image_info_test2017',
              'image_info_unlabeled2017',
              'panoptic_annotations_trainval2017',
              'stuff_annotations_trainval2017'
        ]
        '''
        super().__init__(*args, **kwds)
        self.__dict__ = self
        self.Z = self.__get_Z(root, annType)
        self.names = self.__get_names(self.Z)

    @staticmethod
    def __get_Z(root, annType):
        '''
        Get the file name of the compressed file under the annotations
        '''
        annType = annType + '.zip'
        annDir = os.path.join(root, 'annotations')
        return zipfile.ZipFile(os.path.join(annDir, annType))

    @staticmethod
    def __get_names(Z):
        names = [name for name in Z.namelist() if not name.endswith('/')]
        return names

    @timer
    def json2dict(self, name):
        with self.Z.open(name) as fp:
            dataset = json.load(fp)
        return dataset


class COCOZ(COCO, dict):
    def __init__(self, annZ, annFile, *args, **kwds):
        '''
        ptint(coco):: View Coco's Instance object Coco's 'info'

        example
        ==========
        annZ = AnnZ(annDir, annType)
        '''
        super().__init__(*args, **kwds)
        self.__dict__ = self
        self.dataset = annZ.json2dict(annFile)
        self.createIndex()

    @timer
    def createIndex(self):
        # create index
        print('creating index...')
        cats, anns, imgs = {}, {}, {}
        imgToAnns, catToImgs = {}, {}
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']] = imgToAnns.get(
                    ann['image_id'], []) + [ann]
                anns[ann['id']] = ann
        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img
        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat
        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']] = catToImgs.get(
                    ann['category_id'], []) + [ann['image_id']]

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def __str__(self):
        """
        Print information about the annotation file.
        """
        S = [
            '{}: {}'.format(key, value)
            for key, value in self.dataset['info'].items()
        ]
        return '\n'.join(S)