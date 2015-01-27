__author__ = 'tylin'
__version__ = 1.0
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
#  segToMask  - Convert polygon segmentation to binary mask.
#  showAnns   - Display the specified annotations.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>segToMask, COCO>showAnns

# Microsoft COCO Toolbox.      Version 1.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from skimage.draw import polygon

class COCO:
    def __init__(self, annotation_file='annotations/instances_val2014_1_0.json'):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        print 'loading annotations into memory...'
        time_t = datetime.datetime.utcnow()
        dataset = json.load(open(annotation_file, 'r'))
        print datetime.datetime.utcnow() - time_t
        print 'annotations loaded!'

        time_t = datetime.datetime.utcnow()
        # create index
        print 'creating index...'
        imgToAnns = {ann['image_id']: [] for ann in dataset['annotations']}
        anns =      {ann['id']:       [] for ann in dataset['annotations']}
        for ann in dataset['annotations']:
            imgToAnns[ann['image_id']] += [ann]
            anns[ann['id']] = ann

        imgs      = {im['id']: {} for im in dataset['images']}
        for img in dataset['images']:
            imgs[img['id']] = img

        cats = []
        catToImgs = []
        if dataset['type'] == 'instances':
            cats = {cat['id']: [] for cat in dataset['categories']}
            for cat in dataset['categories']:
                cats[cat['id']] = cat
            catToImgs = {cat['id']: [] for cat in dataset['categories']}
            for ann in dataset['annotations']:
                catToImgs[ann['category_id']] += [ann['image_id']]

        print datetime.datetime.utcnow() - time_t
        print 'index created!'

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
        self.dataset = dataset


    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.datset['info'].items():
            print '%s: %s'%(key, value)

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)

        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                anns = sum([self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns],[])
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if self.dataset['type'] == 'instances':
            if not iscrowd == None:
                ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
            else:
                ids = [ann['id'] for ann in anns]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if type(catNms) == list else [catNms]
        supNms = supNms if type(supNms) == list else [supNms]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for catId in catIds:
                if len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if type(ids) == list:
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if type(ids) == list:
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if type(ids) == list:
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def showAnns(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if self.dataset['type'] == 'instances':
            ax = plt.gca()
            polygons = []
            color = []
            for ann in anns:
                c = np.random.random((1, 3)).tolist()[0]
                if not ann['iscrowd']:
                    # polygon
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape((len(seg)/2, 2))
                        polygons.append(Polygon(poly, True,alpha=0.4))
                        color.append(c)
                else:
                    # mask
                    mask = COCO.decodeMask(ann['segmentation'])
                    img = np.ones( (mask.shape[0], mask.shape[1], 3) )
                    light_green = np.array([2.0,166.0,101.0])/255
                    for i in range(3):
                        img[:,:,i] = light_green[i]
                    ax.imshow(np.dstack( (img, mask*0.5) ))
            p = PatchCollection(polygons, facecolors=color, edgecolors=(0,0,0,1), linewidths=3, alpha=0.4)
            ax.add_collection(p)
        if self.dataset['type'] == 'captions':
            for ann in anns:
                print ann['caption']


    @staticmethod
    def decodeMask(R):
        """
        Decode binary mask M encoded via run-length encoding.
        :param   R (object RLE)    : run-length encoding of binary mask
        :return: M (bool 2D array) : decoded binary mask
        """
        N = len(R['counts'])
        M = np.zeros( (R['size'][0]*R['size'][1], ))
        n = 0
        val = 1
        for pos in range(N):
            val = not val
            for c in range(R['counts'][pos]):
                R['counts'][pos]
                M[n] = val
                n += 1
        return M.reshape((R['size']), order='F')

    @staticmethod
    def encodeMask(M):
        """
        Encode binary mask M using run-length encoding.
        :param   M (bool 2D array)  : binary mask to encode
        :return: R (object RLE)     : run-length encoding of binary mask
        """
        [h, w] = M.shape
        M = M.flatten(order='F')
        N = len(M)
        counts_list = []
        pos = 0
        # counts
        counts_list.append(1)
        diffs = np.logical_xor(M[0:N-1], M[1:N])
        for diff in diffs:
            if diff:
                pos +=1
                counts_list.append(1)
            else:
                counts_list[pos] += 1
        # if array starts from 1. start with 0 counts for 0
        if M[0] == 1:
            counts_list = [0] + counts_list
        return {'size':      [h, w],
               'counts':    counts_list ,
               }

    @staticmethod
    def segToMask( S, h, w ):
         """
         Convert polygon segmentation to binary mask.
         :param   S (float array)   : polygon segmentation mask
         :param   h (int)           : target mask height
         :param   w (int)           : target mask width
         :return: M (bool 2D array) : binary mask
         """
         M = np.zeros((h,w), dtype=np.bool)
         for s in S:
             N = len(s)
             rr, cc = polygon(np.array(s[1:N:2]), np.array(s[0:N:2])) # (y, x)
             M[rr, cc] = 1
         return M