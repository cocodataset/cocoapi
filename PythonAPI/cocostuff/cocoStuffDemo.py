#!/usr/bin/python

__author__ = 'hcaesar'

# A preview script that shows how to use the COCO API with stuff annotations.
#
# It lists the categories and super-categories and shows the
# annotations of an example image.
#
# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
# Licensed under the Simplified BSD License [see coco/license.txt]

from pycocotools.coco import COCO
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

def cocoStuffDemo(dataDir='../..', dataType='train2017'):
    '''
    A preview script that shows how to use the COCO API.
    :param dataDir: location of the COCO root folder
    :param dataType: identifier of the ground-truth annotation file
    :return: None
    '''

    # Define path
    annFile = '%s/annotations/stuff_%s.json' % (dataDir, dataType)

    # Initialize COCO ground truth API
    cocoGt = COCO(annFile)

    # Display COCO stuff categories and supercategories
    categories = cocoGt.loadCats(cocoGt.getCatIds())
    categoryNames = [cat['name'] for cat in categories]
    print 'COCO Stuff leaf categories: \n', ' '.join(categoryNames)

    superCategoryNames = sorted(set([cat['supercategory'] for cat in categories]))
    print 'COCO Stuff super categories: \n', ' '.join(superCategoryNames)

    # Load info for a random image
    imgIds = cocoGt.getImgIds()
    imgId = imgIds[np.random.randint(0, len(imgIds))]
    print 'Processing image', imgId, '\n' 
    img = cocoGt.loadImgs(imgId)[0]

    # Load and display image
    I = skimage.io.imread(img['coco_url'])
    plt.figure()
    plt.subplot(121)
    plt.imshow(I)
    plt.axis('off')
    plt.title('original image')

    # Load and display stuff annotations
    annIds = cocoGt.getAnnIds(imgIds=img['id'])
    anns = cocoGt.loadAnns(annIds)
    plt.subplot(122)
    plt.imshow(I)
    cocoGt.showAnns(anns)
    plt.axis('off')
    plt.title('annotated image')
    plt.show()

if __name__ == "__main__":
    cocoStuffDemo()
