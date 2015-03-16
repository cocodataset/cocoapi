
# coding: utf-8

# In[1]:

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import json
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


# In[2]:

# Demo demonstrating the algorithm result formats for COCO
# select results type for demo
type = ['segmentation','bbox','caption']
type = type[1]
print 'Running demo for %s results'%(type)

# set appropriate files for given type of results
dataDir='..'
if type == 'segmentation':
    annFile = 'instances_val2014'
    resFile = 'instances_val2014_fakeseg_results'
elif type == 'bbox' :
    annFile = 'instances_val2014'
    resFile = 'instances_val2014_fakebox_results'
elif type == 'caption':
    annFile = 'captions_val2014'
    resFile = 'captions_val2014_fakecap_results'

annFile='%s/annotations/%s.json'%(dataDir,annFile)
resFile='%s/results/%s.json'%(dataDir,resFile)


# In[3]:

# initialize COCO ground truth and results api's
print 'Running demo for %s results.'%(type)
coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)


# In[4]:

# visialuze ground truth and results side by side
imgIds = list(set([ann['image_id'] for ann in cocoRes.loadAnns(cocoRes.getAnnIds())]))
nImgs = len(imgIds)
imgId = imgIds[np.random.randint(nImgs)]
img = coco.loadImgs(imgId)[0]
I = io.imread('%s/images/val2014/%s'%(dataDir,img['file_name']))

# show ground truth labels
annIds = coco.getAnnIds(imgIds=imgId)
anns = coco.loadAnns(annIds)
plt.imshow(I)
coco.showAnns(anns)
plt.title('ground truth', fontsize=20)
plt.axis('off')
plt.show()

# show result labels
annIds = cocoRes.getAnnIds(imgIds=imgId)
anns = cocoRes.loadAnns(annIds)
plt.imshow(I)
coco.showAnns(anns)
plt.title('result', fontsize=20)
plt.axis('off')
plt.show()


# In[5]:

# load raw JSON and show exact format for results
res = json.load(open(resFile))
print 'results structure have the following format:'
print res[0].keys()

# the following command can be used to save the results back to disk
# json.dump(res, open(resFile, 'w'))

