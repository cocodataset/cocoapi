
# coding: utf-8

# In[1]:

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt


# In[2]:

dataDir='..'
dataType='val2014'
annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)


# In[3]:

# initialize COCO api for instance annotations
coco=COCO(annFile)


# In[4]:

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print 'COCO categories: \n', ' '.join(nms)
print '\n'
nms = set([cat['supercategory'] for cat in cats])
print 'COCO supercategories: \n', ' '.join(nms)


# In[5]:

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
imgIds = coco.getImgIds(catIds=catIds );
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]


# In[6]:

# load and display image
I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
plt.figure()
plt.imshow(I)
plt.show()

# In[7]:

# load and display instance annotations
plt.imshow(I)
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()

# In[8]:

# initialize COCO api for caption annotations
annFile = '%s/annotations/captions_%s.json'%(dataDir,dataType)
caps=COCO(annFile)


# In[9]:

# load and display caption annotations
annIds = caps.getAnnIds(imgIds=img['id']);
anns = caps.loadAnns(annIds)
caps.showAnns(anns)
plt.imshow(I)
plt.show()
