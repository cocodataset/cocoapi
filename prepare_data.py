"""
download images based on some categories
"""
from cocoapi.PythonAPI.pycocotools.coco import COCO
import tensorflow as tf


tf.flags.DEFINE_string("obj_ann_file", "",
                       "object detection annotations file path")

tf.flags.DEFINE_string("cap_ann_file", "",
                       "captions annotations file path")

tf.flags.DEFINE_string("data_dir", "",
                       "dataset directory")


FLAGS = tf.flags.FLAGS

obj_ann_file = FLAGS.obj_ann_file
cap_ann_file = FLAGS.cap_ann_file
data_dir = FLAGS.data_dir

coco_obj = COCO(obj_ann_file)
coco_cap = COCO(cap_ann_file)

cats = coco_obj.loadCats(coco_obj.getCatIds())
nms = set([cat['supercategory'] for cat in cats])

catIds = coco_obj.getCatIds(supNms=['person', 'outdoor', 'indoor', 'food', 'animal'])
imgIds = coco_obj.getImgIds(catIds=catIds)
# test  imgIds = imgIds[0]
coco_cap.download(data_dir, imgIds)

imgs = coco_cap.loadImgs(imgIds)

annIds = coco_cap.getAnnIds(imgIds)
anns = coco_cap.loadAnns(annIds)

