class RootKeys():
    info = 'info'
    images = 'images'
    annotations = 'annotations'
    licenses = 'licenses'
    categories = 'categories'

class InfoKeys():
    year = "year"
    version = "version"
    description = "description"
    contributor = "contributor"
    url = "url"
    date_created = "date_created"

class ImageKeys():
    id = "id"
    width = "width"
    height = "height"
    file_name = "file_name"
    license = "license"
    flickr_url = "flickr_url"
    coco_url = "coco_url"
    date_captured = "date_captured"

class LicenseKeys():
    id = "id"
    name = "name"
    url = "url"

class _ObjDetAnnotationKeys():
    id = "id"
    image_id = "image_id"
    category_id = "category_id"
    segmentation = "segmentation"
    area = "area"
    bbox = "bbox"
    iscrowd = "iscrowd"
    
class _ObjDetCategoryKeys():
    id = "id"
    name = "name"
    supercategory = "supercategory"
    
class _KeyptDetAnnotationKeys(_ObjDetAnnotationKeys):
    keypoints = "keypoints"
    num_keypoints = "num_keypoints"

class _KeyptDetCategoryKeys(_ObjDetCategoryKeys):
    keypoints = "keypoints"
    skeleton = "skeleton"

class _PanopticSegAnnotationKeys():
    image_id = "image_id"
    file_name = "file_name"
    segments_info = "segments_info"

class _PanopticSegSegInfoKeys():
    id = "id"
    category_id = "category_id"
    area = "area"
    bbox = "bbox"
    iscrowd = "iscrowd"

class _PanopticSegCategoryKeys():
    id = "id"
    name = "name"
    supercategory = "supercategory"
    isthing = "isthing"
    color = "color"

class _ImgCapAnnotationKeys():
    id = "id"
    image_id = "image_id"
    caption = "caption"

class ObjDet():
    AnnotationKeys = _ObjDetAnnotationKeys
    CategoryKeys = _ObjDetCategoryKeys

class KeyptDet():
    AnnotationKeys = _KeyptDetAnnotationKeys
    CategoryKeys = _KeyptDetCategoryKeys

class PanopticSeg():
    AnnotationKeys = _PanopticSegAnnotationKeys
    CategoryKeys = _PanopticSegCategoryKeys
    SegInfoKeys = _PanopticSegSegInfoKeys

class ImgCaption():
    AnnotationKeys = _ImgCapAnnotationKeys
