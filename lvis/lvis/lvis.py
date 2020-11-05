"""
API for accessing LVIS Dataset: https://lvisdataset.org.

LVIS API is a Python API that assists in loading, parsing and visualizing
the annotations in LVIS. In addition to this API, please download
images and annotations from the LVIS website.
"""

import json
import logging
import os
from collections import defaultdict
from urllib.request import urlretrieve

import pycocotools.mask as mask_utils


class LVIS:
    def __init__(self, annotation_path):
        """Class for reading and visualizing annotations.
        Args:
            annotation_path (str): location of annotation file
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info('Loading annotations.')

        self.dataset = self._load_json(annotation_path)

        assert (type(self.dataset) == dict
                ), 'Annotation file format {} not supported.'.format(
                    type(self.dataset))
        self._create_index()

    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _create_index(self):
        self.logger.info('Creating index.')

        self.img_ann_map = defaultdict(list)
        self.cat_img_map = defaultdict(list)

        self.anns = {}
        self.cats = {}
        self.imgs = {}

        for ann in self.dataset['annotations']:
            self.img_ann_map[ann['image_id']].append(ann)
            self.anns[ann['id']] = ann

        for img in self.dataset['images']:
            self.imgs[img['id']] = img

        for cat in self.dataset['categories']:
            self.cats[cat['id']] = cat

        for ann in self.dataset['annotations']:
            self.cat_img_map[ann['category_id']].append(ann['image_id'])

        self.logger.info('Index created.')

    def get_ann_ids(self, img_ids=None, cat_ids=None, area_rng=None):
        """Get ann ids that satisfy given filter conditions.

        Args:
            img_ids (int array): get anns for given imgs
            cat_ids (int array): get anns for given cats
            area_rng (float array): get anns for a given area range.
                e.g [0, inf]

        Returns:
            ids (int array): integer array of ann ids
        """
        anns = []
        if img_ids is not None:
            for img_id in img_ids:
                anns.extend(self.img_ann_map[img_id])
        else:
            anns = self.dataset['annotations']

        # return early if no more filtering required
        if cat_ids is None and area_rng is None:
            return [_ann['id'] for _ann in anns]

        cat_ids = set(cat_ids)

        if area_rng is None:
            area_rng = [0, float('inf')]

        ann_ids = [
            _ann['id'] for _ann in anns if _ann['category_id'] in cat_ids
            and _ann['area'] > area_rng[0] and _ann['area'] < area_rng[1]
        ]
        return ann_ids

    def get_cat_ids(self):
        """Get all category ids.

        Returns:
            ids (int array): integer array of category ids
        """
        return list(self.cats.keys())

    def get_img_ids(self):
        """Get all img ids.

        Returns:
            ids (int array): integer array of image ids
        """
        return list(self.imgs.keys())

    def _load_helper(self, _dict, ids):
        if ids is None:
            return list(_dict.values())
        else:
            return [_dict[id] for id in ids]

    def load_anns(self, ids=None):
        """Load anns with the specified ids. If ids=None load all anns.

        Args:
            ids (int array): integer array of annotation ids

        Returns:
            anns (dict array) : loaded annotation objects
        """
        return self._load_helper(self.anns, ids)

    def load_cats(self, ids):
        """Load categories with the specified ids. If ids=None load all
        categories.

        Args:
            ids (int array): integer array of category ids

        Returns:
            cats (dict array) : loaded category dicts
        """
        return self._load_helper(self.cats, ids)

    def load_imgs(self, ids):
        """Load categories with the specified ids. If ids=None load all images.

        Args:
            ids (int array): integer array of image ids

        Returns:
            imgs (dict array) : loaded image dicts
        """
        return self._load_helper(self.imgs, ids)

    def download(self, save_dir, img_ids=None):
        """Download images from mscoco.org server.
        Args:
            save_dir (str): dir to save downloaded images
            img_ids (int array): img ids of images to download
        """
        imgs = self.load_imgs(img_ids)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for img in imgs:
            file_name = os.path.join(save_dir, img['coco_url'].split('/')[-1])
            if not os.path.exists(file_name):
                urlretrieve(img['coco_url'], file_name)

    def ann_to_rle(self, ann):
        """Convert annotation which can be polygons, uncompressed RLE to RLE.
        Args:
            ann (dict) : annotation object

        Returns:
            ann (rle)
        """
        img_data = self.imgs[ann['image_id']]
        h, w = img_data['height'], img_data['width']
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_utils.frPyObjects(segm, h, w)
            rle = mask_utils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = mask_utils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def ann_to_mask(self, ann):
        """Convert annotation which can be polygons, uncompressed RLE, or RLE
        to binary mask.
        Args:
            ann (dict) : annotation object

        Returns:
            binary mask (numpy 2D array)
        """
        rle = self.ann_to_rle(ann)
        return mask_utils.decode(rle)
