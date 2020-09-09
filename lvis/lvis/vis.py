import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from lvis.colormap import colormap
from lvis.lvis import LVIS
from lvis.results import LVISResults


class LVISVis:
    def __init__(self, lvis_gt, lvis_dt=None, img_dir=None, dpi=75):
        """Constructor for LVISVis.

        Args:
            lvis_gt (LVIS class instance, or str containing path of annotation
                file)
            lvis_dt (LVISResult class instance, or str containing path of
                result file, or list of dict)
            img_dir (str): path of folder containing all images. If None, the
                image to be displayed will be downloaded to the current working
                dir.
            dpi (int): dpi for figure size setup
        """
        self.logger = logging.getLogger(__name__)

        if isinstance(lvis_gt, LVIS):
            self.lvis_gt = lvis_gt
        elif isinstance(lvis_gt, str):
            self.lvis_gt = LVIS(lvis_gt)
        else:
            raise TypeError('Unsupported type {} of lvis_gt.'.format(lvis_gt))

        if lvis_dt is not None:
            if isinstance(lvis_dt, LVISResults):
                self.lvis_dt = lvis_dt
            elif isinstance(lvis_dt, (str, list)):
                self.lvis_dt = LVISResults(self.lvis_gt, lvis_dt)
            else:
                raise TypeError(
                    'Unsupported type {} of lvis_dt.'.format(lvis_dt))
        else:
            self.lvis_dt = None
        self.dpi = dpi
        self.img_dir = img_dir if img_dir else '.'
        if self.img_dir == '.':
            self.logger.warn(
                'img_dir not specified. Images will be downloaded.')

    def coco_segm_to_poly(self, _list):
        x = _list[0::2]
        y = _list[1::2]
        points = np.asarray([x, y])
        return np.transpose(points)

    def get_synset(self, idx):
        synset = self.lvis_gt.load_cats(ids=[idx])[0]['synset']
        text = synset.split('.')
        text = '{}.{}'.format(text[0], int(text[-1]))
        return text

    def setup_figure(self, img, title='', dpi=75):
        fig = plt.figure(frameon=False)
        fig.set_size_inches(img.shape[1] / dpi, img.shape[0] / dpi)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_title(title)
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(img)
        return fig, ax

    def vis_bbox(self, ax, bbox, box_alpha=0.5, edgecolor='g', linestyle='--'):
        # bbox should be of the form x, y, w, h
        ax.add_patch(
            plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                fill=False,
                edgecolor=edgecolor,
                linewidth=2.5,
                alpha=box_alpha,
                linestyle=linestyle,
            ))

    def vis_text(self, ax, bbox, text, color='w'):
        ax.text(
            bbox[0],
            bbox[1] - 2,
            text,
            fontsize=15,
            family='serif',
            bbox=dict(facecolor='none', alpha=0.4, pad=0, edgecolor='none'),
            color=color,
            zorder=10,
        )

    def vis_mask(self, ax, segm, color):
        # segm is numpy array of shape Nx2
        polygon = Polygon(segm,
                          fill=True,
                          facecolor=color,
                          edgecolor=color,
                          linewidth=3,
                          alpha=0.5)
        ax.add_patch(polygon)

    def get_color(self, idx):
        color_list = colormap(rgb=True) / 255
        return color_list[idx % len(color_list), 0:3]

    def load_img(self, img_id):
        img = self.lvis_gt.load_imgs([img_id])[0]
        img_path = os.path.join(self.img_dir, img['coco_url'].split('/')[-1])
        if not os.path.exists(img_path):
            self.lvis_gt.download(self.img_dir, img_ids=[img_id])
        img = cv2.imread(img_path)
        b, g, r = cv2.split(img)
        return cv2.merge([r, g, b])

    def vis_img(self,
                img_id,
                show_boxes=False,
                show_segms=True,
                show_classes=False,
                cat_ids_to_show=None):
        ann_ids = self.lvis_gt.get_ann_ids(img_ids=[img_id])
        anns = self.lvis_gt.load_anns(ids=ann_ids)
        boxes, segms, classes = [], [], []
        for ann in anns:
            boxes.append(ann['bbox'])
            segms.append(ann['segmentation'])
            classes.append(ann['category_id'])

        if len(boxes) == 0:
            self.logger.warn('No gt anno found for img_id: {}'.format(img_id))
            return

        boxes = np.asarray(boxes)
        areas = boxes[:, 2] * boxes[:, 3]
        sorted_inds = np.argsort(-areas)

        fig, ax = self.setup_figure(self.load_img(img_id))

        for idx in sorted_inds:
            if cat_ids_to_show is not None and classes[
                    idx] not in cat_ids_to_show:
                continue
            color = self.get_color(idx)
            if show_boxes:
                self.vis_bbox(ax, boxes[idx], edgecolor=color)
            if show_classes:
                text = self.get_synset(classes[idx])
                self.vis_text(ax, boxes[idx], text)
            if show_segms:
                for segm in segms[idx]:
                    self.vis_mask(ax, self.coco_segm_to_poly(segm), color)

    def vis_result(self,
                   img_id,
                   show_boxes=False,
                   show_segms=True,
                   show_classes=False,
                   cat_ids_to_show=None,
                   score_thrs=0.0,
                   show_scores=True):
        assert self.lvis_dt is not None, 'lvis_dt was not specified.'
        anns = self.lvis_dt.get_top_results(img_id, score_thrs)
        boxes, segms, classes, scores = [], [], [], []
        for ann in anns:
            boxes.append(ann['bbox'])
            segms.append(ann['segmentation'])
            classes.append(ann['category_id'])
            scores.append(ann['score'])

        if len(boxes) == 0:
            self.logger.warn('No gt anno found for img_id: {}'.format(img_id))
            return

        boxes = np.asarray(boxes)
        areas = boxes[:, 2] * boxes[:, 3]
        sorted_inds = np.argsort(-areas)

        fig, ax = self.setup_figure(self.load_img(img_id))

        for idx in sorted_inds:
            if cat_ids_to_show is not None and classes[
                    idx] not in cat_ids_to_show:
                continue
            color = self.get_color(idx)
            if show_boxes:
                self.vis_bbox(ax, boxes[idx], edgecolor=color)
            if show_classes:
                text = self.get_synset(classes[idx])
                if show_scores:
                    text = '{}: {:.2f}'.format(text, scores[idx])
                self.vis_text(ax, boxes[idx], text)
            if show_segms:
                for segm in segms[idx]:
                    self.vis_mask(ax, self.coco_segm_to_poly(segm), color)
