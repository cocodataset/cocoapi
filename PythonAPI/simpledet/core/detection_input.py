from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import mxnet as mx

from six.moves.queue import Queue
from threading import Thread
from operator_py.cython.bbox import bbox_overlaps_cython
from operator_py.bbox_transform import nonlinear_transform as bbox_transform


class DetectionAugmentation(object):
    def __init__(self):
        pass

    def apply(self, input_record):
        pass


class ReadRoiRecord(DetectionAugmentation):
    """
    input: image_url, str
           gt_url, str
    output: image, ndarray(h, w, rgb)
            image_raw_meta, tuple(h, w)
            gt, any
    """

    def __init__(self, gt_select):
        super(ReadRoiRecord, self).__init__()
        self.gt_select = gt_select

    def apply(self, input_record):
        image = cv2.imread(input_record["image_url"], cv2.IMREAD_COLOR)
        input_record["image"] = image[:, :, ::-1]
        # TODO: remove this compatibility method
        input_record["gt_bbox"] = np.concatenate([input_record["gt_bbox"],
                                                  input_record["gt_class"].reshape(-1, 1)],
                                                 axis=1)

        # gt_dict = pkl.load(input_record["gt_url"])
        # for s in self.gt_select:
        #     input_record[s] = gt_dict[s]


class Norm2DImage(DetectionAugmentation):
    """
    input: image, ndarray(h, w, rgb)
    output: image, ndarray(h, w, rgb)
    """

    def __init__(self, pNorm):
        super(Norm2DImage, self).__init__()
        self.p = pNorm  # type: NormParam

    def apply(self, input_record):
        p = self.p

        image = input_record["image"].astype(np.float32)

        image -= p.mean
        image /= p.std

        input_record["image"] = image


class Resize2DImageBbox(DetectionAugmentation):
    """
    input: image, ndarray(h, w, rgb)
           gt_bbox, ndarry(n, 5)
    output: image, ndarray(h', w', rgb)
            im_info, tuple(h', w', scale)
            gt_bbox, ndarray(n, 5)
    """

    def __init__(self, pResize):
        super(Resize2DImageBbox, self).__init__()
        self.p = pResize  # type: ResizeParam

    def apply(self, input_record):
        p = self.p

        image = input_record["image"]
        gt_bbox = input_record["gt_bbox"].astype(np.float32)

        short = min(image.shape[:2])
        long = max(image.shape[:2])
        scale = min(p.short / short, p.long / long)

        input_record["image"] = cv2.resize(image, None, None, scale, scale,
                                           interpolation=cv2.INTER_LINEAR)
        # make sure gt boxes do not overflow
        gt_bbox[:, :4] = gt_bbox[:, :4] * scale
        if image.shape[0] < image.shape[1]:
            gt_bbox[:, [0, 2]] = np.clip(gt_bbox[:, [0, 2]], 0, p.long)
            gt_bbox[:, [1, 3]] = np.clip(gt_bbox[:, [1, 3]], 0, p.short)
        else:
            gt_bbox[:, [0, 2]] = np.clip(gt_bbox[:, [0, 2]], 0, p.short)
            gt_bbox[:, [1, 3]] = np.clip(gt_bbox[:, [1, 3]], 0, p.long)
        input_record["gt_bbox"] = gt_bbox

        # exactly as opencv
        h, w = image.shape[:2]
        input_record["im_info"] = (round(h * scale), round(w * scale), scale)


class Resize2DImageBboxByRoidb(DetectionAugmentation):
    """
    input: image, ndarray(h, w, rgb)
           gt_bbox, ndarry(n, 5)
    output: image, ndarray(h', w', rgb)
            im_info, tuple(h', w', scale)
            gt_bbox, ndarray(n, 5)
    """

    def __init__(self):
        super(Resize2DImageBboxByRoidb, self).__init__()
        class ResizeParam:
            long = None
            short = None
        self.resize_aug = Resize2DImageBbox(ResizeParam)

    def apply(self, input_record):
        self.resize_aug.p.long = input_record["resize_long"]
        self.resize_aug.p.short = input_record["resize_short"]

        self.resize_aug.apply(input_record)


class RandResize2DImageBbox(DetectionAugmentation):
    """
    input: image, ndarray(h, w, rgb)
           gt_bbox, ndarry(n, 4)
    output: image, ndarray(h', w', rgb)
            im_info, tuple(h', w', scale)
            gt_bbox, ndarray(n, 4)
    """

    def __init__(self, pRandResize):
        super(RandResize2DImageBbox, self).__init__()
        self.p = pRandResize
        class ResizeParam:
            long = None
            short = None

        self.resize_aug = Resize2DImageBbox(ResizeParam)

    def apply(self, input_record):
        scale_id = np.random.randint(len(self.p.long_ranges))
        self.resize_aug.p.long = self.p.long_ranges[scale_id]
        self.resize_aug.p.short = self.p.short_ranges[scale_id]

        self.resize_aug.apply(input_record)


class Flip2DImageBbox(DetectionAugmentation):
    """
    input: image, ndarray(h, w, rgb)
           gt_bbox, ndarry(n, 4)
    output: image, ndarray(h, w, rgb)
            gt_bbox, ndarray(n, 4)
    """

    def __init__(self):
        super(Flip2DImageBbox, self).__init__()

    def apply(self, input_record):
        if input_record["flipped"]:
            image = input_record["image"]
            gt_bbox = input_record["gt_bbox"]

            input_record["image"] = image[:, ::-1]
            flipped_bbox = gt_bbox.copy()
            h, w = image.shape[:2]
            flipped_bbox[:, 0] = (w - 1) - gt_bbox[:, 2]
            flipped_bbox[:, 2] = (w - 1) - gt_bbox[:, 0]
            input_record["gt_bbox"] = flipped_bbox


class RandCrop2DImageBbox(DetectionAugmentation):
    def __init__(self, pCrop):
        super(RandCrop2DImageBbox, self).__init__()
        self.p = pCrop
        assert pCrop.mode in ["center", "random"], "The {} crop mode is not supported".format(pCrop.mode)

    def apply(self, input_record):
        p = self.p

        image = input_record["image"]
        gt_bbox = input_record["gt_bbox"]

        if image.shape[0] >= image.shape[1]:
            crop_w, crop_h = p.short, p.long
        else:
            crop_w, crop_h = p.long, p.short
        crop_w = min(crop_w, image.shape[1])
        crop_h = min(crop_h, image.shape[0])

        if p.mode == "center" and gt_bbox.shape[0] > 0:
            # random select a box as cropping center
            rand_index = np.random.randint(gt_bbox.shape[0])
            box = gt_bbox[rand_index, :]

            # decide start point
            ctr_x = (box[2] + box[0]) / 2.0
            ctr_y = (box[3] + box[1]) / 2.0
            noise_h = np.random.randint(-10, 10)
            noise_w = np.random.randint(-30, 30)
            start_h = int(round(ctr_y - crop_h / 2)) + noise_h
            start_w = int(round(ctr_x - crop_w / 2)) + noise_w
            end_h = start_h + crop_h
            end_w = start_w + crop_w

            # prevent crop cross border
            if start_h < 0:
                off = -start_h
                start_h += off
                end_h += off
            if start_w < 0:
                off = -start_w
                start_w += off
                end_w += off
            if end_h > image.shape[0]:
                off = end_h - image.shape[0]
                end_h -= off
                start_h -= off
            if end_w > image.shape[1]:
                off = end_w - image.shape[1]
                end_w -= off
                start_w -= off
        else:
            # random crop from image
            start_h = np.random.randint(0, image.shape[0] - crop_h + 1)
            start_w = np.random.randint(0, image.shape[1] - crop_w + 1)
            end_h = start_h + crop_h
            end_w = start_w + crop_w

        assert start_h >= 0 and start_w >= 0 and end_h <= image.shape[0] and end_w <= image.shape[1]

        # crop then resize
        im_cropped = image[start_h:end_h, start_w:end_w, :]
        # transform ground truth
        ctrs_x = (gt_bbox[:, 2] + gt_bbox[:, 0]) / 2.0
        ctrs_y = (gt_bbox[:, 3] + gt_bbox[:, 1]) / 2.0
        keep = np.where((ctrs_y > start_h) & (ctrs_x > start_w) & (ctrs_y < end_h) & (ctrs_x < end_w))
        gt_bbox = gt_bbox[keep]
        gt_bbox[:, [0, 2]] -= start_w
        gt_bbox[:, [1, 3]] -= start_h
        gt_bbox[:, [0, 2]] = np.clip(gt_bbox[:, [0, 2]], 0, crop_w - 1)
        gt_bbox[:, [1, 3]] = np.clip(gt_bbox[:, [1, 3]], 0, crop_h - 1)

        input_record["image"] = im_cropped
        input_record["gt_bbox"] = gt_bbox
        input_record["im_info"] = (crop_h, crop_w, input_record["im_info"][2])


class Pad2DImageBbox(DetectionAugmentation):
    """
    input: image, ndarray(h, w, rgb)
           gt_bbox, ndarry(n, 5)
    output: image, ndarray(h, w, rgb)
            gt_bbox, ndarray(max_num_gt, 5)
    """

    def __init__(self, pPad):
        super(Pad2DImageBbox, self).__init__()
        self.p = pPad  # type: PadParam

    def apply(self, input_record):
        p = self.p

        image = input_record["image"]
        gt_bbox = input_record["gt_bbox"]

        h, w = image.shape[:2]
        shape = (p.long, p.short, 3) if h >= w \
            else (p.short, p.long, 3)

        padded_image = np.zeros(shape, dtype=np.float32)
        padded_image[:h, :w] = image
        padded_gt_bbox = np.full(shape=(p.max_num_gt, 5), fill_value=-1, dtype=np.float32)
        padded_gt_bbox[:len(gt_bbox)] = gt_bbox

        input_record["image"] = padded_image
        input_record["gt_bbox"] = padded_gt_bbox


class ConvertImageFromHwcToChw(DetectionAugmentation):
    def __init__(self):
        super(ConvertImageFromHwcToChw, self).__init__()

    def apply(self, input_record):
        input_record["image"] = input_record["image"].transpose((2, 0, 1))


class AnchorTarget2D(DetectionAugmentation):
    """
    input: image_meta: tuple(h, w, scale)
           gt_bbox, ndarry(max_num_gt, 5)
    output: anchor_label, ndarray(num_anchor * 2, h, w)
            anchor_bbox_target, ndarray(num_anchor * 4, h, w)
            anchor_bbox_weight, ndarray(num_anchor * 4, h, w)
    """

    def __init__(self, pAnchor):
        super(AnchorTarget2D, self).__init__()
        self.p = pAnchor  # type: AnchorTarget2DParam

        self.__base_anchor = None
        self.__v_all_anchor = None
        self.__h_all_anchor = None
        self.__num_anchor = None

        self.DEBUG = False

    @property
    def base_anchor(self):
        if self.__base_anchor is not None:
            return self.__base_anchor

        p = self.p

        base_anchor = np.array([0, 0, p.generate.stride - 1, self.p.generate.stride - 1])

        w = base_anchor[2] - base_anchor[0] + 1
        h = base_anchor[3] - base_anchor[1] + 1
        x_ctr = base_anchor[0] + 0.5 * (w - 1)
        y_ctr = base_anchor[1] + 0.5 * (h - 1)

        w_ratios = np.round(np.sqrt(w * h / p.generate.aspects))
        h_ratios = np.round(w_ratios * p.generate.aspects)
        ws = (np.outer(w_ratios, p.generate.scales)).reshape(-1)
        hs = (np.outer(h_ratios, p.generate.scales)).reshape(-1)

        base_anchor = np.stack(
            [x_ctr - 0.5 * (ws - 1),
             y_ctr - 0.5 * (hs - 1),
             x_ctr + 0.5 * (ws - 1),
             y_ctr + 0.5 * (hs - 1)],
            axis=1)

        self.__base_anchor = base_anchor
        return self.__base_anchor

    @property
    def v_all_anchor(self):
        if self.__v_all_anchor is not None:
            return self.__v_all_anchor

        p = self.p

        shift_x = np.arange(0, p.generate.short, dtype=np.float32) * p.generate.stride
        shift_y = np.arange(0, p.generate.long, dtype=np.float32) * p.generate.stride
        grid_x, grid_y = np.meshgrid(shift_x, shift_y)
        grid_x, grid_y = grid_x.reshape(-1), grid_y.reshape(-1)
        grid = np.stack([grid_x, grid_y, grid_x, grid_y], axis=1)
        all_anchor = grid[:, None, :] + self.base_anchor[None, :, :]
        all_anchor = all_anchor.reshape(-1, 4)

        self.__v_all_anchor = all_anchor
        self.__num_anchor = all_anchor.shape[0]
        return self.__v_all_anchor

    @property
    def h_all_anchor(self):
        if self.__h_all_anchor is not None:
            return self.__h_all_anchor

        p = self.p

        shift_x = np.arange(0, p.generate.long, dtype=np.float32) * p.generate.stride
        shift_y = np.arange(0, p.generate.short, dtype=np.float32) * p.generate.stride
        grid_x, grid_y = np.meshgrid(shift_x, shift_y)
        grid_x, grid_y = grid_x.reshape(-1), grid_y.reshape(-1)
        grid = np.stack([grid_x, grid_y, grid_x, grid_y], axis=1)
        all_anchor = grid[:, None, :] + self.base_anchor[None, :, :]
        all_anchor = all_anchor.reshape(-1, 4)

        self.__h_all_anchor = all_anchor
        self.__num_anchor = all_anchor.shape[0]
        return self.__h_all_anchor

    @v_all_anchor.setter
    def v_all_anchor(self, value):
        self.__v_all_anchor = value
        self.__num_anchor = value.shape[0]

    @h_all_anchor.setter
    def h_all_anchor(self, value):
        self.__h_all_anchor = value
        self.__num_anchor = value.shape[0]



    def _assign_label_to_anchor(self, valid_anchor, gt_bbox, neg_thr, pos_thr, min_pos_thr):
        num_anchor = valid_anchor.shape[0]
        cls_label = np.full(shape=(num_anchor,), fill_value=-1, dtype=np.float32)

        if len(gt_bbox) > 0:
            # num_anchor x num_gt
            overlaps = bbox_overlaps_cython(valid_anchor.astype(np.float32, copy=False), gt_bbox.astype(np.float32, copy=False))
            max_overlaps = overlaps.max(axis=1)
            argmax_overlaps = overlaps.argmax(axis=1)
            gt_max_overlaps = overlaps.max(axis=0)

            # TODO: speed up this
            # TODO: fix potentially assigning wrong anchors as positive
            # A correct implementation is given as
            # gt_argmax_overlaps = np.where((overlaps.transpose() == gt_max_overlaps[:, None]) &
            #                               (overlaps.transpose() >= min_pos_thr))[1]
            gt_argmax_overlaps = np.where((overlaps == gt_max_overlaps) &
                                          (overlaps >= min_pos_thr))[0]
            # anchor class
            cls_label[max_overlaps < neg_thr] = 0
            # fg label: for each gt, anchor with highest overlap
            cls_label[gt_argmax_overlaps] = 1
            # fg label: above threshold IoU
            cls_label[max_overlaps >= pos_thr] = 1
        else:
            cls_label[:] = 0
            argmax_overlaps = np.zeros(shape=(num_anchor, ))

        return cls_label, argmax_overlaps

    def _sample_anchor(self, label, num, fg_fraction):
        num_fg = int(fg_fraction * num)
        fg_inds = np.where(label == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = np.random.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            if self.DEBUG:
                disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
            label[disable_inds] = -1

        num_bg = num - np.sum(label == 1)
        bg_inds = np.where(label == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            if self.DEBUG:
                disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
            label[disable_inds] = -1

    def _cal_anchor_target(self, label, valid_anchor, gt_bbox, anchor_label):
        num_anchor = valid_anchor.shape[0]
        reg_target = np.zeros(shape=(num_anchor, 4), dtype=np.float32)
        reg_weight = np.zeros(shape=(num_anchor, 4), dtype=np.float32)
        fg_index = np.where(label == 1)[0]
        if len(fg_index) > 0:
            reg_target[fg_index] = bbox_transform(valid_anchor[fg_index], gt_bbox[anchor_label[fg_index], :4])
            reg_weight[fg_index, :] = 1.0

        return reg_target, reg_weight

    def _gather_valid_anchor(self, image_info):
        h, w = image_info[:2]
        all_anchor = self.v_all_anchor if h >= w else self.h_all_anchor
        allowed_border = self.p.assign.allowed_border
        valid_index = np.where((all_anchor[:, 0] >= -allowed_border) &
                               (all_anchor[:, 1] >= -allowed_border) &
                               (all_anchor[:, 2] < w + allowed_border) &
                               (all_anchor[:, 3] < h + allowed_border))[0]
        return valid_index, all_anchor[valid_index]

    def _scatter_valid_anchor(self, valid_index, cls_label, reg_target, reg_weight):
        num_anchor = self.__num_anchor

        all_cls_label = np.full(shape=(num_anchor,), fill_value=-1, dtype=np.float32)
        all_reg_target = np.zeros(shape=(num_anchor, 4), dtype=np.float32)
        all_reg_weight = np.zeros(shape=(num_anchor, 4), dtype=np.float32)

        all_cls_label[valid_index] = cls_label
        all_reg_target[valid_index] = reg_target
        all_reg_weight[valid_index] = reg_weight

        return all_cls_label, all_reg_target, all_reg_weight

    def apply(self, input_record):
        p = self.p

        im_info = input_record["im_info"]
        gt_bbox = input_record["gt_bbox"]
        assert isinstance(gt_bbox, np.ndarray)
        assert gt_bbox.dtype == np.float32
        valid = np.where(gt_bbox[:, 0] != -1)[0]
        gt_bbox = gt_bbox[valid]

        if gt_bbox.shape[1] == 5:
            gt_bbox = gt_bbox[:, :4]

        valid_index, valid_anchor = self._gather_valid_anchor(im_info)
        cls_label, anchor_label = \
            self._assign_label_to_anchor(valid_anchor, gt_bbox,
                                         p.assign.neg_thr, p.assign.pos_thr, p.assign.min_pos_thr)
        self._sample_anchor(cls_label, p.sample.image_anchor, p.sample.pos_fraction)
        reg_target, reg_weight = self._cal_anchor_target(cls_label, valid_anchor, gt_bbox, anchor_label)
        cls_label, reg_target, reg_weight = \
            self._scatter_valid_anchor(valid_index, cls_label, reg_target, reg_weight)

        h, w = im_info[:2]
        if h >= w:
            fh, fw = p.generate.long, p.generate.short
        else:
            fh, fw = p.generate.short, p.generate.long

        input_record["rpn_cls_label"] = cls_label.reshape((fh, fw, -1)).transpose(2, 0, 1).reshape(-1)
        input_record["rpn_reg_target"] = reg_target.reshape((fh, fw, -1)).transpose(2, 0, 1)
        input_record["rpn_reg_weight"] = reg_weight.reshape((fh, fw, -1)).transpose(2, 0, 1)

        return input_record["rpn_cls_label"], \
               input_record["rpn_reg_target"], \
               input_record["rpn_reg_weight"]


class RenameRecord(DetectionAugmentation):
    def __init__(self, mapping):
        super(RenameRecord, self).__init__()
        self.mapping = mapping

    def apply(self, input_record):
        for k, new_k in self.mapping.items():
            input_record[new_k] = input_record[k]
            del input_record[k]


class Loader(mx.io.DataIter):
    """
    Loader is now a 3-thread design,
    Loader.next is called in the main thread,
    multiple worker threads are responsible for performing transform,
    a collector thread is responsible for converting numpy array to mxnet array.
    """

    def __init__(self, roidb, transform, data_name, label_name, batch_size=1,
                 shuffle=False, num_worker=None, num_collector=None, 
                 worker_queue_depth=None, collector_queue_depth=None, kv=None):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param roidb: must be preprocessed
        :param batch_size:
        :param shuffle: bool
        :return: Loader
        """
        super(Loader, self).__init__(batch_size=batch_size)

        if kv:
            (self.rank, self.num_worker) = (kv.rank, kv.num_workers)
        else:
            (self.rank, self.num_worker) = (0, 1)

        # data processing utilities
        self.transform = transform

        # save parameters as properties
        self.roidb = roidb
        self.shuffle = shuffle

        # infer properties from roidb
        self.index = np.arange(len(roidb))

        # decide data and label names
        self.data_name = data_name
        self.label_name = label_name

        # status variable for synchronization between get_data and get_label
        self._cur = 0

        self.data = None
        self.label = None

        # multi-thread settings
        self.num_worker = num_worker
        self.num_collector = num_collector
        self.index_queue = Queue()
        self.data_queue = Queue(maxsize=worker_queue_depth)
        self.result_queue = Queue(maxsize=collector_queue_depth)
        self.workers = None
        self.collectors = None

        # get first batch to fill in provide_data and provide_label
        self._thread_start()
        self.load_first_batch()
        self.reset()

    @property
    def total_record(self):
        return len(self.index) // self.batch_size * self.batch_size

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label)]

    def _insert_queue(self):
        for i in range(0, len(self.index), self.batch_size):
            batch_index = self.index[i:i + self.batch_size]
            if len(batch_index) == self.batch_size:
                self.index_queue.put(batch_index)

    def _thread_start(self):
        self.workers = \
            [Thread(target=self.worker, args=[self.roidb, self.index_queue, self.data_queue])
             for _ in range(self.num_worker)]

        for worker in self.workers:
            worker.daemon = True
            worker.start()

        self.collectors = [Thread(target=self.collector, args=[]) for _ in range(self.num_collector)]

        for c in self.collectors:
            c.daemon = True
            c.start()

    def reset(self):
        self._cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

        self._insert_queue()

    def iter_next(self):
        return self._cur + self.batch_size <= len(self.index)

    def load_first_batch(self):
        self.index_queue.put(range(self.batch_size))
        self.next()

    def load_batch(self):
        self._cur += self.batch_size
        result = self.result_queue.get()
        return result

    def next(self):
        if self.iter_next():
            # print("[worker] %d" % self.data_queue.qsize())
            # print("[collector] %d" % self.result_queue.qsize())
            result = self.load_batch()
            self.data = result.data
            self.label = result.label
            return result
        else:
            raise StopIteration

    def worker(self, roidb, index_queue, data_queue):
        while True:
            batch_index = index_queue.get()

            records = []
            for index in batch_index:
                roi_record = roidb[index].copy()
                for trans in self.transform:
                    trans.apply(roi_record)
                records.append(roi_record)
            data_batch = {}
            for name in self.data_name + self.label_name:
                data_batch[name] = np.stack([r[name] for r in records])
            data_queue.put(data_batch)

    def collector(self):
        while True:
            record = self.data_queue.get()
            data = [mx.nd.array(record[name]) for name in self.data_name]
            label = [mx.nd.array(record[name]) for name in self.label_name]
            provide_data = [(k, v.shape) for k, v in zip(self.data_name, data)]
            provide_label = [(k, v.shape) for k, v in zip(self.label_name, label)]
            data_batch = mx.io.DataBatch(data=data,
                                         label=label,
                                         provide_data=provide_data,
                                         provide_label=provide_label)
            self.result_queue.put(data_batch)


class SequentialLoader(mx.io.DataIter):
    def __init__(self, iters):
        super(SequentialLoader, self).__init__()
        self.iters = iters
        self.exhausted = [False] * len(iters)

    def __getattr__(self, attr):
        # delegate unknown keys to underlying iterators
        first_non_empty_idx = self.exhausted.index(False)
        first_non_empty_iter = self.iters[first_non_empty_idx]
        return getattr(first_non_empty_iter, attr)

    def next(self):
        while True:
            if all(self.exhausted):
                raise StopIteration
            first_non_empty_idx = self.exhausted.index(False)
            first_non_empty_iter = self.iters[first_non_empty_idx]
            try:
                result = first_non_empty_iter.next()
                return result
            except StopIteration:
                self.exhausted[first_non_empty_idx] = True

    def reset(self):
        for it in self.iters:
            it.reset()
        self.exhausted = [False] * len(self.iters)

    @property
    def provide_data(self):
        return self.iters[0].provide_data

    @property
    def provide_label(self):
        return self.iters[0].provide_label


class AnchorLoader(mx.io.DataIter):
    def __init__(self, roidb, transform, data_name, label_name, batch_size=1,
                 shuffle=False, num_worker=12, num_collector=4, worker_queue_depth=4,
                 collector_queue_depth=4, kv=None):
        super(AnchorLoader, self).__init__(batch_size=batch_size)

        v_roidb, h_roidb = self.roidb_aspect_group(roidb)

        if kv:
            rank, num_rank = kv.rank, kv.num_workers
        else:
            rank, num_rank = 0, 1

        if num_rank > 1:
            v_part = len(v_roidb) // num_rank
            v_remain = len(v_roidb) % num_rank
            v_roidb_part = v_roidb[rank * v_part:(rank + 1) * v_part]
            v_roidb_part += v_roidb[-v_remain:][rank:rank+1]
            h_part = len(h_roidb) // num_rank
            h_remain = len(h_roidb) % num_rank
            h_roidb_part = h_roidb[rank * h_part:(rank + 1) * h_part]
            h_roidb_part += h_roidb[-h_remain:][rank:rank+1]
        else:
            v_roidb_part = v_roidb
            h_roidb_part = h_roidb

        loaders = []
        if len(h_roidb_part) >= batch_size:
            h_loader = Loader(roidb=h_roidb_part,
                              transform=transform,
                              data_name=data_name,
                              label_name=label_name,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_worker=num_worker,
                              num_collector=num_collector,
                              worker_queue_depth=worker_queue_depth,
                              collector_queue_depth=collector_queue_depth,
                              kv=kv)
            loaders.append(h_loader)
        if len(v_roidb_part) >= batch_size:
            v_loader = Loader(roidb=v_roidb_part,
                              transform=transform,
                              data_name=data_name,
                              label_name=label_name,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_worker=num_worker,
                              num_collector=num_collector,
                              worker_queue_depth=worker_queue_depth,
                              collector_queue_depth=collector_queue_depth,
                              kv=kv)
            loaders.append(v_loader)
        assert len(loaders) > 0, "at least one loader should be constructed"
        self.__loader = SequentialLoader(loaders)

    @property
    def total_record(self):
        return sum([it.total_record for it in self.__loader.iters])

    def __len__(self):
        return self.total_record

    def __getattr__(self, attr):
        # delegate unknown keys to underlying iterators
        return getattr(self.__loader, attr)

    def next(self):
        return self.__loader.next()

    def reset(self):
        return self.__loader.reset()

    @staticmethod
    def roidb_aspect_group(roidb):
        v_roidb, h_roidb = [], []
        for roirec in roidb:
            if roirec["h"] >= roirec["w"]:
                v_roidb.append(roirec)
            else:
                h_roidb.append(roirec)
        return v_roidb, h_roidb


def visualize_anchor_loader(batch_data):
    image = batch_data.data[0][0].asnumpy().astype(np.uint8).transpose((1, 2, 0)).copy()
    gt_bbox = batch_data.data[2][0].asnumpy().astype(np.int32)
    for box in gt_bbox:
        cv2.rectangle(image, tuple(box[:2]), tuple(box[2:4]), color=(0, 255, 0))
    cv2.imshow("image", image)
    cv2.waitKey()


def visualize_anchor_loader_old(batch_data):
    image = batch_data.data[0][0].asnumpy().astype(np.uint8).transpose((1, 2, 0)).copy()
    gt_bbox = batch_data.label[3][0].asnumpy().astype(np.int32)
    for box in gt_bbox:
        cv2.rectangle(image, tuple(box[:2]), tuple(box[2:4]), color=(0, 255, 0))
    cv2.imshow("image", image)
    cv2.waitKey()


def visualize_original_input(roirec):
    image = cv2.imread(roirec["image_url"], cv2.IMREAD_COLOR)
    gt_bbox = roirec["gt_bbox"]
    for box in gt_bbox:
        cv2.rectangle(image, tuple(box[:2]), tuple(box[2:4]), color=(0, 255, 0))
    cv2.imshow("image", image)
    cv2.waitKey()
