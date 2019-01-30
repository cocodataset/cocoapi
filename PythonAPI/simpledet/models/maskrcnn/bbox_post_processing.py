import mxnet as mx
import numpy as np

from operator_py.nms import py_nms_wrapper

def multiclass_nms(nms, cls_score, bbox_xyxy, min_det_score, max_det_per_image):
    # remove background
    cls_score = cls_score[:, 1:]
    # TODO: the output shape of class_agnostic box is [n, 4], while class_aware box is [n, 4 * (1 + class)]
    bbox_xyxy = bbox_xyxy[:, 4:] if bbox_xyxy.shape[1] != 4 else bbox_xyxy
    num_class = cls_score.shape[1]

    cls_det = [np.empty((0, 6), dtype=np.float32) for _ in range(num_class)] # [x1, y1, x2, y2, score, cls]

    for cid in range(num_class):
        score = cls_score[:, cid]
        if bbox_xyxy.shape[1] != 4:
            _bbox_xyxy = bbox_xyxy[:, cid * 4:(cid + 1) * 4]
        else:
            _bbox_xyxy = bbox_xyxy
        valid_inds = np.where(score > min_det_score)[0]
        box = _bbox_xyxy[valid_inds]
        score = score[valid_inds]
        det = np.concatenate((box, score.reshape(-1, 1)), axis=1).astype(np.float32)
        det = nms(det)
        cls = np.full((det.shape[0], 1), cid, dtype=np.float32)
        cls_det[cid] = np.hstack((det, cls))

    cls_det = np.vstack([det for det in cls_det])
    scores = cls_det[:, -2]
    top_index = np.argsort(scores)[::-1][:max_det_per_image]
    return cls_det[top_index]


class BboxPostProcessingOperator(mx.operator.CustomOp):
    def __init__(self, max_det_per_image, min_det_score, nms_type, nms_thr):
        super(BboxPostProcessingOperator, self).__init__()
        self.max_det_per_image = max_det_per_image
        self.min_det_score = min_det_score
        self.nms_type = nms_type
        self.nms_thr = nms_thr

    def forward(self, is_train, req, in_data, out_data, aux):
        if self.nms_type == 'nms':
            nms = py_nms_wrapper(self.nms_thr)
        else:
            raise NotImplementedError

        cls_score = in_data[0].asnumpy()
        bbox_xyxy = in_data[1].asnumpy()

        cls_score_shape = cls_score.shape # (b, n, num_class_withbg)
        bbox_xyxy_shape = bbox_xyxy.shape # (b, n, 4) or (b, n, 4 * num_class_withbg)
        batch_image = cls_score_shape[0]
        num_bbox = cls_score_shape[1]
        num_class_withbg = cls_score_shape[2]

        post_score = np.zeros((batch_image, self.max_det_per_image, 1), dtype=np.float32)
        post_bbox_xyxy = np.zeros((batch_image, self.max_det_per_image, 4), dtype=np.float32)
        post_cls = np.full((batch_image, self.max_det_per_image, 1), -1, dtype=np.float32)

        for i, (per_image_cls_score, per_image_bbox_xyxy) in enumerate(zip(cls_score, bbox_xyxy)):
            cls_det = multiclass_nms(nms, per_image_cls_score, per_image_bbox_xyxy, \
                                     self.min_det_score, self.max_det_per_image)
            num_det = cls_det.shape[0]
            post_bbox_xyxy[i, :num_det] = cls_det[:, :4]
            post_score[i, :num_det] = cls_det[:, -2][:, np.newaxis] # convert to (n, 1)
            post_cls[i, :num_det] = cls_det[:, -1][:, np.newaxis] # convert to (n, 1)

        self.assign(out_data[0], req[0], post_score)
        self.assign(out_data[1], req[1], post_bbox_xyxy)
        self.assign(out_data[2], req[2], post_cls)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register("BboxPostProcessing")
class BboxPostProcessingProp(mx.operator.CustomOpProp):
    def __init__(self, max_det_per_image, min_det_score, nms_type, nms_thr):
        super(BboxPostProcessingProp, self).__init__(need_top_grad=False)
        self.max_det_per_image = int(max_det_per_image)
        self.min_det_score = float(min_det_score)
        self.nms_type = str(nms_type)
        self.nms_thr = float(nms_thr)

    def list_arguments(self):
        return ['cls_score', 'bbox_xyxy']

    def list_outputs(self):
        return ['post_score', 'post_bbox_xyxy', 'post_cls']

    def infer_shape(self, in_shape):
        cls_score_shape = in_shape[0] # (b, n, num_class_withbg)
        bbox_xyxy_shape = in_shape[1] # (b, n, 4) or (b, n, 4 * num_class_withbg)

        batch_image = cls_score_shape[0]

        post_score_shape = (batch_image, self.max_det_per_image, 1)
        post_bbox_xyxy_shape = (batch_image, self.max_det_per_image, 4)
        post_cls_shape = (batch_image, self.max_det_per_image, 1)

        return [cls_score_shape, bbox_xyxy_shape], \
               [post_score_shape, post_bbox_xyxy_shape, post_cls_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return BboxPostProcessingOperator(self.max_det_per_image, self.min_det_score, self.nms_type, self.nms_thr)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
