"""
Assign Layer operator for FPN
"""

import mxnet as mx
import numpy as np
import numpy.random as npr
from distutils.util import strtobool
import pdb


class AssignLayerFPNOperator(mx.operator.CustomOp):
    def __init__(self):
        super(AssignLayerFPNOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        all_rois = in_data[0].asnumpy()
        batch_images = all_rois.shape[0]

        rois = []
        rois_s4, rois_s8, rois_s16, rois_s32 = [[] for _ in range(4)]
        rcnn_feat_stride = [32, 16, 8, 4]
        for i in range(batch_images):
            rois_i = all_rois[i]
            rois_num, _ = rois_i.shape
            thresholds = [[np.inf, 448], [448, 224], [224, 112], [112, 0]]
            rois_area = np.sqrt((rois_i[:, 2] - rois_i[:, 0] + 1) * (rois_i[:, 3] - rois_i[:, 1] + 1))
            assign_levels = np.zeros(rois_num, dtype=np.uint8)
            for thresh, stride in zip(thresholds, rcnn_feat_stride):
                inds = np.logical_and(thresh[1] <= rois_area, rois_area < thresh[0])
                assign_levels[inds] = stride

            # assign rois to levels
            for idx, s in enumerate(rcnn_feat_stride):
                index = np.where(assign_levels == s)
                _rois = np.zeros(shape=(rois_num, 4), dtype=np.float32)
                _rois[index] = rois_i[index]
                if s == 4:
                    rois_s4.append(_rois)
                elif s == 8:
                    rois_s8.append(_rois)
                elif s == 16:
                    rois_s16.append(_rois)
                else:
                    rois_s32.append(_rois)
                #rois_on_levels.update({"stride%s" % s: _rois})
            rois.append(rois_i)
        
        # pdb.set_trace()
        rois = np.array(rois, dtype=np.float32)
        rois_s4 = np.array(rois_s4, dtype=np.float32)
        rois_s8 = np.array(rois_s8, dtype=np.float32)
        rois_s16 = np.array(rois_s16, dtype=np.float32)
        rois_s32 = np.array(rois_s32, dtype=np.float32)

        for ind, val in enumerate([rois, rois_s4, rois_s8, rois_s16, rois_s32]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)


@mx.operator.register('assign_layer_fpn')
class AssignLayerFPNProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(AssignLayerFPNProp, self).__init__(need_top_grad=False)


    def list_arguments(self):
        return ['rois']

    def list_outputs(self):
        return ['rois', 'rois_s4', 'rois_s8', 'rois_s16', 'rois_s32']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]

        output_rois_shape = rpn_rois_shape

        return [rpn_rois_shape], \
               [output_rois_shape, output_rois_shape, output_rois_shape, output_rois_shape, output_rois_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return AssignLayerFPNOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []