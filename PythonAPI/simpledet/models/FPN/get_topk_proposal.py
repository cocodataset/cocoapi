"""
Assign Layer operator for FPN
"""

import mxnet as mx
import numpy as np
import numpy.random as npr
from distutils.util import strtobool
import pdb


class GetTopProposalOperator(mx.operator.CustomOp):
    def __init__(self, rpn_post_nms_top_n):
        super(GetTopProposalOperator, self).__init__()
        self._rpn_post_nms_top_n = int(rpn_post_nms_top_n)

    def forward(self, is_train, req, in_data, out_data, aux):
        all_rois = in_data[0].asnumpy()
        rois_scores = in_data[1].asnumpy()
        batch_images = all_rois.shape[0]
        rois = []
        for i in range(batch_images):
            all_rois_i = all_rois[i]
            rois_scores_i = rois_scores[i]
            all_rois_i = all_rois_i[np.argsort(-rois_scores_i[:, 0])]
            all_rois_i = all_rois_i[:self._rpn_post_nms_top_n]

            rois.append(all_rois_i)
        
        # pdb.set_trace()
        rois = np.array(rois, dtype=np.float32)

        for ind, val in enumerate([rois]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('get_top_proposal')
class GetTopProposalProp(mx.operator.CustomOpProp):
    def __init__(self, rpn_post_nms_top_n):
        super(GetTopProposalProp, self).__init__(need_top_grad=False)
        self._rpn_post_nms_top_n = int(rpn_post_nms_top_n)


    def list_arguments(self):
        return ['rois', 'rois_scores']

    def list_outputs(self):
        return ['rois']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        rois_scores_shape = in_shape[1]

        output_rois_shape = (rpn_rois_shape[0], self._rpn_post_nms_top_n, 4)

        return [rpn_rois_shape, rois_scores_shape], \
               [output_rois_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return GetTopProposalOperator(self._rpn_post_nms_top_n)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []