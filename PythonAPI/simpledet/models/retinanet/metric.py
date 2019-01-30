import numpy as np
import mxnet as mx

class FGAccMetric(mx.metric.EvalMetric):
    def __init__(self, name, output_names, label_names):
        super(FGAccMetric, self).__init__(name, output_names, label_names)

    def update(self, labels, preds):
        pred = preds[0]
        label = labels[0]

        # pred (b, p, c) , c is 0 ~ CLASSNUM-1
        pred_label = mx.nd.argmax(pred,axis=2)
        # pred_label = mx.nd.argmax_channel(pred)
        pred_label = pred_label.asnumpy().astype('int32')
        # label (b, p)
        label = label.asnumpy().astype('int32')

        keep_inds = np.where(label >= 1)
        pred_label = pred_label[keep_inds]
        # label is 1 ~ CLASSNUM , so label = label - 1
        label = label[keep_inds] - 1

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)
