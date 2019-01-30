import numpy as np
import mxnet as mx

class SigmoidCELossMetric(mx.metric.EvalMetric):
    def __init__(self, name, output_names, label_names):
        super(SigmoidCELossMetric, self).__init__(name, output_names, label_names)

    def update(self, labels, preds):
        self.sum_metric += preds[0].mean().asscalar()
        self.num_inst += 1