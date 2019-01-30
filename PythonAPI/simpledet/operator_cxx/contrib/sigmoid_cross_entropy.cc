/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2018 by Contributors
 * \file sigmoid_cross_entropy.cc
 * \brief
 * \author Yuntao Chen
*/

#include "./sigmoid_cross_entropy-inl.h"

namespace mshadow {

template<typename T>
inline void SigmoidCrossEntropyForward(const Tensor<cpu, 2, T> &data,
                                       const Tensor<cpu, 2, T> &label,
                                       Tensor<cpu, 2, T> &loss,
                                       Tensor<cpu, 1, T> &loss_sum,
                                       Tensor<cpu, 2, T> &count,
                                       Tensor<cpu, 1, T> &count_sum,
                                       Tensor<cpu, 1, T> &out,
                                       T scale) {
  LOG(FATAL) << "NotImplemented";
}

template<typename T>
inline void SigmoidCrossEntropyBackward(const Tensor<cpu, 2, T> &data,
                                        const Tensor<cpu, 2, T> &label,
                                        Tensor<cpu, 2, T> &d_data,
                                        Tensor<cpu, 2, T> &count,
                                        Tensor<cpu, 1, T> &count_sum,
                                        T scale) {
  LOG(FATAL) << "NotImplemented";
}

}

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SigmoidCrossEntropyParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SigmoidCrossEntropyOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SigmoidCrossEntropyProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SigmoidCrossEntropyParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_SigmoidCrossEntropy, SigmoidCrossEntropyProp)
.describe(R"DOC(
Compute sigmoid activations followed by averaged binary cross entropy loss. The
target values may be in {-1, 0, 1}, where -1 indicates that the corresponding
sample should be ignored and {0, 1} correspond to the binary classes 0 and 1. By
default the loss is divided by the number of targets > -1 and then multiplied by
the `grad_scale` op argument. The divisive normalization may be disable by setting
the op argument `normalize` to 0 (the multiplication by `scale` still takes
effect).
This op fuses sigmoid and cross entropy for numerical stability in both forward
and gradient computation.
)DOC" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input array.")
.add_argument("label", "NDArray-or-Symbol", "Ground truth label.")
.add_arguments(SigmoidCrossEntropyParam::__FIELDS__());


}  // namespace op
}  // namespace mxnet
