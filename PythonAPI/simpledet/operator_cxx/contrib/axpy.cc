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
 * \file axpy.cc
 * \brief port from https://github.com/hujie-frank/SENet
 * \author Yuntao Chen
*/
#include "./axpy-inl.h"

namespace mshadow {
template <typename Dtype>
inline void AxpyForwardLauncher(const Tensor<cpu, 2, Dtype> &scale_data,
                                const Tensor<cpu, 4, Dtype> &x_data,
                                const Tensor<cpu, 4, Dtype> &y_data,
                                const Tensor<cpu, 1, Dtype> &out) {
    LOG(FATAL) << "NotImplemented";
}

template <typename Dtype>
inline void AxpyBackwardLauncher(const Tensor<cpu, 2, Dtype> &scale_data,
                                 const Tensor<cpu, 4, Dtype> &x_data,
                                 const Tensor<cpu, 4, Dtype> &y_data,
                                 const Tensor<cpu, 2, Dtype> &scale_grad,
                                 const Tensor<cpu, 4, Dtype> &x_grad,
                                 const Tensor<cpu, 4, Dtype> &y_grad,
                                 const Tensor<cpu, 4, Dtype> &out_grad,
                                 Stream<cpu> *s) {
    LOG(FATAL) << "NotImplemented";
}
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(AxpyParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new AxpyOp<cpu, DType>(param);
  });
  return op;
}

Operator *AxpyProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(AxpyParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_Axpy, AxpyProp)
.describe(R"code(Accelerate Squeeze and Excitation Network)code" ADD_FILELINE)
.add_argument("scale", "NDArray-or-Symbol", "channel scaling factor")
.add_argument("x", "NDArray-or-Symbol", "resnet increase output")
.add_argument("y", "NDArray-or-Symbol", "resnet shortcut output")
.add_arguments(AxpyParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
