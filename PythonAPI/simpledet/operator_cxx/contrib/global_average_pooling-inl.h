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
 * \file global_average_pooling-inl.h
 * \brief port from https://github.com/hujie-frank/SENet
 * \author Chenxia Han
*/

#ifndef MXNET_OPERATOR_GAP_INL_H_
#define MXNET_OPERATOR_GAP_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"

namespace mxnet {
namespace op {

namespace gap_enum {
enum GAPOpInputs {kData};
enum GAPOpOutputs {kOut};
enum GAPOpPadConventionType {kValid, kFull};
}  // namespace gap_enum

struct GAPParam : public dmlc::Parameter<GAPParam> {
  TShape kernel;
  TShape stride;
  TShape pad;
  int pool_type;
  int pooling_convention;
  DMLC_DECLARE_PARAMETER(GAPParam) {
    DMLC_DECLARE_FIELD(kernel).set_default(TShape())
    .enforce_nonzero()
    .describe("pooling kernel size: (y, x) or (d, y, x)");

    DMLC_DECLARE_FIELD(pooling_convention).set_default(gap_enum::kValid)
    .add_enum("full", gap_enum::kFull)
    .add_enum("valid", gap_enum::kValid)
    .describe("Pooling convention to be applied.");

    DMLC_DECLARE_FIELD(stride).set_default(TShape())
    .enforce_nonzero()
    .describe("stride: for pooling (y, x) or (d, y, x)");

    DMLC_DECLARE_FIELD(pad).set_default(TShape())
    .describe("pad for pooling: (y, x) or (d, y, x)");
  }
};

template<typename xpu, typename DType>
class GAPOp : public Operator {
 public:
  explicit GAPOp(GAPParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (param_.kernel.ndim() == 3) {
      LOG(FATAL) << "3D kernel not implemented";
    }

    // reset padding size for global pooling
    TShape padding = param_.pad;
    // TShape kernel = param_.kernel;
    padding[0] = padding[1] = 0;
    // kernel[0] = kernel[1] = 0;

    Tensor<xpu, 4, DType> data = in_data[gap_enum::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[gap_enum::kOut].get<xpu, 4, DType>(s);

    GAPForward(out, data);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(in_grad.size(), 1);
    // TODO(bing): remove pad (0,0)
    if (param_.kernel.ndim() == 3) {
      LOG(FATAL) << "3D kernel not implemented";
    }

    // reset padding size for global pooling
    TShape padding = param_.pad;
    padding[0] = padding[1] = 0;

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> grad = out_grad[gap_enum::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> input_grad = in_grad[gap_enum::kData].get<xpu, 4, DType>(s);

    GAPBackward(input_grad, grad);
  }

 private:
  GAPParam param_;
};  // class GAPOp

template<typename xpu>
Operator* CreateOp(GAPParam param, int dtype);


#if DMLC_USE_CXX11
class GAPProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    using namespace mshadow;
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 1U);
    const TShape &dshape = (*in_shape)[0];
    CHECK_GE(dshape.ndim(), 4U) << "Pooling: Input data should be 4D in (batch, channel, y, x) "
                               << "Or 5D in (batch, channel, d, y, x)";
    CHECK_LE(dshape.ndim(), 5U) << "Pooling: Input data should be 4D in (batch, channel, y, x) "
                               << "Or 5D in (batch, channel, d, y, x)";
    TShape oshape = dshape;
    if (dshape.ndim() ==  0) return false;
    if (dshape.ndim() == 4) {
      oshape[2] = 1;
      oshape[3] = 1;
    } else {
      oshape[2] = 1;
      oshape[3] = 1;
      oshape[4] = 1;
    }
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 1);
    int dtype = (*in_type)[0];

    if (dtype == -1) {
      LOG(FATAL) << "Input type to pooling is not specified.";
      return false;
    }

    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    GAPProp *prop_sym = new GAPProp();
    prop_sym->param_ = this->param_;
    return prop_sym;
  }

  std::string TypeString() const override {
    return "_contrib_GAP";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[gap_enum::kOut], in_data[gap_enum::kData],
            out_data[gap_enum::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
#if MXNET_USE_CUDNN == 1
    return {};
#else
    return {{in_data[gap_enum::kData], in_grad[gap_enum::kData]}};
#endif
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  GAPParam param_;
};  // class GAPProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_GAP_INL_H_
