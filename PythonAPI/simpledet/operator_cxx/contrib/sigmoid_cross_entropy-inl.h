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
 * \file sigmoid_cross_entropy-inl.h
 * \brief
 * \author Yuntao Chen
*/
#ifndef MXNET_OPERATOR_SIGMOID_CROSS_ENTROPY_INL_H_
#define MXNET_OPERATOR_SIGMOID_CROSS_ENTROPY_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../operator_common.h"

namespace mxnet {
namespace op {

namespace sigmoid_ce {
enum SigmoidCrossEntropyOpInputs {kData, kLabel};
enum SigmoidCrossEntropyOpOutputs {kOut, kLoss, kLossSum, kCount, kCountSum};
enum SigmoidCrossEntropyNormType {kNull, kValid};
enum SigmoidCrossEntropyOpResource {kTempSpace};
}  // namespace sigmoid_ce

struct SigmoidCrossEntropyParam : public dmlc::Parameter<SigmoidCrossEntropyParam> {
  float grad_scale;
  int normalization;
  DMLC_DECLARE_PARAMETER(SigmoidCrossEntropyParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scales the gradient by a float factor.");
    DMLC_DECLARE_FIELD(normalization)
    .add_enum("null", sigmoid_ce::kNull)
    .add_enum("valid", sigmoid_ce::kValid)
    .set_default(sigmoid_ce::kValid)
    .describe("Normalizes the gradient.");
  };
};

template<typename xpu, typename T>
class SigmoidCrossEntropyOp : public Operator {
 public:
  explicit SigmoidCrossEntropyOp(SigmoidCrossEntropyParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using std::cout;
    CHECK_EQ(in_data.size(), 2U) << "SigmoidCrossEntropy Input: [data, label]";
    CHECK_EQ(out_data.size(), 5U) << "SigmoidCrossEntropy Output: [output, loss, loss_sum, count, count_sum]";
    Stream<xpu> *s = ctx.get_stream<xpu>();

    int n = in_data[sigmoid_ce::kData].shape_[0];
    int k = in_data[sigmoid_ce::kData].shape_.Size() / n;
    Shape<2> s2 = Shape2(n, k);
    Tensor<xpu, 2, T> data = in_data[sigmoid_ce::kData].get_with_shape<xpu, 2, T>(s2, s);
    Tensor<xpu, 2, T> label = in_data[sigmoid_ce::kLabel].get_with_shape<xpu, 2, T>(s2, s);
    Tensor<xpu, 1, T> out = out_data[sigmoid_ce::kOut].FlatTo1D<xpu, T>(s);
    Tensor<xpu, 2, T> loss = out_data[sigmoid_ce::kLoss].get_with_shape<xpu, 2, T>(s2, s);
    Tensor<xpu, 1, T> loss_sum = out_data[sigmoid_ce::kLossSum].FlatTo1D<xpu, T>(s);
    Tensor<xpu, 2, T> count = out_data[sigmoid_ce::kCount].get_with_shape<xpu, 2, T>(s2, s);
    Tensor<xpu, 1, T> count_sum = out_data[sigmoid_ce::kCountSum].FlatTo1D<xpu, T>(s);

    SigmoidCrossEntropyForward(data, label, loss, loss_sum, count, count_sum, out, static_cast<T>(param_.grad_scale));
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
    CHECK_EQ(in_data.size(), 2U); // [data, label]
    CHECK_EQ(out_data.size(), 5U); // [out]
    CHECK_GE(in_grad.size(), 1U); // [d_data]
    CHECK_GE(req.size(), 1U); // [req_data]
    Stream<xpu> *s = ctx.get_stream<xpu>();

    int n = in_data[sigmoid_ce::kData].shape_[0];
    int k = in_data[sigmoid_ce::kData].shape_.Size() / n;
    Shape<2> s2 = Shape2(n, k);
    Tensor<xpu, 2, T> data = in_data[sigmoid_ce::kData].get_with_shape<xpu, 2, T>(s2, s);
    Tensor<xpu, 2, T> label = in_data[sigmoid_ce::kLabel].get_with_shape<xpu, 2, T>(s2, s);
    Tensor<xpu, 2, T> count = out_data[sigmoid_ce::kCount].get_with_shape<xpu, 2, T>(s2, s);
    Tensor<xpu, 1, T> count_sum = out_data[sigmoid_ce::kCountSum].FlatTo1D<xpu, T>(s);
    Tensor<xpu, 2, T> d_data = in_grad[sigmoid_ce::kData].get_with_shape<xpu, 2, T>(s2, s);

    SigmoidCrossEntropyBackward(data, label, d_data, count, count_sum, static_cast<T>(param_.grad_scale));
  }

 private:
  SigmoidCrossEntropyParam param_;
};  // class SigmoidCrossEntropyOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(SigmoidCrossEntropyParam param, int dtype);

#if DMLC_USE_CXX11
class SigmoidCrossEntropyProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  std::vector<std::string> ListOutputs() const {
    return {"output", "loss", "loss_sum", "count", "count_sum"};
  }

  int NumVisibleOutputs() const {
    return 1;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, label]";
    const TShape &dshape = in_shape->at(0);
    const TShape &lshape = in_shape->at(1);

    CHECK_GE(dshape.ndim(), 2U);
    CHECK_GE(lshape.ndim(), 2U);

    TShape oshape = Shape1(dshape[0]);
    
    out_shape->clear();
    out_shape->push_back(oshape); // out shape
    out_shape->push_back(dshape); // loss shape
    out_shape->push_back(oshape); // loss_sum shape
    out_shape->push_back(dshape); // count shape
    out_shape->push_back(oshape); // count_sum shape
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";

    out_type->clear();
    out_type->push_back(dtype);
    out_type->push_back(dtype);
    out_type->push_back(dtype);
    out_type->push_back(dtype);
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SigmoidCrossEntropyProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_SigmoidCrossEntropy";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[sigmoid_ce::kData], in_data[sigmoid_ce::kLabel], 
            out_data[sigmoid_ce::kCount], out_data[sigmoid_ce::kCountSum]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 protected:
  SigmoidCrossEntropyParam param_;
};  // class SigmoidCrossEntropyProp

#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SIGMOID_CROSS_ENTROPY_INL_H_
