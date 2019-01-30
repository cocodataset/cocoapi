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

/*
 * Axpy Layer
 *
 * Created on: May 1, 2017
 * Author: hujie
 */

/*!
 * Copyright (c) 2018 by Contributors
 * \file axpy-inl.h
 * \brief port from https://github.com/hujie-frank/SENet
 * \author Yuntao Chen
*/
#ifndef MXNET_OPERATOR_AXPY_INL_H_
#define MXNET_OPERATOR_AXPY_INL_H_

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

namespace axpy {
enum AxpyOpInputs {kScale, kX, kY};
enum AxpyOpOutputs {kOut};
enum AxpyOpResource {kTempSpace};
}  // namespace axpy

struct AxpyParam : public dmlc::Parameter<AxpyParam> {
  DMLC_DECLARE_PARAMETER(AxpyParam) {
  };
};

template<typename xpu, typename T>
class AxpyOp : public Operator {
 public:
  explicit AxpyOp(AxpyParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using std::cout;
    CHECK_EQ(in_data.size(), 3U) << "Axpy Input: [scale, x, y]";
    CHECK_EQ(out_data.size(), 1U) << "Axpy Output: [out]";
    Stream<xpu> *s = ctx.get_stream<xpu>();

    int n = in_data[axpy::kX].shape_[0];
    int c = in_data[axpy::kX].shape_[1];
    int h = in_data[axpy::kX].shape_[2];
    int w = in_data[axpy::kX].shape_[3];
    Shape<2> s2 = Shape2(n, c);
    Shape<4> s4 = Shape4(n, c, h, w);

    Tensor<xpu, 2, T> scale = in_data[axpy::kScale].get_with_shape<xpu, 2, T>(s2, s);
    Tensor<xpu, 4, T> x_data = in_data[axpy::kX].get_with_shape<xpu, 4, T>(s4, s);
    Tensor<xpu, 4, T> y_data = in_data[axpy::kY].get_with_shape<xpu, 4, T>(s4, s);

    Tensor<xpu, 1, T> out = out_data[axpy::kOut].FlatTo1D<xpu, T>(s);

    AxpyForwardLauncher(scale, x_data, y_data, out);
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
    CHECK_EQ(in_data.size(), 3U) << "Axpy Input: [scale, x, y]";
    Stream<xpu> *s = ctx.get_stream<xpu>();

    int n = in_data[axpy::kX].shape_[0];
    int c = in_data[axpy::kX].shape_[1];
    int h = in_data[axpy::kX].shape_[2];
    int w = in_data[axpy::kX].shape_[3];
    Shape<2> s2 = Shape2(n, c);
    Shape<4> s4 = Shape4(n, c, h, w);

    Tensor<xpu, 2, T> scale = in_data[axpy::kScale].get_with_shape<xpu, 2, T>(s2, s);
    Tensor<xpu, 4, T> x_data = in_data[axpy::kX].get_with_shape<xpu, 4, T>(s4, s);
    Tensor<xpu, 4, T> y_data = in_data[axpy::kY].get_with_shape<xpu, 4, T>(s4, s);

    Tensor<xpu, 2, T> scale_grad = in_grad[axpy::kScale].get_with_shape<xpu, 2, T>(s2, s);
    Tensor<xpu, 4, T> x_data_grad = in_grad[axpy::kX].get_with_shape<xpu, 4, T>(s4, s);
    Tensor<xpu, 4, T> y_data_grad = in_grad[axpy::kY].get_with_shape<xpu, 4, T>(s4, s);

    Tensor<xpu, 4, T> o_grad = out_grad[axpy::kOut].get_with_shape<xpu, 4, T>(s4, s);

    AxpyBackwardLauncher(scale, x_data, y_data, scale_grad, x_data_grad, y_data_grad, o_grad, s);
  }

 private:
  AxpyParam param_;
};  // class AxpyOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(AxpyParam param, int dtype);

#if DMLC_USE_CXX11
class AxpyProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"scale", "x", "y"};
  }

  std::vector<std::string> ListOutputs() const {
    return {"output"};
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
    CHECK_EQ(in_shape->size(), 3U) << "Input:[scale, x, y]";

    TShape dshape = in_shape->at(axpy::kX);
    TShape oshape = dshape;
    
    out_shape->clear();
    out_shape->push_back(oshape); // out shape
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 3U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";

    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new AxpyProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_Axpy";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[axpy::kScale], in_data[axpy::kX], in_data[axpy::kY], out_grad[axpy::kOut]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 protected:
  AxpyParam param_;
};  // class AxpyProp

#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_AXPY_INL_H_
