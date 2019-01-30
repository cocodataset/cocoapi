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
 * \file bbox_norm-inl.h
 * \brief BBoxNorm Operator
 * \author Chenxia Han
*/
#ifndef MXNET_OPERATOR_CONTRIB_BBOX_NORM_INL_H_
#define MXNET_OPERATOR_CONTRIB_BBOX_NORM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <cmath>
#include <cstring>
#include <iostream>
#include "../operator_common.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

namespace bbox_norm {
enum BBoxNormOpInputs {kData, kLabel};
enum BBoxNormOpOutputs {kOut};
enum BBoxNormOpType {kNull, kBatch, kValid};
enum BBoxNormOpResource {kTempSpace};
}  // bbox_norm

struct BBoxNormParam : public dmlc::Parameter<BBoxNormParam> {
  int normalization;
  DMLC_DECLARE_PARAMETER(BBoxNormParam) {
    DMLC_DECLARE_FIELD(normalization)
    .add_enum("null", bbox_norm::kNull)
    .add_enum("batch", bbox_norm::kBatch)
    .add_enum("valid", bbox_norm::kValid)
    .set_default(bbox_norm::kValid)
    .describe("If this is set to null, the output gradient will not be normalized."
              "If this is set to batch, the output gradient will be divided by the batch size. "
              "If this is set to valid, the output gradient will be divided by the number of "
              "valid input elements.");
  }
};

template<typename xpu, typename DType>
class BBoxNormOp : public Operator{
 public:
  explicit BBoxNormOp(BBoxNormParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    CHECK_EQ(req[bbox_norm::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    /*
     * data:    (nbatch, 4 * nanchor, npos)
     * label:   (nbatch, nanchor * npos)
     * out:     (nbatch, 4 * nanchor, npos)
     */
    Tensor<xpu, 2, DType> data = in_data[bbox_norm::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[bbox_norm::kOut].FlatTo2D<xpu, DType>(s);

    Assign(out, req[bbox_norm::kOut], F<mshadow_op::identity>(data))
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_grad.size(), 2U);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> label = in_data[bbox_norm::kLabel].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> gdata  = in_grad[bbox_norm::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> glabel = in_grad[bbox_norm::kLabel].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> gout = out_grad[bbox_norm::kOut].FlatTo2D<xpu, DType>(s);

    // gdata = gout / max(sum(label >= 1), 1)
    Tensor<xpu, 1, DType> temp = ctx.requested[bbox_norm::kTempSpace]
      .get_space_typed<xpu, 1, DType>(mshadow::Shape1(1), s);
    temp = sumall_except_dim<0>(reduce_keepdim<red::sum, false>(
      F<mshadow_op::le>(ScalarExp<DType>(1.f), label), 0));
    temp = F<mshadow_op::plus>(temp, ScalarExp<DType>(1.f));
    temp = F<mshadow_op::maximum>(ScalarExp<DType>(1.f), temp);

    Assign(gdata, req[bbox_norm::kData],
      gout / broadcast<0>(broadcast_keepdim(
      temp, 0, gout.shape_[0]), gout.shape_));

    Assign(glabel, req[bbox_norm::kLabel], 0);
  }

 private:
  BBoxNormParam param_;
};  // class BBoxNormOp

template<typename xpu>
Operator *CreateOp(BBoxNormParam param, int dtype);

#if DMLC_USE_CXX11
class BBoxNormProp : public OperatorProperty {
 public:
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
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, label]";
    const TShape &dshape = in_shape->at(bbox_norm::kData);

    const int nbatch = dshape[0];
    const int nanchor = dshape[1] / 4;
    const int npos = dshape[2];

    auto data_shape = Shape3(nbatch, 4 * nanchor, npos);
    auto label_shape = Shape2(nbatch, nanchor * npos);

    SHAPE_ASSIGN_CHECK(*in_shape, bbox_norm::kData, data_shape);
    SHAPE_ASSIGN_CHECK(*in_shape, bbox_norm::kLabel, label_shape);

    out_shape->clear();
    // output
    out_shape->push_back(in_shape->at(bbox_norm::kData));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 2U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (size_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new BBoxNormProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_BBoxNorm";
  }

  int NumOutputs() const override {
    return 1;
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[bbox_norm::kLabel], out_grad[bbox_norm::kOut]};
  }

  std::vector<ResourceRequest> BackwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  /*
  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[bbox_norm::kData], out_data[bbox_norm::kOut]}};
  }
  */

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  BBoxNormParam param_;
};  // class BBoxNormProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  //  MXNET_OPERATOR_CONTRIB_BBOX_NORM_INL_H_
