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
 * Copyright (c) 2015 by Contributors
 * \file roi_pooling-inl.h
 * \brief roi pooling operator and symbol
 * \author Kye-Hyeon Kim, Jian Guo
*/
#ifndef MXNET_OPERATOR_ROI_POOLING_V1_INL_H_
#define MXNET_OPERATOR_ROI_POOLING_V1_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./mshadow_op.h"
#include "./operator_common.h"


namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace roipoolv1 {
enum ROIPoolingOpInputs {kData, kBox};
enum ROIPoolingOpOutputs {kOut, kMaxIdx};
}  // roipoolv1

struct ROIPoolingParam_v1 : public dmlc::Parameter<ROIPoolingParam_v1> {
  TShape pooled_size;
  float spatial_scale;
  DMLC_DECLARE_PARAMETER(ROIPoolingParam_v1) {
    DMLC_DECLARE_FIELD(pooled_size)
    .set_expect_ndim(2).enforce_nonzero()
    .describe("ROI pooling output shape (h,w) ");
    DMLC_DECLARE_FIELD(spatial_scale).set_range(0.0, 1.0)
    .describe("Ratio of input feature map height (or w) to raw image height (or w). "
    "Equals the reciprocal of total stride in convolutional layers");
  }
};

template<typename xpu, typename DType>
class ROIPoolingOp_v1 : public Operator {
 public:
  explicit ROIPoolingOp_v1(ROIPoolingParam_v1 p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t expected = 2;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), expected);
    CHECK_EQ(out_data[roipoolv1::kOut].shape_[0], in_data[roipoolv1::kBox].shape_[0]);
    CHECK_EQ(out_data[roipoolv1::kMaxIdx].shape_[0], in_data[roipoolv1::kBox].shape_[0]);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[roipoolv1::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> bbox = in_data[roipoolv1::kBox].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[roipoolv1::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> max_idx = out_data[roipoolv1::kMaxIdx].get<xpu, 4, DType>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    CHECK_EQ(max_idx.CheckContiguous(), true);
    out = -FLT_MAX;
    max_idx = -1.0f;
    ROIPoolForward_v1(out, data, bbox, max_idx, param_.spatial_scale);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t expected = 2;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), expected);
    CHECK_EQ(out_grad[roipoolv1::kOut].shape_[0], in_data[roipoolv1::kBox].shape_[0]);
    CHECK_EQ(out_data[roipoolv1::kMaxIdx].shape_[0], in_data[roipoolv1::kBox].shape_[0]);
    CHECK_NE(req[roipoolv1::kData], kWriteInplace) <<
      "ROIPooling: Backward doesn't support kWriteInplace.";
    CHECK_NE(req[roipoolv1::kBox], kWriteInplace) <<
      "ROIPooling: Backward doesn't support kWriteInplace.";
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> grad_out = out_grad[roipoolv1::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> bbox = in_data[roipoolv1::kBox].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> max_idx = out_data[roipoolv1::kMaxIdx].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grad_in = in_grad[roipoolv1::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 2, DType> grad_roi = in_grad[roipoolv1::kBox].get<xpu, 2, DType>(s);
    CHECK_EQ(grad_out.CheckContiguous(), true);
    CHECK_EQ(bbox.CheckContiguous(), true);
    CHECK_EQ(max_idx.CheckContiguous(), true);
    CHECK_EQ(grad_in.CheckContiguous(), true);
    if (kAddTo == req[roipoolv1::kData] || kWriteTo == req[roipoolv1::kData]) {
      if (kWriteTo == req[roipoolv1::kData]) {
        grad_in = 0.0f;
      }
      ROIPoolBackwardAcc_v1(grad_in, grad_out, bbox, max_idx, param_.spatial_scale);
    }
    if (kWriteTo == req[roipoolv1::kBox]) {
      grad_roi = 0.0f;
    }
  }

 private:
  ROIPoolingParam_v1 param_;
};  // class ROIPoolingOp_v1

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(ROIPoolingParam_v1 param, int dtype);

#if DMLC_USE_CXX11
class ROIPoolingProp_v1 : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "rois"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "maxidx"};
  }

  int NumOutputs() const override {
    return 2;
  }

  int NumVisibleOutputs() const override {
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
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, rois]";

    // data: [batch_size, c, h, w]
    TShape dshape = in_shape->at(roipoolv1::kData);
    CHECK_EQ(dshape.ndim(), 4U) << "data should be a 4D tensor";

    // bbox: [num_rois, 5]
    TShape bshape = in_shape->at(roipoolv1::kBox);
    CHECK_EQ(bshape.ndim(), 2U) << "bbox should be a 2D tensor of shape [batch, 5]";
    CHECK_EQ(bshape[1], 5U) << "bbox should be a 2D tensor of shape [batch, 5]";

    // out: [num_rois, c, pooled_h, pooled_w]
    // max_idx: [num_rois, c, pooled_h, pooled_w]
    out_shape->clear();
    out_shape->push_back(
         Shape4(bshape[0], dshape[1], param_.pooled_size[0], param_.pooled_size[1]));
    out_shape->push_back(
         Shape4(bshape[0], dshape[1], param_.pooled_size[0], param_.pooled_size[1]));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 2U);
    int dtype = (*in_type)[0];
    CHECK_EQ(dtype, (*in_type)[1]);
    CHECK_NE(dtype, -1) << "Input must have specified type";

    out_type->clear();
    out_type->push_back(dtype);
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    ROIPoolingProp_v1* roi_pooling_sym = new ROIPoolingProp_v1();
    roi_pooling_sym->param_ = this->param_;
    return roi_pooling_sym;
  }

  std::string TypeString() const override {
    return "ROIPooling_v1";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[roipoolv1::kOut], in_data[roipoolv1::kBox], out_data[roipoolv1::kMaxIdx]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  ROIPoolingParam_v1 param_;
};  // class ROIPoolingProp_v1
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ROI_POOLING_V1_INL_H_
