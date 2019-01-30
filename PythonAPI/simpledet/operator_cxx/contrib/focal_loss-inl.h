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
 * \file focal_loss-inl.h
 * \brief FocalLoss Operator
 * \author Chenxia Han
*/
#ifndef MXNET_OPERATOR_CONTRIB_FOCAL_LOSS_INL_H_
#define MXNET_OPERATOR_CONTRIB_FOCAL_LOSS_INL_H_

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
#include "../tensor/control_flow_op.h"
#include "../tensor/indexing_op.h"

namespace mxnet {
namespace op {

namespace focal_loss_enum {
enum FocalLossOpInputs {kData, kLabel};
enum FocalLossOpOutputs {kOut};
enum FOcalLossOpType {kNull, kBatch, kValid};
enum FocalLossOpResource {kTempSpace};
}  // namespace focal_loss_enum

struct FocalLossParam : public dmlc::Parameter<FocalLossParam> {
  float alpha, gamma;
  float grad_scale;
  int normalization;
  uint64_t workspace;
  DMLC_DECLARE_PARAMETER(FocalLossParam) {
    DMLC_DECLARE_FIELD(alpha).set_default(0.25f)
    .describe("Alpha for focal loss");
    DMLC_DECLARE_FIELD(gamma).set_default(2.0f)
    .describe("Gamma for focal loss");
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Gradient scale as a supplement to unary and binary operators");
    DMLC_DECLARE_FIELD(normalization)
    .add_enum("null", focal_loss_enum::kNull)
    .add_enum("batch", focal_loss_enum::kBatch)
    .add_enum("valid", focal_loss_enum::kValid)
    .set_default(focal_loss_enum::kNull)
    .describe("If this is set to null, the output gradient will not be normalized. "
              "If this is set to batch, the output gradient will be divided by the batch size. "
              "If this is set to valid, the output gradient will be divided by the number of "
              "valid input elements.");
    DMLC_DECLARE_FIELD(workspace).set_default(256)
    .describe("Workspace for focal loss in MB, default to 256");
  }
};

template<typename xpu, typename DType>
class FocalLossOp : public Operator{
 public:
  explicit FocalLossOp(FocalLossParam param) {
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
    CHECK_EQ(req[focal_loss_enum::kOut], kWriteTo);

    /*
     * data:    (nbatch, nbox, nclass)
     * label:   (nbatch, nbox)
     * out:     (nbatch, nbox, nclass)
     */
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> data = in_data[focal_loss_enum::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[focal_loss_enum::kOut].FlatTo2D<xpu, DType>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);

    Assign(out, req[focal_loss_enum::kOut], F<mshadow_op::sigmoid>(data));
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
    using namespace mxnet_op;
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(in_grad.size(), 2U);
    CHECK_EQ(req[focal_loss_enum::kData], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> label = in_data[focal_loss_enum::kLabel].get<xpu, 2, DType>(s);
    Tensor<xpu, 3, DType> out = out_data[focal_loss_enum::kOut].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> gdata = in_grad[focal_loss_enum::kData].get<xpu, 3, DType>(s);
    CHECK_EQ(label.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    CHECK_EQ(gdata.CheckContiguous(), true);

    uint64_t WORKSPACE_LIMIT = 1024 * 1024 * param_.workspace;
    Tensor<xpu, 1, uint8_t> workspace = ctx.requested[focal_loss_enum::kTempSpace]
      .get_space_typed<xpu, 1, uint8_t>(Shape1(WORKSPACE_LIMIT), s);
    uint64_t allocated_bytes = 0ULL;

    int nbatch = gdata.size(0);
    int nbox = gdata.size(1);
    int nclass = gdata.size(2);

    Tensor<xpu, 3, DType> positive(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), out.shape_, s);
    allocated_bytes += positive.shape_.Size() * sizeof(DType);
    CHECK_LT(allocated_bytes, WORKSPACE_LIMIT) << "Allocating more memory than workspace limit";

    Tensor<xpu, 3, DType> negative(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), out.shape_, s);
    allocated_bytes += negative.shape_.Size() * sizeof(DType);
    CHECK_LT(allocated_bytes, WORKSPACE_LIMIT) << "Allocating more memory than workspace limit";

    Tensor<xpu, 3, DType> one_hot(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), out.shape_, s);
    allocated_bytes += one_hot.shape_.Size() * sizeof(DType);
    CHECK_LT(allocated_bytes, WORKSPACE_LIMIT) << "Allocating more memory than workspace limit";

    Tensor<xpu, 2, DType> label_tmp(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), Shape2(nbatch, nbox), s);
    allocated_bytes += label_tmp.shape_.Size() * sizeof(DType);
    CHECK_LT(allocated_bytes, WORKSPACE_LIMIT) << "Allocating more memory than workspace limit";

    Tensor<xpu, 3, DType> grad(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), out.shape_, s);
    allocated_bytes += grad.shape_.Size() * sizeof(DType);
    CHECK_LT(allocated_bytes, WORKSPACE_LIMIT) << "Allocating more memory than workspace limit";

    Tensor<xpu, 3, DType> ignore_index(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), out.shape_, s);
    allocated_bytes += ignore_index.shape_.Size() * sizeof(DType);
    CHECK_LT(allocated_bytes, WORKSPACE_LIMIT) << "Allocating more memory than workspace limit";

    Tensor<xpu, 3, DType> one_bc(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), out.shape_, s);
    allocated_bytes += one_bc.shape_.Size() * sizeof(DType);
    CHECK_LT(allocated_bytes, WORKSPACE_LIMIT) << "Allocating more memory than workspace limit";

    Tensor<xpu, 1, DType> temp(reinterpret_cast<DType*>(workspace.dptr_ + allocated_bytes), Shape1(1), s);
    allocated_bytes += temp.shape_.Size() * sizeof(DType);
    CHECK_LT(allocated_bytes, WORKSPACE_LIMIT) << "Allocating more memory than workspace limit";

    float alpha = param_.alpha;
    float gamma = param_.gamma;
    const ScalarExp<DType> one_expr(1.f);
    const ScalarExp<DType> alpha_expr(alpha);
    const ScalarExp<DType> gamma_expr(gamma);
    const ScalarExp<DType> eps_expr(1e-14);
    /*
     * positive = self.alpha * (1 - pred) ** self.gamma
     *            * (self.gamma * pred * mx.nd.log(pred + eps) + pred - 1)
     * negative = - (1 - self.alpha) * pred ** self.gamma
     *            * (self.gamma * (1 - pred) * mx.nd.log((1 - pred) + eps) - pred)
    */
    positive = alpha_expr * F<mshadow_op::power>(one_expr - out, gamma_expr)
               * (gamma_expr * out * F<mshadow_op::log>(out + eps_expr) + out - one_expr);
    negative = F<mshadow_op::negation>((one_expr - alpha_expr)
               * F<mshadow_op::power>(out, gamma_expr)
               * (gamma_expr * (one_expr - out)
                  * F<mshadow_op::log>(one_expr - out + eps_expr) - out));
    label_tmp = label - one_expr;
    one_hot = 0.f;
    Kernel<mxnet::op::one_hot<kWriteTo>, xpu>::Launch(s, label_tmp.shape_.Size(),
      one_hot.dptr_, label_tmp.dptr_, nclass, static_cast<DType>(1.0));
    Kernel<mxnet::op::where<kWriteTo>, xpu>::Launch(s, grad.shape_.Size(),
      grad.dptr_, one_hot.dptr_, positive.dptr_, negative.dptr_);

    ignore_index = broadcast_with_axis(F<mshadow_op::eq>(label, ScalarExp<DType>(-1.f)), 1, nclass);
    temp = ScalarExp<DType>(0.f);
    one_bc = broadcast_scalar(temp, one_bc.shape_);
    Kernel<mxnet::op::where<kWriteTo>, xpu>::Launch(s, grad.shape_.Size(),
      grad.dptr_, ignore_index.dptr_, one_bc.dptr_, grad.dptr_);

    if (param_.normalization == focal_loss_enum::kValid) {
      temp = sumall_except_dim<0>(reduce_keepdim<red::sum, false>(
        F<mshadow_op::le>(ScalarExp<DType>(1.f), label), 0));
      temp = F<mshadow_op::plus>(temp, ScalarExp<DType>(1.f));
      temp = F<mshadow_op::maximum>(ScalarExp<DType>(1.f), temp);
      Assign(gdata, req[focal_loss_enum::kData],
        grad * ScalarExp<DType>(param_.grad_scale) / broadcast<0>(broadcast_keepdim(
        temp, 0, grad.shape_[0]), grad.shape_));
    } else if (param_.normalization == focal_loss_enum::kBatch) {
      Assign(gdata, req[focal_loss_enum::kData],
        grad * ScalarExp<DType>(param_.grad_scale / grad.shape_[0]));
    } else {
      Assign(gdata, req[focal_loss_enum::kData], grad * ScalarExp<DType>(param_.grad_scale));
    }
  }

 private:
  FocalLossParam param_;
};  // class FocalLossOp

template<typename xpu>
Operator *CreateOp(FocalLossParam param, int dtype);

#if DMLC_USE_CXX11
class FocalLossProp : public OperatorProperty {
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
    const TShape &dshape = in_shape->at(focal_loss_enum::kData);

    const int nbatch = dshape[0];
    const int nbox = dshape[1];
    const int nclass = dshape[2];

    auto data_shape = Shape3(nbatch, nbox, nclass);
    auto label_shape = Shape2(nbatch, nbox);

    SHAPE_ASSIGN_CHECK(*in_shape, focal_loss_enum::kData, data_shape);
    SHAPE_ASSIGN_CHECK(*in_shape, focal_loss_enum::kLabel, label_shape);

    out_shape->clear();
    // output
    out_shape->push_back(in_shape->at(focal_loss_enum::kData));
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
    auto ptr = new FocalLossProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_FocalLoss";
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
    return {in_data[focal_loss_enum::kLabel], out_data[focal_loss_enum::kOut]};
  }

  std::vector<ResourceRequest> BackwardResource(
    const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  /*
  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[focal_loss_enum::kData], out_data[focal_loss_enum::kOut]}};
  }
  */

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  FocalLossParam param_;
};  // class FocalLossProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  //  MXNET_OPERATOR_CONTRIB_FOCAL_LOSS_INL_H_
