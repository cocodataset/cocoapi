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
 * \file generate_proposal-inl.h
 * \brief GenerateProposal Operator
 * \author Piotr Teterwak, Bing Xu, Jian Guo, Pengfei Chen, Yuntao Chen, Yanghao Li
*/
#ifndef MXNET_OPERATOR_CONTRIB_GENERATE_PROPOSAL_INL_H_
#define MXNET_OPERATOR_CONTRIB_GENERATE_PROPOSAL_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <cstring>
#include <iostream>
#include "../operator_common.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

namespace gen_proposal {
enum GenProposalOpInputs {kClsProb, kBBoxPred, kImInfo, kAnchor};
enum GenProposalOpOutputs {kOut, kScore};
enum GenProposalForwardResource {kTempSpace};
}  // gen_proposal

struct GenProposalParam : public dmlc::Parameter<GenProposalParam> {
  int rpn_pre_nms_top_n;
  int rpn_min_size;
  int feature_stride;
  bool iou_loss;
  uint64_t workspace;

  DMLC_DECLARE_PARAMETER(GenProposalParam) {
    DMLC_DECLARE_FIELD(rpn_pre_nms_top_n).set_default(6000)
    .describe("Number of top scoring boxes to keep after applying NMS to RPN proposals");
    DMLC_DECLARE_FIELD(rpn_min_size).set_default(16)
    .describe("Minimum height or width in proposal");
    DMLC_DECLARE_FIELD(feature_stride).set_default(16)
    .describe("The size of the receptive field each unit in the convolution layer of the rpn,"
              "for example the product of all stride's prior to this layer.");
    DMLC_DECLARE_FIELD(iou_loss).set_default(false)
    .describe("Usage of IoU Loss");
    DMLC_DECLARE_FIELD(workspace).set_default(256)
    .describe("Workspace for proposal in MB, default to 256");
  }
};

template<typename xpu>
Operator *CreateOp(GenProposalParam param);

#if DMLC_USE_CXX11
class GenProposalProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 4) << "Input:[cls_prob, bbox_pred, im_info, anchors]";
    const TShape &dshape = in_shape->at(gen_proposal::kClsProb);
    if (dshape.ndim() == 0) return false;
    Shape<4> bbox_pred_shape;
    bbox_pred_shape = Shape4(dshape[0], dshape[1] * 2, dshape[2], dshape[3]);
    SHAPE_ASSIGN_CHECK(*in_shape, gen_proposal::kBBoxPred,
                       bbox_pred_shape);
    Shape<2> im_info_shape;
    im_info_shape = Shape2(dshape[0], 3);
    SHAPE_ASSIGN_CHECK(*in_shape, gen_proposal::kImInfo, im_info_shape);
    Shape<2> anchors_shape;
    anchors_shape = Shape2(dshape[2] * dshape[3] * dshape[1] / 2,  4);
    SHAPE_ASSIGN_CHECK(*in_shape, gen_proposal::kAnchor, anchors_shape);
    out_shape->clear();
    // output
    out_shape->push_back(Shape3(dshape[0], param_.rpn_pre_nms_top_n, 5));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new GenProposalProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_GenProposal";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {};
  }

  int NumOutputs() const override {
    return 1;
  }

  std::vector<std::string> ListArguments() const override {
    return {"cls_prob", "bbox_pred", "im_info", "anchors"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  GenProposalParam param_;
};  // class GenProposalProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  //  MXNET_OPERATOR_CONTRIB_GENERATE_PROPOSAL_INL_H_
