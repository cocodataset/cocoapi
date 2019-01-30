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
 * \file nms-inl.h
 * \brief NMS Operator
 * \author Yanghao Li
*/
#ifndef MXNET_OPERATOR_CONTRIB_NMS_INL_H_
#define MXNET_OPERATOR_CONTRIB_NMS_INL_H_

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

namespace nms {
enum NMSOpInputs {kBBox};
enum NMSOpOutputs {kOut, kScore};
enum NMSForwardResource {kTempSpace};
}  // nms

struct NMSParam : public dmlc::Parameter<NMSParam> {
  int rpn_pre_nms_top_n;
  int rpn_post_nms_top_n;
  float threshold;
  bool output_score;
  bool already_sorted;
  uint64_t workspace;

  DMLC_DECLARE_PARAMETER(NMSParam) {
    DMLC_DECLARE_FIELD(rpn_pre_nms_top_n).set_default(6000)
    .describe("Number of top scoring boxes to keep before applying NMS to RPN proposals");
    DMLC_DECLARE_FIELD(rpn_post_nms_top_n).set_default(300)
    .describe("Number of top scoring boxes to keep after applying NMS to RPN proposals");
    DMLC_DECLARE_FIELD(threshold).set_default(0.7)
    .describe("NMS value, below which to suppress.");
    DMLC_DECLARE_FIELD(output_score).set_default(false)
    .describe("Add score to outputs");
    DMLC_DECLARE_FIELD(already_sorted).set_default(false)
    .describe("if input rois have been sorted by confidence");
    DMLC_DECLARE_FIELD(workspace).set_default(256)
    .describe("Workspace for NMS in MB, default to 256");
  }
};

template<typename xpu>
Operator *CreateOp(NMSParam param);

#if DMLC_USE_CXX11
class NMSProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 1) << "Input:[bbox]";
    const TShape &dshape = in_shape->at(nms::kBBox);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    // output
    out_shape->push_back(Shape3(dshape[0], param_.rpn_post_nms_top_n, 4));
    // score
    out_shape->push_back(Shape3(dshape[0], param_.rpn_post_nms_top_n, 1));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new NMSProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_NMS";
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

  int NumVisibleOutputs() const override {
    if (param_.output_score) {
      return 2;
    } else {
      return 1;
    }
  }

  int NumOutputs() const override {
    return 2;
  }

  std::vector<std::string> ListArguments() const override {
    return {"rois"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "score"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  NMSParam param_;
};  // class NMSProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  //  MXNET_OPERATOR_CONTRIB_NMS_INL_H_
