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
 * \file decodebbox-inl.h
 * \brief DecodeBBox Operator
 * \author Ziyang Zhou, Chenxia Han
*/
#ifndef MXNET_OPERATOR_CONTRIB_DECODEBBOX_INL_H_
#define MXNET_OPERATOR_CONTRIB_DECODEBBOX_INL_H_

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

namespace decodebbox {
enum DecodeBBoxOpInputs {kRois, kBBoxPred, kImInfo};
enum DecodeBBoxOpOutputs {kOut};
}  // decodebbox

struct DecodeBBoxParam : public dmlc::Parameter<DecodeBBoxParam> {
  nnvm::Tuple<float> bbox_mean;
  nnvm::Tuple<float> bbox_std;
  bool class_agnostic;

  DMLC_DECLARE_PARAMETER(DecodeBBoxParam) {
    float tmp[] = {0.f, 0.f, 0.f, 0.f};
    DMLC_DECLARE_FIELD(bbox_mean).set_default(nnvm::Tuple<float>(tmp, tmp+4)).describe("Bounding box mean");
    tmp[0] = 0.1f; tmp[1] = 0.1f; tmp[2] = 0.2f; tmp[3] = 0.2f;
    DMLC_DECLARE_FIELD(bbox_std).set_default(nnvm::Tuple<float>(tmp, tmp+4)).describe("Bounding box std");
    DMLC_DECLARE_FIELD(class_agnostic).set_default(true)
    .describe("Whether use class agnostic");
  }
};

template<typename xpu>
Operator *CreateOp(DecodeBBoxParam param);

#if DMLC_USE_CXX11
class DecodeBBoxProp : public OperatorProperty {
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
    CHECK_EQ(in_shape->size(), 3) << "Input:[rois, bbox_pred, im_info]";
    const TShape &dshape = in_shape->at(decodebbox::kBBoxPred);

    const bool class_agnostic = param_.class_agnostic;
    TShape bbox_shape;
    if (class_agnostic) {
      const int nbatch = dshape[0];
      const int nrois = dshape[1];
      bbox_shape = Shape3(nbatch, nrois, 4);
    } else {
      bbox_shape = dshape;
    }

    out_shape->clear();
    aux_shape->clear();
    out_shape->push_back(bbox_shape);

    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new DecodeBBoxProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_DecodeBBox";
  }

  int NumOutputs() const override {
    return 1;
  }

  std::vector<std::string> ListArguments() const override {
    return {"rois", "bbox_pred", "im_info"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {};
  }

  // Operator* CreateOperator(Context ctx) const override;
  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  DecodeBBoxParam param_;
};  // class DecodeBBoxProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

#endif  //  MXNET_OPERATOR_CONTRIB_DECODEBBOX_INL_H_
