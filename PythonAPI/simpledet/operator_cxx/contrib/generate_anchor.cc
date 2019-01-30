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
 * \file generate_anchor.cc
 * \brief
 * \author Yanghao Li
*/

#include "./generate_anchor-inl.h"

namespace mxnet {
namespace op {

template<typename xpu>
class GenAnchorOp : public Operator{
 public:
  explicit GenAnchorOp(GenAnchorParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(req[gen_anchor::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> scores = in_data[gen_anchor::kClsProb].get<cpu, 4, real_t>(s);

    Tensor<cpu, 2> all_anchors = out_data[gen_anchor::kOut].get<cpu, 2, real_t>(s);

    int num_anchors = scores.size(1) / 2;
    int height = scores.size(2);
    int width = scores.size(3);

    // Generate anchors
    std::vector<float> base_anchor(4);
    base_anchor[0] = 0.0;
    base_anchor[1] = 0.0;
    base_anchor[2] = param_.feature_stride - 1.0;
    base_anchor[3] = param_.feature_stride - 1.0;
    CHECK_EQ(num_anchors, param_.ratios.info.size() * param_.scales.info.size());
    std::vector<float> anchors;
    utils::GenerateAnchors(base_anchor,
                           param_.ratios.info,
                           param_.scales.info,
                           &anchors);

    // Enumerate all shifted anchors
    for (index_t i = 0; i < num_anchors; ++i) {
      for (index_t j = 0; j < height; ++j) {
        for (index_t k = 0; k < width; ++k) {
          index_t index = j * (width * num_anchors) + k * (num_anchors) + i;
          all_anchors[index][0] = anchors[i * 4 + 0] + k * param_.feature_stride;
          all_anchors[index][1] = anchors[i * 4 + 1] + j * param_.feature_stride;
          all_anchors[index][2] = anchors[i * 4 + 2] + k * param_.feature_stride;
          all_anchors[index][3] = anchors[i * 4 + 3] + j * param_.feature_stride;
        }
      }
    }
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
    CHECK_EQ(in_grad.size(), 1);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> gscores = in_grad[gen_anchor::kClsProb].get<xpu, 4, real_t>(s);

    // can not assume the grad would be zero
    Assign(gscores, req[gen_anchor::kClsProb], 0);
  }

 private:
  GenAnchorParam param_;
};  // class GenAnchorOp

template<>
Operator *CreateOp<cpu>(GenAnchorParam param) {
  return new GenAnchorOp<cpu>(param);
}

Operator* GenAnchorProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(GenAnchorParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_GenAnchor, GenAnchorProp)
.describe("Generate region anchors")
.add_argument("cls_prob", "NDArray-or-Symbol", "Probability of how likely proposal is object.")
.add_arguments(GenAnchorParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
