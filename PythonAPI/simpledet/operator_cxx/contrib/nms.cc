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
 * \file NMS.cc
 * \brief
 * \author Yanghao Li
*/

#include "./nms-inl.h"

//=====================
// NMS Utils
//=====================
namespace mxnet {
namespace op {
namespace utils {

struct ReverseArgsortCompl {
  const float *val_;
  explicit ReverseArgsortCompl(float *val)
    : val_(val) {}
  bool operator() (float i, float j) {
    return (val_[static_cast<index_t>(i)] >
            val_[static_cast<index_t>(j)]);
  }
};

// copy score and init order
inline void CopyScore(const mshadow::Tensor<cpu, 3>& dets,
                      mshadow::Tensor<cpu, 2> *score,
                      mshadow::Tensor<cpu, 2> *order) {
  for (index_t n = 0; n < dets.size(0); n++) {
    for (index_t i = 0; i < dets.size(1); i++) {
      (*score)[n][i] = dets[n][i][4];
      (*order)[n][i] = i;
    }
  }
}

// sort order array according to score
inline void ReverseArgsort(const mshadow::Tensor<cpu, 1>& score,
                           mshadow::Tensor<cpu, 1> *order) {
  ReverseArgsortCompl cmpl(score.dptr_);
  std::sort(order->dptr_, order->dptr_ + score.size(0), cmpl);
}

// reorder proposals according to order and keep the pre_nms_top_n proposals
// dets.size(0) == pre_nms_top_n
inline void ReorderProposals(const mshadow::Tensor<cpu, 2>& prev_dets,
                             const mshadow::Tensor<cpu, 1>& order,
                             const index_t pre_nms_top_n,
                             mshadow::Tensor<cpu, 2> *dets) {
  CHECK_EQ(dets->size(0), pre_nms_top_n);
  for (index_t i = 0; i < dets->size(0); i++) {
    const index_t index = order[i];
    for (index_t j = 0; j < dets->size(1); j++) {
      (*dets)[i][j] = prev_dets[index][j];
    }
  }
}

// greedily keep the max detections (already sorted)
inline void NonMaximumSuppression(const mshadow::Tensor<cpu, 2>& dets,
                                  const float thresh,
                                  const index_t post_nms_top_n,
                                  mshadow::Tensor<cpu, 1> *area,
                                  mshadow::Tensor<cpu, 1> *suppressed,
                                  mshadow::Tensor<cpu, 1> *keep,
                                  index_t *out_size) {
  CHECK_EQ(dets.shape_[1], 5) << "dets: [x1, y1, x2, y2, score]";
  CHECK_GT(dets.shape_[0], 0);
  CHECK_EQ(dets.CheckContiguous(), true);
  CHECK_EQ(area->CheckContiguous(), true);
  CHECK_EQ(suppressed->CheckContiguous(), true);
  CHECK_EQ(keep->CheckContiguous(), true);
  // calculate area
  for (index_t i = 0; i < dets.size(0); ++i) {
    (*area)[i] = (dets[i][2] - dets[i][0] + 1) *
                 (dets[i][3] - dets[i][1] + 1);
  }

  // calculate nms
  *out_size = 0;
  for (index_t i = 0; i < dets.size(0) && (*out_size) < post_nms_top_n; ++i) {
    float ix1 = dets[i][0];
    float iy1 = dets[i][1];
    float ix2 = dets[i][2];
    float iy2 = dets[i][3];
    float iarea = (*area)[i];

    if ((*suppressed)[i] > 0.0f) {
      continue;
    }

    (*keep)[(*out_size)++] = i;
    for (index_t j = i + 1; j < dets.size(0); j ++) {
      if ((*suppressed)[j] > 0.0f) {
        continue;
      }
      float xx1 = std::max(ix1, dets[j][0]);
      float yy1 = std::max(iy1, dets[j][1]);
      float xx2 = std::min(ix2, dets[j][2]);
      float yy2 = std::min(iy2, dets[j][3]);
      float w = std::max(0.0f, xx2 - xx1 + 1.0f);
      float h = std::max(0.0f, yy2 - yy1 + 1.0f);
      float inter = w * h;
      float ovr = inter / (iarea + (*area)[j] - inter);
      if (ovr > thresh) {
        (*suppressed)[j] = 1.0f;
      }
    }
  }
}

}  // namespace utils
}  // namespace op
}  // namespace mxnet


namespace mxnet {
namespace op {

template<typename xpu>
class NMSOp : public Operator{
 public:
  explicit NMSOp(NMSParam param) {
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
    CHECK_EQ(out_data.size(), 2);
    CHECK_GT(req.size(), 1);
    CHECK_EQ(req[nms::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 3> proposals = in_data[nms::kBBox].get<cpu, 3, real_t>(s);

    Tensor<cpu, 3> out = out_data[nms::kOut].get<cpu, 3, real_t>(s);
    Tensor<cpu, 3> out_score = out_data[nms::kScore].get<cpu, 3, real_t>(s);

    int nbatch = proposals.size(0);
    int count = proposals.size(1);
    int rpn_pre_nms_top_n = (param_.rpn_pre_nms_top_n > 0) ? param_.rpn_pre_nms_top_n : count;
    rpn_pre_nms_top_n = std::min(rpn_pre_nms_top_n, count);
    int rpn_post_nms_top_n = std::min(param_.rpn_post_nms_top_n, rpn_pre_nms_top_n);

    int workspace_size = nbatch * (2 * count + rpn_pre_nms_top_n * 5 + 3 * rpn_pre_nms_top_n);
    Tensor<cpu, 1> workspace = ctx.requested[nms::kTempSpace].get_space<cpu>(
      Shape1(workspace_size), s);
    int start = 0;
    Tensor<cpu, 3> workspace_pre_nms(workspace.dptr_ + start, Shape3(2, nbatch, count));
    start += nbatch * 2 * count;
    Tensor<cpu, 3> workspace_ordered_proposals(workspace.dptr_ + start,
                                               Shape3(nbatch, rpn_pre_nms_top_n, 5));
    start += nbatch * rpn_pre_nms_top_n * 5;
    Tensor<cpu, 3> workspace_nms(workspace.dptr_ + start, Shape3(3, nbatch, rpn_pre_nms_top_n));
    start += nbatch * 3 * rpn_pre_nms_top_n;
    CHECK_EQ(workspace_size, start) << workspace_size << " " << start << std::endl;

    Tensor<cpu, 2> score = workspace_pre_nms[0];
    Tensor<cpu, 2> order = workspace_pre_nms[1];

    utils::CopyScore(proposals,
                     &score,
                     &order);

    Tensor<cpu, 2> area = workspace_nms[0];
    Tensor<cpu, 2> suppressed = workspace_nms[1];
    Tensor<cpu, 2> keep = workspace_nms[2];

    for(int n = 0; n < nbatch; n++) {
      Tensor<cpu, 1> cur_order = order[n];
      Tensor<cpu, 1> cur_area = area[n];
      Tensor<cpu, 1> cur_keep = keep[n];
      Tensor<cpu, 1> cur_suppressed = suppressed[n];
      Tensor<cpu, 2> cur_workspace_ordered_proposals = workspace_ordered_proposals[n];
      if (!param_.already_sorted) {
          utils::ReverseArgsort(score[n],
                                &cur_order);
      }
      utils::ReorderProposals(proposals[n],
                              cur_order,
                              rpn_pre_nms_top_n,
                              &cur_workspace_ordered_proposals);
      index_t out_size = 0;
//      suppressed = 0;  // surprised!

      utils::NonMaximumSuppression(cur_workspace_ordered_proposals,
                                   param_.threshold,
                                   rpn_post_nms_top_n,
                                   &cur_area,
                                   &cur_suppressed,
                                   &cur_keep,
                                   &out_size);

      // fill in output rois
      for (index_t i = 0; i < out.size(1); ++i) {
//        // batch index 0
//        out[n][i][0] = n;
        if (i < rpn_pre_nms_top_n) {
          for (index_t j = 0; j < 4; ++j) {
            out[n][i][j] =  cur_workspace_ordered_proposals[i][j];
          }
        } else {
          for (index_t j = 0; j < 4; ++j) {
            out[n][i][j] = 0;
          }
        }
      }

      // fill in output score
      for (index_t i = 0; i < out_score.size(1); i++) {
        if (i < rpn_pre_nms_top_n) {
          index_t index = cur_keep[i];
          out_score[n][i][0] = cur_workspace_ordered_proposals[index][4];
        } else {
          index_t index = cur_keep[i % out_size];
          out_score[n][i][0] = cur_workspace_ordered_proposals[index][4];
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
    CHECK_EQ(in_grad.size(), 3);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 3> gbbox = in_grad[nms::kBBox].get<xpu, 3, real_t>(s);

    // can not assume the grad would be zero
    Assign(gbbox, req[nms::kBBox], 0);
  }

 private:
  NMSParam param_;
};  // class NMSOp

template<>
Operator *CreateOp<cpu>(NMSParam param) {
  return new NMSOp<cpu>(param);
}

Operator* NMSProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(NMSParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_NMS, NMSProp)
.describe("Apply NMS on proposals")
.add_argument("bbox", "NDArray-or-Symbol", "Proposals Predicted from RPN")
.add_arguments(NMSParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
