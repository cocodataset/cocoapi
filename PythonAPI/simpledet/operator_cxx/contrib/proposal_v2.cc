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
 * \file proposal.cc
 * \brief proposal op for SNIP
 * \author Piotr Teterwak, Bing Xu, Jian Guo, Yuntao Chen, Yanghao Li
*/

#include "./proposal_v2-inl.h"

//============================
// Bounding Box Transform Utils
//============================
namespace mxnet {
namespace op {
namespace proposal_v2_utils {

// bbox prediction and clip to the image borders
inline void BBoxTransformInv(const mshadow::Tensor<cpu, 3>& boxes,
                             const mshadow::Tensor<cpu, 4>& deltas,
                             const mshadow::Tensor<cpu, 2>& im_info, 
                             const int feature_stride,
                             mshadow::Tensor<cpu, 3> *out_pred_boxes) {
  CHECK_GE(boxes.size(1), 4);
  CHECK_GE(out_pred_boxes->size(1), 4);
  int nbatch = deltas.size(0);
  int anchors = deltas.size(1)/4;
  int heights = deltas.size(2);
  int widths = deltas.size(3);

  for (int n = 0; n < nbatch; ++n) {
    int real_height = static_cast<int>(im_info[n][0] / feature_stride);
    int real_width = static_cast<int>(im_info[n][1] / feature_stride);
    float im_height = im_info[n][0];
    float im_width = im_info[n][1];

    for (int a = 0; a < anchors; ++a) {
      for (int h = 0; h < heights; ++h) {
        for (int w = 0; w < widths; ++w) {
          index_t index = h * (widths * anchors) + w * (anchors) + a;
          float width = boxes[n][index][2] - boxes[n][index][0] + 1.0;
          float height = boxes[n][index][3] - boxes[n][index][1] + 1.0;
          float ctr_x = boxes[n][index][0] + 0.5 * (width - 1.0);
          float ctr_y = boxes[n][index][1] + 0.5 * (height - 1.0);

          float dx = deltas[n][a*4 + 0][h][w];
          float dy = deltas[n][a*4 + 1][h][w];
          float dw = deltas[n][a*4 + 2][h][w];
          float dh = deltas[n][a*4 + 3][h][w];

          float pred_ctr_x = dx * width + ctr_x;
          float pred_ctr_y = dy * height + ctr_y;
          float pred_w = exp(dw) * width;
          float pred_h = exp(dh) * height;

          float pred_x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
          float pred_y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
          float pred_x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
          float pred_y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);

          pred_x1 = std::max(std::min(pred_x1, im_width - 1.0f), 0.0f);
          pred_y1 = std::max(std::min(pred_y1, im_height - 1.0f), 0.0f);
          pred_x2 = std::max(std::min(pred_x2, im_width - 1.0f), 0.0f);
          pred_y2 = std::max(std::min(pred_y2, im_height - 1.0f), 0.0f);

          (*out_pred_boxes)[n][index][0] = pred_x1;
          (*out_pred_boxes)[n][index][1] = pred_y1;
          (*out_pred_boxes)[n][index][2] = pred_x2;
          (*out_pred_boxes)[n][index][3] = pred_y2;

          if (h >= real_height || w >= real_width) {
            (*out_pred_boxes)[n][index][4] = -1.0;
          }
        }
      }
    }
  }
}

// iou prediction and clip to the image border
inline void IoUTransformInv(const mshadow::Tensor<cpu, 3>& boxes,
                            const mshadow::Tensor<cpu, 4>& deltas,
                            const mshadow::Tensor<cpu, 2>& im_info, 
                            const int feature_stride,
                            mshadow::Tensor<cpu, 3> *out_pred_boxes) {
  CHECK_GE(boxes.size(2), 4);
  CHECK_GE(out_pred_boxes->size(2), 4);
  int nbatch = deltas.size(0);
  int anchors = deltas.size(1)/4;
  int heights = deltas.size(2);
  int widths = deltas.size(3);
  for (int n = 0; n < nbatch; ++n){
    int real_height = static_cast<int>(im_info[n][0] / feature_stride);
    int real_width = static_cast<int>(im_info[n][1] / feature_stride);
    float im_height = im_info[n][0];
    float im_width = im_info[n][1];
    for (int a = 0; a < anchors; ++a) {
      for (int h = 0; h < heights; ++h) {
        for (int w = 0; w < widths; ++w) {
          index_t index = h * (widths * anchors) + w * (anchors) + a;
          float x1 = boxes[n][index][0];
          float y1 = boxes[n][index][1];
          float x2 = boxes[n][index][2];
          float y2 = boxes[n][index][3];

          float dx1 = deltas[n][a * 4 + 0][h][w];
          float dy1 = deltas[n][a * 4 + 1][h][w];
          float dx2 = deltas[n][a * 4 + 2][h][w];
          float dy2 = deltas[n][a * 4 + 3][h][w];

          float pred_x1 = x1 + dx1;
          float pred_y1 = y1 + dy1;
          float pred_x2 = x2 + dx2;
          float pred_y2 = y2 + dy2;

          pred_x1 = std::max(std::min(pred_x1, im_width - 1.0f), 0.0f);
          pred_y1 = std::max(std::min(pred_y1, im_height - 1.0f), 0.0f);
          pred_x2 = std::max(std::min(pred_x2, im_width - 1.0f), 0.0f);
          pred_y2 = std::max(std::min(pred_y2, im_height - 1.0f), 0.0f);

          (*out_pred_boxes)[n][index][0] = pred_x1;
          (*out_pred_boxes)[n][index][1] = pred_y1;
          (*out_pred_boxes)[n][index][2] = pred_x2;
          (*out_pred_boxes)[n][index][3] = pred_y2;

          if (h >= real_height || w >= real_width) {
            (*out_pred_boxes)[n][index][4] = -1.0f;
          }
        }
      }
    }
  }
}

// filter box by set confidence to zero
// * height or width < rpn_min_size
inline void FilterBox(mshadow::Tensor<cpu, 3> *dets,
                      const float rpn_min_size,
                      const mshadow::Tensor<cpu, 2>& im_info,
                      bool filter_scales,
                      mshadow::Tensor<cpu, 2>& valid_ranges) {
  for (index_t n = 0; n < dets->size(0); n++) {
    float min_size = rpn_min_size * im_info[n][2];
    float valid_min = valid_ranges[n][0] * valid_ranges[n][0];
    float valid_max = valid_ranges[n][1] * valid_ranges[n][1];
    for (index_t i = 0; i < dets->size(1); i++) {
      float iw = (*dets)[n][i][2] - (*dets)[n][i][0] + 1.0f;
      float ih = (*dets)[n][i][3] - (*dets)[n][i][1] + 1.0f;
      if (iw < min_size || ih < min_size) {
        (*dets)[n][i][0] -= min_size / 2;
        (*dets)[n][i][1] -= min_size / 2;
        (*dets)[n][i][2] += min_size / 2;
        (*dets)[n][i][3] += min_size / 2;
        (*dets)[n][i][4] = -1.0f;
      } else if (filter_scales) {
        if (iw * ih < valid_min || iw * ih > valid_max)
          (*dets)[n][i][4] = -1.0f;
      }
    }
  }
}

}  // namespace proposal_v2_utils
}  // namespace op
}  // namespace mxnet

//=====================
// NMS Utils
//=====================
namespace mxnet {
namespace op {
namespace proposal_v2_utils {

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

}  // namespace proposal_v2_utils
}  // namespace op
}  // namespace mxnet


namespace mxnet {
namespace op {

template<typename xpu>
class ProposalOp_v2 : public Operator{
 public:
  explicit ProposalOp_v2(ProposalParam_v2 param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 4);
    CHECK_EQ(out_data.size(), 2);
    CHECK_GT(req.size(), 1);
    CHECK_EQ(req[proposal_v2::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    // Shape<3> fg_scores_shape = Shape3(in_data[proposal::kClsProb].shape_[1] / 2,
    //                                   in_data[proposal::kClsProb].shape_[2],
    //                                   in_data[proposal::kClsProb].shape_[3]);

    // real_t* foreground_score_ptr = in_data[proposal::kClsProb].dptr<real_t>()
    //                                 + fg_scores_shape.Size();
    Tensor<xpu, 4> scores = in_data[proposal_v2::kClsProb].get<cpu, 4, real_t>(s);
    Tensor<cpu, 4> bbox_deltas = in_data[proposal_v2::kBBoxPred].get<cpu, 4, real_t>(s);
    Tensor<cpu, 2> im_info = in_data[proposal_v2::kImInfo].get<cpu, 2, real_t>(s);
    Tensor<cpu, 2> valid_ranges = in_data[proposal_v2::kValidRanges].get<cpu, 2, real_t>(s);

    Tensor<cpu, 3> out = out_data[proposal_v2::kOut].get<cpu, 3, real_t>(s);
    Tensor<cpu, 3> out_score = out_data[proposal_v2::kScore].get<cpu, 3, real_t>(s);

    int nbatch = scores.size(0);
    int num_anchors = scores.size(1) / 2;
    int height = scores.size(2);
    int width = scores.size(3);
    int count = num_anchors * height * width;
    int rpn_pre_nms_top_n = (param_.rpn_pre_nms_top_n > 0) ? param_.rpn_pre_nms_top_n : count;
    rpn_pre_nms_top_n = std::min(rpn_pre_nms_top_n, count);
    int rpn_post_nms_top_n = std::min(param_.rpn_post_nms_top_n, rpn_pre_nms_top_n);

    int workspace_size = nbatch * (count * 5 + 2 * count + rpn_pre_nms_top_n * 5 + 3 * rpn_pre_nms_top_n);
    Tensor<cpu, 1> workspace = ctx.requested[proposal_v2::kTempSpace].get_space<cpu>(
      Shape1(workspace_size), s);
    int start = 0;
    Tensor<cpu, 3> workspace_proposals(workspace.dptr_ + start, Shape3(nbatch, count, 5));
    start += nbatch * count * 5;
    Tensor<cpu, 3> workspace_pre_nms(workspace.dptr_ + start, Shape3(2, nbatch, count));
    start += nbatch * 2 * count;
    Tensor<cpu, 3> workspace_ordered_proposals(workspace.dptr_ + start,
                                               Shape3(nbatch, rpn_pre_nms_top_n, 5));
    start += nbatch * rpn_pre_nms_top_n * 5;
    Tensor<cpu, 3> workspace_nms(workspace.dptr_ + start, Shape3(3, nbatch, rpn_pre_nms_top_n));
    start += nbatch * 3 * rpn_pre_nms_top_n;
    CHECK_EQ(workspace_size, start) << workspace_size << " " << start << std::endl;

    // Generate anchors
    std::vector<float> base_anchor(4);
    base_anchor[0] = 0.0;
    base_anchor[1] = 0.0;
    base_anchor[2] = param_.feature_stride - 1.0;
    base_anchor[3] = param_.feature_stride - 1.0;
    CHECK_EQ(num_anchors, param_.ratios.info.size() * param_.scales.info.size());
    std::vector<float> anchors;
    proposal_v2_utils::GenerateAnchors(base_anchor,
                                       param_.ratios.info,
                                       param_.scales.info,
                                       &anchors);
    for(int n = 0; n < nbatch; n++) {
      std::memcpy(workspace_proposals.dptr_ + n * 5 * count, &anchors[0], sizeof(float) * anchors.size());
    }

    // Enumerate all shifted anchors
    for (index_t n = 0; n < nbatch; ++n) {
      for (index_t i = 0; i < num_anchors; ++i) {
        for (index_t j = 0; j < height; ++j) {
          for (index_t k = 0; k < width; ++k) {
            index_t index = j * (width * num_anchors) + k * (num_anchors) + i;
            workspace_proposals[n][index][0] = workspace_proposals[n][i][0] + k * param_.feature_stride;
            workspace_proposals[n][index][1] = workspace_proposals[n][i][1] + j * param_.feature_stride;
            workspace_proposals[n][index][2] = workspace_proposals[n][i][2] + k * param_.feature_stride;
            workspace_proposals[n][index][3] = workspace_proposals[n][i][3] + j * param_.feature_stride;
            workspace_proposals[n][index][4] = scores[n][i + width * height * num_anchors][j][k];
          }
        }
      }
    }

    // prevent padded predictions

    if (param_.iou_loss) {
      proposal_v2_utils::IoUTransformInv(workspace_proposals, bbox_deltas, im_info, param_.feature_stride,
                                         &(workspace_proposals));
    } else {
      proposal_v2_utils::BBoxTransformInv(workspace_proposals, bbox_deltas,im_info, param_.feature_stride,
                                          &(workspace_proposals));
    }
    proposal_v2_utils::FilterBox(&workspace_proposals, param_.rpn_min_size, im_info,
                                 param_.filter_scales, valid_ranges);

    Tensor<cpu, 2> score = workspace_pre_nms[0];
    Tensor<cpu, 2> order = workspace_pre_nms[1];

    proposal_v2_utils::CopyScore(workspace_proposals,
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
      proposal_v2_utils::ReverseArgsort(score[n],
                                        &cur_order);
      proposal_v2_utils::ReorderProposals(workspace_proposals[n],
                                          cur_order,
                                          rpn_pre_nms_top_n,
                                          &cur_workspace_ordered_proposals);
      index_t out_size = 0;
      suppressed = 0;  // surprised!

      proposal_v2_utils::NonMaximumSuppression(cur_workspace_ordered_proposals,
                                               param_.threshold,
                                               rpn_post_nms_top_n,
                                               &cur_area,
                                               &cur_suppressed,
                                               &cur_keep,
                                               &out_size);

      // fill in output rois
      for (index_t i = 0; i < out.size(1); ++i) {
        if (i < out_size) {
          index_t index = cur_keep[i];
          for (index_t j = 0; j < 4; ++j) {
            out[n][i][j] =  cur_workspace_ordered_proposals[index][j];
          }
        } else {
          index_t index = cur_keep[i % out_size];
          for (index_t j = 0; j < 4; ++j) {
            out[n][i][j] = cur_workspace_ordered_proposals[index][j];
          }
        }
      }

      // fill in output score
      for (index_t i = 0; i < out_score.size(1); i++) {
        if (i < out_size) {
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
    CHECK_EQ(in_grad.size(), 4);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> gscores = in_grad[proposal_v2::kClsProb].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gbbox = in_grad[proposal_v2::kBBoxPred].get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> ginfo = in_grad[proposal_v2::kImInfo].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> granges = in_grad[proposal_v2::kValidRanges].get<xpu, 2, real_t>(s);

    // can not assume the grad would be zero
    Assign(gscores, req[proposal_v2::kClsProb], 0);
    Assign(gbbox, req[proposal_v2::kBBoxPred], 0);
    Assign(ginfo, req[proposal_v2::kImInfo], 0);
    Assign(granges, req[proposal_v2::kValidRanges], 0);
  }

 private:
  ProposalParam_v2 param_;
};  // class ProposalOp

template<>
Operator *CreateOp<cpu>(ProposalParam_v2 param) {
  return new ProposalOp_v2<cpu>(param);
}

Operator* ProposalProp_v2::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(ProposalParam_v2);

MXNET_REGISTER_OP_PROPERTY(_contrib_Proposal_v2, ProposalProp_v2)
.describe("Generate region proposals via RPN")
.add_argument("cls_prob", "NDArray-or-Symbol", "Probability of how likely proposal is object.")
.add_argument("bbox_pred", "NDArray-or-Symbol", "BBox Predicted deltas from anchors for proposals")
.add_argument("im_info", "NDArray-or-Symbol", "Image size and scale.")
.add_argument("valid_range", "NDArray-or-Symbol", "Valid scale ranges for image scale.")
.add_arguments(ProposalParam_v2::__FIELDS__());

}  // namespace op
}  // namespace mxnet
