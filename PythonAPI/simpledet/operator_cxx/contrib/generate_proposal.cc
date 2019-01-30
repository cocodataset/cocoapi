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
 * \file generate_proposal.cc
 * \brief
 * \author Piotr Teterwak, Bing Xu, Jian Guo, Yuntao Chen, Yanghao Li
*/

#include "./generate_proposal-inl.h"

//============================
// Bounding Box Transform Utils
//============================
namespace mxnet {
namespace op {
namespace utils {

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
                      const mshadow::Tensor<cpu, 2>& im_info) {
  for (index_t n = 0; n < dets->size(0); n++) {
    float min_size = rpn_min_size * im_info[n][2];
    for (index_t i = 0; i < dets->size(1); i++) {
      float iw = (*dets)[n][i][2] - (*dets)[n][i][0] + 1.0f;
      float ih = (*dets)[n][i][3] - (*dets)[n][i][1] + 1.0f;
      if (iw < min_size || ih < min_size) {
        (*dets)[n][i][0] -= min_size / 2;
        (*dets)[n][i][1] -= min_size / 2;
        (*dets)[n][i][2] += min_size / 2;
        (*dets)[n][i][3] += min_size / 2;
        (*dets)[n][i][4] = -1.0f;
      }
    }
  }
}

}  // namespace utils
}  // namespace op
}  // namespace mxnet

//=====================
// SORT Utils
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

}  // namespace utils
}  // namespace op
}  // namespace mxnet


namespace mxnet {
namespace op {

template<typename xpu>
class GenProposalOp : public Operator{
 public:
  explicit GenProposalOp(GenProposalParam param) {
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
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(req[gen_proposal::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    // Shape<3> fg_scores_shape = Shape3(in_data[proposal::kClsProb].shape_[1] / 2,
    //                                   in_data[proposal::kClsProb].shape_[2],
    //                                   in_data[proposal::kClsProb].shape_[3]);

    // real_t* foreground_score_ptr = in_data[proposal::kClsProb].dptr<real_t>()
    //                                 + fg_scores_shape.Size();
    Tensor<xpu, 4> scores = in_data[gen_proposal::kClsProb].get<cpu, 4, real_t>(s);
    Tensor<cpu, 4> bbox_deltas = in_data[gen_proposal::kBBoxPred].get<cpu, 4, real_t>(s);
    Tensor<cpu, 2> im_info = in_data[gen_proposal::kImInfo].get<cpu, 2, real_t>(s);
    Tensor<cpu, 2> anchors = in_data[gen_proposal::kAnchor].get<cpu, 2, real_t>(s);

    Tensor<cpu, 3> out = out_data[gen_proposal::kOut].get<cpu, 3, real_t>(s);

    int nbatch = scores.size(0);
    int num_anchors = scores.size(1) / 2;
    int height = scores.size(2);
    int width = scores.size(3);
    int count = num_anchors * height * width;
    int rpn_pre_nms_top_n = (param_.rpn_pre_nms_top_n > 0) ? param_.rpn_pre_nms_top_n : count;
    rpn_pre_nms_top_n = std::min(rpn_pre_nms_top_n, count);

    int workspace_size = nbatch * (count * 5 + 2 * count + rpn_pre_nms_top_n * 5);
    Tensor<cpu, 1> workspace = ctx.requested[gen_proposal::kTempSpace].get_space<cpu>(
      Shape1(workspace_size), s);
    int start = 0;
    Tensor<cpu, 3> workspace_proposals(workspace.dptr_ + start, Shape3(nbatch, count, 5));
    start += nbatch * count * 5;
    Tensor<cpu, 3> workspace_pre_nms(workspace.dptr_ + start, Shape3(2, nbatch, count));
    start += nbatch * 2 * count;
    Tensor<cpu, 3> workspace_ordered_proposals(workspace.dptr_ + start,
                                               Shape3(nbatch, rpn_pre_nms_top_n, 5));
    start += nbatch * rpn_pre_nms_top_n * 5;
    CHECK_EQ(workspace_size, start) << workspace_size << " " << start << std::endl;


    // Enumerate all shifted anchors
    for (index_t n = 0; n < nbatch; ++n) {
      for (index_t i = 0; i < num_anchors; ++i) {
        for (index_t j = 0; j < height; ++j) {
          for (index_t k = 0; k < width; ++k) {
            index_t index = j * (width * num_anchors) + k * (num_anchors) + i;
            workspace_proposals[n][index][0] = anchors[index][0];
            workspace_proposals[n][index][1] = anchors[index][1];
            workspace_proposals[n][index][2] = anchors[index][2];
            workspace_proposals[n][index][3] = anchors[index][3];
            workspace_proposals[n][index][4] = scores[n][i + width * height * num_anchors][j][k];
          }
        }
      }
    }

    // prevent padded predictions
    if (param_.iou_loss) {
      utils::IoUTransformInv(workspace_proposals, bbox_deltas, im_info, param_.feature_stride,
                             &(workspace_proposals));
    } else {
      utils::BBoxTransformInv(workspace_proposals, bbox_deltas, im_info, param_.feature_stride,
                              &(workspace_proposals));
    }
    utils::FilterBox(&workspace_proposals, param_.rpn_min_size, im_info);

    Tensor<cpu, 2> score = workspace_pre_nms[0];
    Tensor<cpu, 2> order = workspace_pre_nms[1];

    utils::CopyScore(workspace_proposals,
                     &score,
                     &order);

    for(int n = 0; n < nbatch; n++) {
      Tensor<cpu, 1> cur_order = order[n];
      Tensor<cpu, 2> cur_workspace_ordered_proposals = workspace_ordered_proposals[n];
      utils::ReverseArgsort(score[n],
                            &cur_order);
      utils::ReorderProposals(workspace_proposals[n],
                              cur_order,
                              rpn_pre_nms_top_n,
                              &cur_workspace_ordered_proposals);

      // fill in output rois
      for (index_t i = 0; i < out.size(1); ++i) {
        if (i < rpn_pre_nms_top_n) {
          for (index_t j = 0; j < 5; ++j) {
            out[n][i][j] =  cur_workspace_ordered_proposals[i][j];
          }
        } else {
          for (index_t j = 0; j < 5; ++j) {
            out[n][i][j] = 0;
          }
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
    Tensor<xpu, 4> gscores = in_grad[gen_proposal::kClsProb].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gbbox = in_grad[gen_proposal::kBBoxPred].get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> ginfo = in_grad[gen_proposal::kImInfo].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> ganchors = in_grad[gen_proposal::kAnchor].get<xpu, 2, real_t>(s);

    // can not assume the grad would be zero
    Assign(gscores, req[gen_proposal::kClsProb], 0);
    Assign(gbbox, req[gen_proposal::kBBoxPred], 0);
    Assign(ginfo, req[gen_proposal::kImInfo], 0);
    Assign(ganchors, req[gen_proposal::kAnchor], 0);
  }

 private:
  GenProposalParam param_;
};  // class GenProposalOp

template<>
Operator *CreateOp<cpu>(GenProposalParam param) {
  return new GenProposalOp<cpu>(param);
}

Operator* GenProposalProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(GenProposalParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_GenProposal, GenProposalProp)
.describe("Generate region proposals via RPN")
.add_argument("cls_prob", "NDArray-or-Symbol", "Probability of how likely proposal is object.")
.add_argument("bbox_pred", "NDArray-or-Symbol", "BBox Predicted deltas from anchors for proposals")
.add_argument("im_info", "NDArray-or-Symbol", "Image size and scale.")
.add_argument("anchors", "NDArray-or-Symbol", "Generated Anchors.")
.add_arguments(GenProposalParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
