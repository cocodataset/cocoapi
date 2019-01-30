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
 * \brief
 * \author Ziyang Zhou, Chenxia Han
*/

#include "./decodebbox-inl.h"

namespace mxnet {
namespace op {
namespace utils {

template <typename DType>
// bbox prediction and clip to the image borders
inline void BBoxTransformInv(const mshadow::Tensor<cpu, 3>& boxes,
                             const mshadow::Tensor<cpu, 3>& deltas,
                             const int nbatch,
                             const DType* means,
                             const DType* stds,
                             const bool class_agnostic,
                             const mshadow::Tensor<cpu, 2>& im_info,
                             mshadow::Tensor<cpu, 3> out_pred_boxes) {
  CHECK_GE(boxes.size(2), 4);
  CHECK_GE(out_pred_boxes.size(2), 4);
  int rois_num = boxes.size(1);
  int num_class = class_agnostic ? 1 : (deltas.size(2) / 4);
  for (int n = 0; n < nbatch; ++n) {
    for (int index = 0; index < rois_num; ++index) {
      for (int cls = 0; cls < num_class; ++cls) {
        float width = boxes[n][index][2] - boxes[n][index][0] + 1.0f;
        float height = boxes[n][index][3] - boxes[n][index][1] + 1.0f;
        float ctr_x = boxes[n][index][0] + 0.5f * (width - 1.0f);
        float ctr_y = boxes[n][index][1] + 0.5f * (height - 1.0f);

        int decode_cls = class_agnostic ? 1 : cls;
        float dx = deltas[n][index][decode_cls*4+0] * stds[0] + means[0];
        float dy = deltas[n][index][decode_cls*4+1] * stds[1] + means[1];
        float dw = deltas[n][index][decode_cls*4+2] * stds[2] + means[2];
        float dh = deltas[n][index][decode_cls*4+3] * stds[3] + means[3];

        float pred_ctr_x = dx * width + ctr_x;
        float pred_ctr_y = dy * height + ctr_y;
        float pred_w = exp(dw) * width;
        float pred_h = exp(dh) * height;

        float pred_x1 = pred_ctr_x - 0.5f * (pred_w - 1.0f);
        float pred_y1 = pred_ctr_y - 0.5f * (pred_h - 1.0f);
        float pred_x2 = pred_ctr_x + 0.5f * (pred_w - 1.0f);
        float pred_y2 = pred_ctr_y + 0.5f * (pred_h - 1.0f);
        
        pred_x1 = std::max(std::min(pred_x1, im_info[n][1] - 1.0f), 0.0f);
        pred_y1 = std::max(std::min(pred_y1, im_info[n][0] - 1.0f), 0.0f);
        pred_x2 = std::max(std::min(pred_x2, im_info[n][1] - 1.0f), 0.0f);
        pred_y2 = std::max(std::min(pred_y2, im_info[n][0] - 1.0f), 0.0f);

        out_pred_boxes[n][index][cls*4+0] = pred_x1;
        out_pred_boxes[n][index][cls*4+1] = pred_y1;
        out_pred_boxes[n][index][cls*4+2] = pred_x2;
        out_pred_boxes[n][index][cls*4+3] = pred_y2;
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
class DecodeBBoxOp : public Operator{
 public:
  explicit DecodeBBoxOp(DecodeBBoxParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(req[decodebbox::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 3> xpu_rois = in_data[decodebbox::kRois].get<xpu, 3, float>(s); // nbatch, num_roi, 4
    Tensor<xpu, 3> xpu_bbox_deltas = in_data[decodebbox::kBBoxPred].get<xpu, 3, float>(s); // nbatch, num_roi, 8
    Tensor<xpu, 2> xpu_im_info = in_data[decodebbox::kImInfo].get<xpu, 2, float>(s); // nbatch, 3(height, width, scale)

    Tensor<xpu, 3> xpu_out = out_data[decodebbox::kOut].get<xpu, 3, float>(s); // nbatch, num_roi, 4
    
    TensorContainer<cpu, 3> rois        (xpu_rois.shape_);
    TensorContainer<cpu, 3> bbox_deltas (xpu_bbox_deltas.shape_);
    TensorContainer<cpu, 2> im_info     (xpu_im_info.shape_);

    Copy(rois, xpu_rois, s);
    Copy(bbox_deltas, xpu_bbox_deltas, s);
    Copy(im_info, xpu_im_info, s);

    float bbox_mean[4];
    float bbox_std[4];
    bbox_mean[0] = param_.bbox_mean[0];
    bbox_mean[1] = param_.bbox_mean[1];
    bbox_mean[2] = param_.bbox_mean[2];
    bbox_mean[3] = param_.bbox_mean[3];
    bbox_std[0] = param_.bbox_std[0];
    bbox_std[1] = param_.bbox_std[1];
    bbox_std[2] = param_.bbox_std[2];
    bbox_std[3] = param_.bbox_std[3];

    int nbatch = rois.size(0);

    const bool class_agnostic = param_.class_agnostic;
    if (class_agnostic) {
      TensorContainer<cpu, 3> out(Shape3(rois.size(0), rois.size(1), 4), 0.f);
      utils::BBoxTransformInv(rois, bbox_deltas, nbatch, bbox_mean, bbox_std, class_agnostic, im_info, out);
      Copy(xpu_out, out, s);
    } else {
      TensorContainer<cpu, 3> out(xpu_bbox_deltas.shape_, 0.f);
      utils::BBoxTransformInv(rois, bbox_deltas, nbatch, bbox_mean, bbox_std, class_agnostic, im_info, out);
      Copy(xpu_out, out, s);
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
    Tensor<xpu, 3> grois        = in_grad[decodebbox::kRois].get<xpu, 3, float>(s);
    Tensor<xpu, 3> gbbox_deltas = in_grad[decodebbox::kBBoxPred].get<xpu, 3,float>(s);
    Tensor<xpu, 2> gim_info     = in_grad[decodebbox::kImInfo].get<xpu, 2,float>(s);

    grois = 0.f;
    gbbox_deltas = 0.f;
    gim_info = 0.f;
  }

 private:
  DecodeBBoxParam param_;
};  // class ProposalOp

template<>
Operator *CreateOp<cpu>(DecodeBBoxParam param) {
  return new DecodeBBoxOp<cpu>(param);
}
template<>
Operator *CreateOp<gpu>(DecodeBBoxParam param) {
  return new DecodeBBoxOp<gpu>(param);
}

Operator *DecodeBBoxProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                               std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_);
}

// Operator* DecodeBBoxProp::CreateOperator(Context ctx) const {
//   DO_BIND_DISPATCH(CreateOp, param_);
// }

DMLC_REGISTER_PARAMETER(DecodeBBoxParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_DecodeBBox, DecodeBBoxProp)
.describe("Decode rois using bbox predicted deltas")
.add_argument("rois", "NDArray-or-Symbol", "proposal")
.add_argument("bbox_pred", "NDArray-or-Symbol", "BBox Predicted deltas from anchors for proposals")
.add_argument("im_info", "NDArray-or-Symbol", "Image size and scale.")
.add_arguments(DecodeBBoxParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
